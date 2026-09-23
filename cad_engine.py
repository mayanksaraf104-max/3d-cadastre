from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakePolygon, BRepBuilderAPI_MakeFace, BRepBuilderAPI_Sewing
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeSolid
from OCC.Core.TopoDS import topods, TopoDS_Compound, TopoDS_Shape
from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Common, BRepAlgoAPI_Cut
from OCC.Core.gp import gp_Pnt
from OCC.Core.BRepGProp import brepgprop
from OCC.Core.GProp import GProp_GProps
from OCC.Core.BRep import BRep_Builder, BRep_Tool
from OCC.Core.BRepCheck import BRepCheck_Analyzer
from OCC.Core.ShapeFix import ShapeFix_Shell, ShapeFix_Solid
from OCC.Core.TopAbs import TopAbs_SHELL, TopAbs_SOLID


def build_ogc_polyhedral_surface(list_of_3d_faces):
    """
    Constructs an irregular OGC PolyhedralSurface solid from measured 3D faces.

    Raw sewn shells from face points can have inconsistent face orientation;
    ShapeFix_Shell / ShapeFix_Solid repair ORIENTATION only. Nothing else is
    repaired or invented: the input must sew into exactly ONE closed, valid
    shell. Open geometry (free edges), non-manifold edges, several disconnected
    shells, or a shell/solid that fails validation is REJECTED with ValueError
    -- no fragment is picked, no gap is filled, and no partial shape is
    returned for registration.
    """
    sewing = BRepBuilderAPI_Sewing()
    for face_points in list_of_3d_faces:
        polygon_builder = BRepBuilderAPI_MakePolygon()
        for pt in face_points:
            polygon_builder.Add(gp_Pnt(pt[0], pt[1], pt[2]))
        polygon_builder.Close()
        face = BRepBuilderAPI_MakeFace(polygon_builder.Wire()).Face()
        sewing.Add(face)
    sewing.Perform()

    if sewing.NbFreeEdges() > 0:
        raise ValueError(f"Rejected: input faces do not close -- {sewing.NbFreeEdges()} free "
                         f"edge(s) remain after sewing (open shell / gaps). Nothing is filled.")
    if sewing.NbMultipleEdges() > 0:
        raise ValueError(f"Rejected: {sewing.NbMultipleEdges()} non-manifold edge(s) after sewing "
                         f"(edge shared by more than two faces).")
    sewn_shape = sewing.SewedShape()

    # sewn_shape can come back as a single Shell or a Compound of shells.
    shells = []
    if sewn_shape.ShapeType() == TopAbs_SHELL:
        shells.append(topods.Shell(sewn_shape))
    else:
        from OCC.Core.TopExp import TopExp_Explorer
        exp = TopExp_Explorer(sewn_shape, TopAbs_SHELL)
        while exp.More():
            shells.append(topods.Shell(exp.Current()))
            exp.Next()
    if len(shells) != 1:
        raise ValueError(f"Rejected: sewing produced {len(shells)} shell(s), expected exactly 1 "
                         f"connected shell. No fragment is selected on the caller's behalf.")

    # Repair face orientation on the shell itself.
    shell_fixer = ShapeFix_Shell(shells[0])
    shell_fixer.Perform()
    if shell_fixer.NbShells() != 1:
        raise ValueError(f"Rejected: shell repair left {shell_fixer.NbShells()} shell(s), "
                         f"expected exactly 1 connected shell.")
    fixed_shell = shell_fixer.Shell()
    if not BRep_Tool.IsClosed(fixed_shell) or not BRepCheck_Analyzer(fixed_shell).IsValid():
        raise ValueError("Rejected: shell is not closed and valid after orientation repair.")

    try:
        raw_solid = BRepBuilderAPI_MakeSolid(fixed_shell).Solid()
        # Repair the solid itself (orientation) as a second pass.
        solid_fixer = ShapeFix_Solid(raw_solid)
        solid_fixer.Perform()
        solid = solid_fixer.Solid()
    except Exception as e:
        raise ValueError(f"Rejected: geometry is not watertight after orientation repair: {e}") from e
    if not BRepCheck_Analyzer(solid).IsValid():
        raise ValueError("Rejected: resulting solid fails B-Rep validation.")
    return solid


def check_irregular_cadastre_overlap(shape_a, shape_b, tolerance=0.05):
    """Executes an OGC-compliant exact Boolean intersection."""
    intersection = BRepAlgoAPI_Common(shape_a, shape_b)
    intersection.Build()
    if intersection.IsDone():
        overlap_shape = intersection.Shape()
        props = GProp_GProps()
        brepgprop.VolumeProperties(overlap_shape, props)
        volume = props.Mass()
        if volume > tolerance:
            return True, round(volume, 4)
    return False, 0.0


def _is_valid_solid(shape):
    """True only for a non-null, B-Rep-valid TopAbs_SOLID."""
    return (isinstance(shape, TopoDS_Shape) and not shape.IsNull()
            and shape.ShapeType() == TopAbs_SOLID
            and BRepCheck_Analyzer(shape).IsValid())


def build_ogc_multisolid(list_of_solids):
    """
    Groups multiple OGC Solids into a single OGC MultiSolid.

    Only validated TopAbs_SOLID inputs are accepted. Anything else (open
    shell, compound, null or B-Rep-invalid shape) is REJECTED with ValueError
    naming the offending inputs -- never silently packaged or dropped.
    """
    bad = [i for i, solid in enumerate(list_of_solids, start=1) if not _is_valid_solid(solid)]
    if bad:
        raise ValueError(f"Rejected: input(s) {bad} are not valid TopAbs_SOLID B-Rep solids; "
                         f"nothing is packaged into the MultiSolid.")
    compound = TopoDS_Compound()
    builder = BRep_Builder()
    builder.MakeCompound(compound)
    for solid in list_of_solids:
        builder.Add(compound, solid)
    return compound


def _single_solid(shape):
    """The one TopAbs_SOLID inside `shape` (itself, or in a Boolean result compound); else None."""
    if shape.ShapeType() == TopAbs_SOLID:
        return shape
    from OCC.Core.TopExp import TopExp_Explorer
    found = []
    exp = TopExp_Explorer(shape, TopAbs_SOLID)
    while exp.More():
        found.append(topods.Solid(exp.Current()))
        exp.Next()
    return found[0] if len(found) == 1 else None


def resolve_internal_overlaps(solids_list, ai_segmentation_correction=False):
    """
    Measured B-Reps are authoritative and are NOT modified by default: exact
    3D intersection (check_irregular_cadastre_overlap) is used only to detect
    and REPORT overlapping pairs, and the solids are returned unchanged.

    Only when `ai_segmentation_correction=True` -- an explicit authorization
    that these solids are AI-segmentation output whose mask bleed may be
    corrected -- are overlaps resolved by sequential exact Boolean Subtraction
    (BRepAlgoAPI_Cut), each solid cut by those already locked. A Cut returns a
    compound, so each result is unwrapped to its single solid. A cut that
    fails, or leaves zero or several solids, raises ValueError: the unit is
    rejected, never kept as uncorrected geometry or packaged.
    """
    if not solids_list:
        return []

    if not ai_segmentation_correction:
        print("🔍 Checking measured solids for overlap (exact 3D intersection; "
              "geometry is preserved, nothing is cut)...")
        for i, solid_a in enumerate(solids_list):
            for j in range(i + 1, len(solids_list)):
                overlaps, volume = check_irregular_cadastre_overlap(solid_a, solids_list[j])
                if overlaps:
                    print(f"   🚨 Unit {i + 1} and Unit {j + 1} overlap by {volume} cubic meters "
                          f"-- reported only; measured geometry left unmodified.")
        return list(solids_list)

    print("🔪 AI-segmentation correction explicitly authorized: resolving internal AI mask "
          "bleeds with exact Boolean Subtraction...")
    resolved_solids = []

    for i, current_solid in enumerate(solids_list):
        for j, locked_solid in enumerate(resolved_solids):
            cut_algo = BRepAlgoAPI_Cut(current_solid, locked_solid)
            cut_algo.Build()
            if not cut_algo.IsDone():
                raise ValueError(f"Rejected: the authorized Boolean cut of Unit {i + 1} against "
                                 f"Unit {j + 1} failed; the uncorrected geometry is not kept.")
            current_solid = cut_algo.Shape()

        single = _single_solid(current_solid)
        if single is None:
            raise ValueError(f"Rejected: the authorized cut left Unit {i + 1} as zero or several "
                             f"solids, not exactly one.")
        resolved_solids.append(single)
        print(f"   ✔️ Perfected boundaries for Unit {i + 1}")

    return resolved_solids


def process_and_group_blueprint(image_path=None, measured_solids=None,
                                ai_segmentation_correction=False):
    """
    Groups independently MEASURED 3D B-Rep solids into an OGC MultiSolid
    compound, resolving overlaps with the exact 3D Boolean utilities above.

    Blueprint/AI 2D geometry NEVER creates 3D geometry here: `image_path` is
    accepted only so existing callers do not break, and is neither read nor
    used. There is no extraction, no face-from-outline, no extrusion, and no
    assumed base or top elevation. `measured_solids` must already be valid
    measured B-Rep solids (e.g. from the measured-surface reconstruction);
    if ANY input is not a valid TopAbs_SOLID the whole batch is rejected
    (ValueError) -- an invalid solid is never dropped to produce an
    incomplete MultiSolid.

    Overlaps between measured solids are detected and reported only; they are
    cut only if `ai_segmentation_correction=True` explicitly authorizes it.

    Returns the MultiSolid compound, or None when no valid measured 3D
    evidence is supplied -- no geometry is created in its absence.
    """
    if not measured_solids:
        print("⚠️ No independently measured 3D B-Rep evidence supplied -- no geometry created "
              "(blueprint/AI 2D outlines never create 3D geometry).")
        return None

    bad = [i for i, solid in enumerate(measured_solids, start=1) if not _is_valid_solid(solid)]
    if bad:
        raise ValueError(f"Rejected batch: measured input(s) {bad} are not valid B-Rep solids; "
                         f"no MultiSolid is produced.")

    resolved = resolve_internal_overlaps(measured_solids, ai_segmentation_correction)

    # Bundle into an OGC MultiSolid compound
    return build_ogc_multisolid(resolved)


if __name__ == "__main__":
    from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeBox

    print("📐 Testing OpenCASCADE Irregular Geometry Engine...")
    solid_a = BRepPrimAPI_MakeBox(gp_Pnt(0, 0, 0), gp_Pnt(2, 2, 2)).Shape()
    solid_b = BRepPrimAPI_MakeBox(gp_Pnt(1, 1, 1), gp_Pnt(3, 3, 3)).Shape()

    is_overlap, vol = check_irregular_cadastre_overlap(solid_a, solid_b)
    if is_overlap:
        print(f"🚨 ILLEGAL ENCROACHMENT DETECTED: {vol} cubic meters of overlap.")
    else:
        print("✅ Boundaries are clear. No spatial conflict.")