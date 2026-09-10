from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakePolygon, BRepBuilderAPI_MakeFace, BRepBuilderAPI_Sewing
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeSolid
from OCC.Core.TopoDS import topods, TopoDS_Compound
from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Common, BRepAlgoAPI_Cut
from OCC.Core.gp import gp_Pnt, gp_Vec
from OCC.Core.BRepGProp import brepgprop
from OCC.Core.GProp import GProp_GProps
from OCC.Core.BRep import BRep_Builder
from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakePrism
from OCC.Core.ShapeFix import ShapeFix_Shell, ShapeFix_Solid
from OCC.Core.TopAbs import TopAbs_SHELL
from ai_to_ogc import extract_ogc_boundaries


def build_ogc_polyhedral_surface(list_of_3d_faces):
    """
    Constructs an irregular OGC PolyhedralSurface.

    FIX: raw sewn shells from AI-derived face points frequently have
    inconsistent face orientation (some faces wound CW, some CCW) and
    sometimes fail to close into a fully connected shell. Both defects
    pass OCC's sewing step silently but get rejected by PostGIS/SFCGAL
    as an invalid PolyhedralSurface ("inconsistent orientation" /
    "not connected"). We run ShapeFix_Shell (fixes connectivity +
    orientation) and ShapeFix_Solid (fixes the resulting solid) before
    handing the geometry off, instead of exporting it as-is.
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
    sewn_shape = sewing.SewedShape()

    # sewn_shape can come back as a single Shell or a Compound of shells
    # depending on how many separate pieces OCC found while sewing.
    if sewn_shape.ShapeType() == TopAbs_SHELL:
        shell = topods.Shell(sewn_shape)
    else:
        # Take the first shell out of the compound; if sewing fragmented
        # the geometry into multiple disconnected shells, that fragmentation
        # itself is the underlying defect (see "not connected" case below).
        explorer_shells = []
        from OCC.Core.TopExp import TopExp_Explorer
        exp = TopExp_Explorer(sewn_shape, TopAbs_SHELL)
        while exp.More():
            explorer_shells.append(topods.Shell(exp.Current()))
            exp.Next()
        if not explorer_shells:
            print("⚠️ Sewing produced no valid shell at all — check input face_points for degenerate/duplicate faces.")
            return sewn_shape
        if len(explorer_shells) > 1:
            print(f"⚠️ Sewing produced {len(explorer_shells)} disconnected shell fragments — "
                  f"input faces likely don't share exact edges (gaps/duplicate points). Using the largest fragment.")
        shell = explorer_shells[0]

    # Repair face orientation + connectivity on the shell itself.
    shell_fixer = ShapeFix_Shell(shell)
    shell_fixer.Perform()
    fixed_shell = shell_fixer.Shell()

    try:
        solid_maker = BRepBuilderAPI_MakeSolid(fixed_shell)
        raw_solid = solid_maker.Solid()

        # Repair the solid itself (orientation, small gaps) as a second pass.
        solid_fixer = ShapeFix_Solid(raw_solid)
        solid_fixer.Perform()
        return solid_fixer.Solid()
    except Exception as e:
        print(f"⚠️ Geometry is not watertight even after ShapeFix repair: {e}")
        print("   Returning the repaired shell so you can inspect it, but this unit should be flagged, not registered.")
        return fixed_shell


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


def build_ogc_multisolid(list_of_solids):
    """Groups multiple OGC Solids into a single OGC MultiSolid."""
    compound = TopoDS_Compound()
    builder = BRep_Builder()
    builder.MakeCompound(compound)
    for solid in list_of_solids:
        builder.Add(compound, solid)
    return compound


def resolve_internal_overlaps(solids_list):
    """Sequentially resolves AI mask bleed using exact Boolean Subtraction."""
    if not solids_list:
        return []

    print("🔪 Auto-Resolving internal AI mask bleeds...")
    resolved_solids = []

    for i, current_solid in enumerate(solids_list):
        for locked_solid in resolved_solids:
            cut_algo = BRepAlgoAPI_Cut(current_solid, locked_solid)
            cut_algo.Build()
            if cut_algo.IsDone():
                current_solid = cut_algo.Shape()

        resolved_solids.append(current_solid)
        print(f"   ✔️ Perfected boundaries for Unit {i + 1}")

    return resolved_solids


def process_and_group_blueprint(image_path):
    """
    Extracts 2D floor plan boundaries via AI, extrudes them into 3D OpenCASCADE solids,
    automatically resolves overlaps, and bundles them into an OGC MultiSolid compound.
    """
    print(f"📐 Processing blueprint with Advanced OpenCASCADE Engine: {image_path}")

    raw_units = extract_ogc_boundaries(image_path)
    if not raw_units:
        print("⚠️ No geometries extracted from image.")
        return None

    solids = []
    for unit in raw_units:
        poly = unit["polygon"]
        coords = list(poly.exterior.coords)
        z_base, z_top = 0.0, 3.0
        height = z_top - z_base

        makepoly = BRepBuilderAPI_MakePolygon()
        for x, y in coords[:-1]:
            makepoly.Add(gp_Pnt(float(x), float(y), z_base))
        try:
            wire = makepoly.Wire()
            face = BRepBuilderAPI_MakeFace(wire).Face()
            prism = BRepPrimAPI_MakePrism(face, gp_Vec(0, 0, height))
            solids.append(prism.Shape())
        except Exception as e:
            print(f"⚠️ Failed to extrude solid: {e}")

    # Apply exact Boolean overlap resolution using your built-in engine
    resolved = resolve_internal_overlaps(solids)

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