import os
import glob
from ultralytics import YOLO

# OpenCASCADE Imports (inspection only: nothing in this module constructs geometry)
from OCC.Core.BRepCheck import BRepCheck_Analyzer
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopAbs import TopAbs_COMPOUND, TopAbs_SOLID, TopAbs_SHELL, TopAbs_FACE, TopAbs_WIRE, TopAbs_EDGE
from OCC.Core.BRepTools import BRepTools_WireExplorer, breptools
from OCC.Core.BRep import BRep_Tool
from OCC.Core.TopoDS import topods, TopoDS_Iterator
from OCC.Core.GeomAbs import GeomAbs_Line, GeomAbs_Plane
from OCC.Core.BRepAdaptor import BRepAdaptor_Curve, BRepAdaptor_Surface

# Local Project Imports
from db_engine import CadastreDatabaseEngine
import config

# ==========================================
# 1. MEASURED 3D INPUT (YOLO = 2D REFERENCE ONLY) & LOSSLESS EXTRACTOR
# ==========================================
def extract_vertices_from_wire(wire, face=None):
    """
    Extracts the ordered, closed vertex ring of a wire, losslessly: every
    3D vertex is kept (no XY-based or any other decimation) and coordinates
    are written at full float precision (repr round-trips), never rounded.
    A wire with fewer than 3 vertices is rejected, not dropped or patched.

    FIX: pass the parent `face` through to BRepTools_WireExplorer. Without
    it, the explorer walks the wire's own raw edge order and ignores the
    face's Orientation() flag. If ShapeFix_Shell/ShapeFix_Solid repaired a
    face's winding by flipping that flag (the common, cheap repair) rather
    than physically re-ordering its vertices, this function would silently
    export the *pre-fix* vertex order anyway -- so a small number of faces
    that were topologically fixed upstream still come out with inconsistent
    winding in the exported WKT, which is exactly what PostGIS/SFCGAL then
    rejects. Passing `face` makes the explorer respect that flag.
    """
    if face is not None:
        ordered_vertices = BRepTools_WireExplorer(wire, face)
    else:
        ordered_vertices = BRepTools_WireExplorer(wire)

    coords = []
    while ordered_vertices.More():
        pnt = BRep_Tool.Pnt(ordered_vertices.CurrentVertex())
        coords.append(f"{pnt.X()!r} {pnt.Y()!r} {pnt.Z()!r}")
        ordered_vertices.Next()

    if len(coords) < 3:
        raise ValueError(f"Degenerate wire: {len(coords)} vertices (need >= 3).")
    if coords[0] != coords[-1]:
        coords.append(coords[0])
    return coords


def process_and_group_blueprint(image_path, measured_rooms, weights_path=config.YOLO_INFERENCE_MODEL_PATH):
    """
    `measured_rooms`: iterable of independently measured / validated 3D OCC
    B-Rep solids (the ONLY source of room geometry). If none is supplied, or
    any is missing / not a valid solid, this raises; nothing is generated.

    YOLO masks are 2D semantic/reference correspondence only: they are
    returned as `reference_masks` and are never converted into geometry (no
    scale, offset, base Z, height or extrusion). Measured solids are returned
    exactly as supplied: not grouped, fused, trimmed, sliced or repaired.
    db_engine registers ONE POLYHEDRALSURFACE Z per property, so each solid
    is serialized and registered on its own (see main).

    Returns (measured_solids, reference_masks).
    """
    rooms = list(measured_rooms or [])
    if not rooms:
        raise ValueError(
            "No measured 3D geometry supplied. Blueprint/YOLO polygons are not "
            "geometry and are never extruded.")
    for i, room in enumerate(rooms):
        if room is None or room.IsNull() or room.ShapeType() != TopAbs_SOLID:
            raise ValueError(f"measured room {i}: missing or not a B-Rep solid.")
        if not BRepCheck_Analyzer(room).IsValid():
            raise ValueError(f"measured room {i}: invalid B-Rep solid.")

    print(f"🏗️ Processing Blueprint (2D reference): {image_path}")
    model = YOLO(weights_path)
    results = model(image_path)
    reference_masks = [poly for r in results if r.masks is not None for poly in r.masks.xy]
    print(f"🗂️ {len(reference_masks)} YOLO masks kept as 2D reference only (not geometry).")

    return rooms, reference_masks


# ==========================================
# 2. OGC ROUTER (POLYHEDRALSURFACE Z ONLY)
# ==========================================
def is_wire_curved(wire):
    edge_explorer = TopExp_Explorer(wire, TopAbs_EDGE)
    while edge_explorer.More():
        edge = topods.Edge(edge_explorer.Current())
        if BRepAdaptor_Curve(edge).GetType() != GeomAbs_Line:
            return True
        edge_explorer.Next()
    return False


def occ_to_wkt(shape):
    """
    Lossless OCC B-Rep -> WKT for the POLYHEDRALSURFACE Z contract only.

    Every face is emitted as one polygon with its outer wire first, then its
    inner wires (holes), at full coordinate precision. Anything that cannot
    be written exactly is REJECTED (ValueError), never approximated, elided
    ("...") or fabricated: curved edges, non-planar faces, solids with more
    than one shell, degenerate wires, a compound of more than one shape, and
    any shape that is not a solid or shell (bare faces, wires, edges,
    vertices). Output is always a single POLYHEDRALSURFACE Z, never a
    GEOMETRYCOLLECTION.
    """
    if shape is None or shape.IsNull():
        raise ValueError("Missing geometry: nothing to convert.")
    shape_type = shape.ShapeType()

    if shape_type == TopAbs_COMPOUND:
        # A compound is only a container. Exactly one child is serialized as
        # itself; several independent solids are REJECTED (never emitted as a
        # GEOMETRYCOLLECTION, merged, unioned or otherwise combined): register
        # each solid on its own.
        iterator = TopoDS_Iterator(shape)
        children = []
        while iterator.More():
            children.append(iterator.Value())
            iterator.Next()

        if len(children) != 1:
            raise ValueError(
                f"Compound holds {len(children)} shapes; the POLYHEDRALSURFACE Z "
                f"contract takes exactly one solid per registration. Register "
                f"each measured solid separately.")
        return occ_to_wkt(children[0])

    if shape_type in (TopAbs_SOLID, TopAbs_SHELL):
        if shape_type == TopAbs_SOLID:
            shell_count, shells = 0, TopExp_Explorer(shape, TopAbs_SHELL)
            while shells.More():
                shell_count += 1
                shells.Next()
            if shell_count != 1:
                raise ValueError(
                    f"Solid has {shell_count} shells; only a single-shell "
                    f"POLYHEDRALSURFACE Z is supported.")

        face_explorer = TopExp_Explorer(shape, TopAbs_FACE)
        wkt_faces = []
        while face_explorer.More():
            face = topods.Face(face_explorer.Current())
            if BRepAdaptor_Surface(face).GetType() != GeomAbs_Plane:
                raise ValueError("Non-planar face: not representable exactly as a polygon.")

            outer = breptools.OuterWire(face)
            wires = [outer]
            wire_explorer = TopExp_Explorer(face, TopAbs_WIRE)
            while wire_explorer.More():
                wire = topods.Wire(wire_explorer.Current())
                if not wire.IsSame(outer):
                    wires.append(wire)
                wire_explorer.Next()

            rings = []
            for wire in wires:
                if is_wire_curved(wire):
                    raise ValueError("Curved edge: not representable exactly in POLYHEDRALSURFACE Z.")
                # pass `face` so wire traversal respects its Orientation() flag
                rings.append(f"({', '.join(extract_vertices_from_wire(wire, face))})")
            wkt_faces.append(f"({', '.join(rings)})")
            face_explorer.Next()

        if not wkt_faces:
            raise ValueError("Solid/shell has no faces.")
        return f"POLYHEDRALSURFACE Z ({', '.join(wkt_faces)})"

    raise ValueError(
        f"Unsupported shape type {shape_type} for the POLYHEDRALSURFACE Z contract.")


# ==========================================
# 3. MAIN EXECUTION
# ==========================================
# Usage: python generate_3d.py MEASURED.step [MORE.step ...]
# Room geometry comes ONLY from the supplied measured 3D STEP solids, which
# must already be in the cadastre CRS (nothing is reprojected or moved here).
# Requires CADASTRE_STATE_CODE and CADASTRE_DISTRICT_CODE (recorded in each
# ULPIN by db_engine.register_property). Each solid is registered separately.
if __name__ == "__main__":
    import sys
    from OCC.Core.STEPControl import STEPControl_Reader
    from OCC.Core.IFSelect import IFSelect_RetDone

    def load_measured_solids(path):
        reader = STEPControl_Reader()
        if reader.ReadFile(path) != IFSelect_RetDone:
            raise SystemExit(f"Cannot read STEP file: {path}")
        reader.TransferRoots()
        explorer = TopExp_Explorer(reader.OneShape(), TopAbs_SOLID)
        solids = []
        while explorer.More():
            solids.append(topods.Solid(explorer.Current()))
            explorer.Next()
        return solids

    image_paths = glob.glob("data/floor_plans/images/val/*")
    step_paths = sys.argv[1:]
    missing = [k for k in ("CADASTRE_STATE_CODE", "CADASTRE_DISTRICT_CODE") if not os.environ.get(k)]
    if not image_paths:
        print("❌ No validation images found.")
    elif not step_paths:
        print("❌ No measured 3D geometry supplied. Usage: python generate_3d.py MEASURED.step [...]")
    elif missing:
        print(f"❌ Set {', '.join(missing)}: they are recorded in each ULPIN and have no default.")
    else:
        print("\n--- 🚀 STARTING FULL PIPELINE ---")

        measured = [s for p in step_paths for s in load_measured_solids(p)]
        rooms, _reference_masks = process_and_group_blueprint(image_paths[0], measured)
        # Serialize every solid first: one unsupported solid rejects the batch
        # before anything is registered.
        wkt_strings = [occ_to_wkt(room) for room in rooms]

        # Increment the version ID to prevent spatial overlap collision with previous inserts
        base_id = "UNIT_AI_GEN_v6"
        try:
            with CadastreDatabaseEngine(use_pool=False) as db:
                for n, wkt_string in enumerate(wkt_strings, start=1):
                    unit_id = base_id if len(wkt_strings) == 1 else f"{base_id}_P{n}"
                    ulpin = db.register_property(
                        unit_id=unit_id,
                        ogc_3d_wkt=wkt_string,
                        state_code=os.environ["CADASTRE_STATE_CODE"],
                        district_code=os.environ["CADASTRE_DISTRICT_CODE"],
                    )
                    if ulpin is None:
                        print(f"❌ {unit_id} was not registered (rejected -- see the log above).")
        except Exception as e:
            print(f"❌ Database error: {e}")