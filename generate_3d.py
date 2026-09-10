import os
import glob
from ultralytics import YOLO

# OpenCASCADE Imports
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakePolygon, BRepBuilderAPI_MakeFace
from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakePrism
from OCC.Core.gp import gp_Pnt, gp_Vec
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopAbs import TopAbs_COMPOUND, TopAbs_SOLID, TopAbs_SHELL, TopAbs_FACE, TopAbs_WIRE, TopAbs_EDGE, TopAbs_VERTEX
from OCC.Core.BRepTools import BRepTools_WireExplorer
from OCC.Core.BRep import BRep_Tool
from OCC.Core.TopoDS import topods, TopoDS_Iterator
from OCC.Core.GeomAbs import GeomAbs_Line, GeomAbs_Circle
from OCC.Core.BRepAdaptor import BRepAdaptor_Curve

# Local Project Imports
from cad_engine import build_ogc_multisolid, resolve_internal_overlaps
from db_engine import CadastreDatabaseEngine

# ==========================================
# 1. AI TO CAD EXTRUSION & OPTIMIZED EXTRACTOR
# ==========================================
def extract_vertices_from_wire(wire, face=None, tolerance=0.02):
    """
    Extracts ordered coordinates and strips redundant collinear vertices,
    with a strict safety floor to ensure polygons never drop below 4 points.

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

    raw_coords = []

    while ordered_vertices.More():
        vertex = ordered_vertices.CurrentVertex()
        pnt = BRep_Tool.Pnt(vertex)
        raw_coords.append((pnt.X(), pnt.Y(), pnt.Z()))
        ordered_vertices.Next()

    if not raw_coords:
        return []

    # Collinear Decimation filter
    optimized_coords = [raw_coords[0]]
    for i in range(1, len(raw_coords) - 1):
        prev = optimized_coords[-1]
        curr = raw_coords[i]
        nxt = raw_coords[i + 1]

        dx1, dy1 = curr[0] - prev[0], curr[1] - prev[1]
        dx2, dy2 = nxt[0] - curr[0], nxt[1] - curr[1]

        cross_product = abs(dx1 * dy2 - dy1 * dx2)
        if cross_product > tolerance:
            optimized_coords.append(curr)

    optimized_coords.append(raw_coords[-1])

    # SAFETY FLOOR: If decimation left us with fewer than 4 points, revert to raw coords to avoid PostGIS rejection
    if len(optimized_coords) < 4:
        optimized_coords = raw_coords

    formatted_coords = [f"{pt[0]:.4f} {pt[1]:.4f} {pt[2]:.4f}" for pt in optimized_coords]
    if formatted_coords and formatted_coords[0] != formatted_coords[-1]:
        formatted_coords.append(formatted_coords[0])

    return formatted_coords


def extrude_yolo_to_occ(poly_pts, z_base=0.0, height=3.0, scale_factor=0.05, offset_x=0.0, offset_y=0.0):
    """Converts raw 2D YOLO pixels into a 3D OpenCASCADE solid anchored to world coordinates."""
    polygon_builder = BRepBuilderAPI_MakePolygon()
    for pt in poly_pts:
        world_x = float(pt[0] * scale_factor) + offset_x
        world_y = float(pt[1] * scale_factor) + offset_y
        polygon_builder.Add(gp_Pnt(world_x, world_y, z_base))
    polygon_builder.Close()

    face = BRepBuilderAPI_MakeFace(polygon_builder.Wire()).Face()
    return BRepPrimAPI_MakePrism(face, gp_Vec(0, 0, height)).Shape()


def process_and_group_blueprint(image_path, weights_path="runs/segment/runs/sih_model_v2/weights/best.pt"):
    print(f"🏗️ Processing Blueprint: {image_path}")
    model = YOLO(weights_path)
    results = model(image_path)

    individual_rooms = []
    for r in results:
        if r.masks is None:
            continue
        for i, poly_pts in enumerate(r.masks.xy):
            if len(poly_pts) >= 3:
                try:
                    solid = extrude_yolo_to_occ(poly_pts, offset_x=30000.0, offset_y=10000.0)
                    individual_rooms.append(solid)
                except Exception as e:
                    print(f"⚠️ Failed to build solid: {e}")

    print(f"✅ Extruded {len(individual_rooms)} individual rooms.")

    # Run the Boolean Auto-Resolver to slice away fuzzy AI overlap
    clean_rooms = resolve_internal_overlaps(individual_rooms)

    print("🔗 Grouping perfectly flushed rooms into a single OGC MultiSolid...")
    return build_ogc_multisolid(clean_rooms)


# ==========================================
# 2. UNIVERSAL OGC ROUTER
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
    """TRUE 15-CLASS OGC ROUTER (Clean Collection Handler)"""
    shape_type = shape.ShapeType()

    if shape_type == TopAbs_COMPOUND:
        iterator = TopoDS_Iterator(shape)
        child_wkts = []

        while iterator.More():
            child_shape = iterator.Value()
            child_wkt = occ_to_wkt(child_shape)
            if child_wkt:
                child_wkts.append(child_wkt)
            iterator.Next()

        # Wrap all individual shapes cleanly into an OGC GeometryCollection
        return f"GEOMETRYCOLLECTION Z ({', '.join(child_wkts)})"

    elif shape_type in (TopAbs_SOLID, TopAbs_SHELL):
        face_explorer = TopExp_Explorer(shape, TopAbs_FACE)
        wkt_polygons, is_tin = [], True
        while face_explorer.More():
            face = topods.Face(face_explorer.Current())
            wire_explorer = TopExp_Explorer(face, TopAbs_WIRE)
            while wire_explorer.More():
                # FIX: pass `face` so wire traversal respects its Orientation() flag
                coords = extract_vertices_from_wire(topods.Wire(wire_explorer.Current()), face)
                if coords:
                    wkt_polygons.append(f"(({', '.join(coords)}))")
                    if len(coords) > 4:
                        is_tin = False
                wire_explorer.Next()
            face_explorer.Next()
        if is_tin and wkt_polygons:
            return f"TIN Z ({', '.join(wkt_polygons)})"
        return f"POLYHEDRALSURFACE Z ({', '.join(wkt_polygons)})"

    elif shape_type == TopAbs_FACE:
        face = topods.Face(shape)
        wire_explorer, rings, is_curved = TopExp_Explorer(face, TopAbs_WIRE), [], False
        while wire_explorer.More():
            wire = topods.Wire(wire_explorer.Current())
            if is_wire_curved(wire):
                is_curved = True
            # FIX: pass `face` here too, for the same reason as above
            coords = extract_vertices_from_wire(wire, face)
            if coords:
                rings.append(f"({', '.join(coords)})")
            wire_explorer.Next()
        if is_curved:
            return f"CURVEPOLYGON Z ({', '.join(rings)})"
        if len(rings) == 1 and rings[0].count(',') == 3:
            return f"TRIANGLE Z {rings[0]}"
        return f"POLYGON Z ({', '.join(rings)})"

    elif shape_type == TopAbs_WIRE:
        if is_wire_curved(topods.Wire(shape)):
            return f"COMPOUNDCURVE Z (...)"
        return f"LINESTRING Z (...)"

    elif shape_type == TopAbs_EDGE:
        edge = topods.Edge(shape)
        v1 = topods.Vertex(TopExp_Explorer(edge, TopAbs_VERTEX).Current())
        pnt1 = BRep_Tool.Pnt(v1)
        if BRepAdaptor_Curve(edge).GetType() == GeomAbs_Circle:
            return f"CIRCULARSTRING Z ({pnt1.X():.4f} {pnt1.Y():.4f} {pnt1.Z():.4f}, ...)"
        return f"LINESTRING Z ({pnt1.X():.4f} {pnt1.Y():.4f} {pnt1.Z():.4f}, ...)"

    elif shape_type == TopAbs_VERTEX:
        pnt = BRep_Tool.Pnt(topods.Vertex(shape))
        return f"POINT Z ({pnt.X():.4f} {pnt.Y():.4f} {pnt.Z():.4f})"

    return None


# ==========================================
# 3. MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    image_paths = glob.glob("data/floor_plans/images/val/*")
    if image_paths:
        print("\n--- 🚀 STARTING FULL PIPELINE ---")

        final_property = process_and_group_blueprint(image_paths[0])
        wkt_string = occ_to_wkt(final_property)

        try:
            db = CadastreDatabaseEngine()
            # Increment the version ID to prevent spatial overlap collision with previous inserts
            db.register_property("UNIT_AI_GEN_v6", wkt_string)
        except Exception as e:
            print(f"❌ Database Connection Failed: {e}")
    else:
        print("❌ No validation images found.")