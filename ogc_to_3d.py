from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopAbs import TopAbs_FACE, TopAbs_WIRE, TopAbs_EDGE, TopAbs_VERTEX
from OCC.Core.BRepTools import BRepTools_WireExplorer
from OCC.Core.BRep import BRep_Tool
from OCC.Core.TopoDS import topods
from OCC.Core.GeomAbs import GeomAbs_Line, GeomAbs_Circle
from OCC.Core.BRepAdaptor import BRepAdaptor_Curve
from db_engine import CadastreDatabaseEngine

def occ_to_wkt(shape, geometry_type="Solid"):
    """
    Dynamically routes OpenCASCADE geometry into OGC MultiPolygon Z format 
    for full PostGIS spatial indexing and 3D intersection compatibility.
    """
    if geometry_type == "Solid" or geometry_type == "MultiSolid":
        face_explorer = TopExp_Explorer(shape, TopAbs_FACE)
        wkt_polygons = []
        
        while face_explorer.More():
            face = topods.Face(face_explorer.Current())
            wire_explorer = TopExp_Explorer(face, TopAbs_WIRE)
            
            while wire_explorer.More():
                wire = topods.Wire(wire_explorer.Current())
                ordered_vertices = BRepTools_WireExplorer(wire)
                coords = []
                
                while ordered_vertices.More():
                    vertex = ordered_vertices.CurrentVertex()
                    pnt = BRep_Tool.Pnt(vertex)
                    coords.append(f"{pnt.X():.4f} {pnt.Y():.4f} {pnt.Z():.4f}")
                    ordered_vertices.Next()
                    
                if coords:
                    coords.append(coords[0]) # Close the loop
                    wkt_polygons.append(f"(({', '.join(coords)}))")
                    
                wire_explorer.Next()
            face_explorer.Next()
            
        # Switch from PolyhedralSurface to MultiPolygon Z for universal PostGIS ST_Subdivide support
        return f"MULTIPOLYGON Z ({', '.join(wkt_polygons)})"

    elif geometry_type == "Edge":
        edge = topods.Edge(shape)
        curve_adaptor = BRepAdaptor_Curve(edge)
        curve_type = curve_adaptor.GetType()
        
        v1 = topods.Vertex(TopExp_Explorer(edge, TopAbs_VERTEX).Current())
        pnt1 = BRep_Tool.Pnt(v1)
        
        if curve_type == GeomAbs_Circle:
            return f"CIRCULARSTRING Z ({pnt1.X():.4f} {pnt1.Y():.4f} {pnt1.Z():.4f}, ...)" 
        else:
            return f"LINESTRING Z ({pnt1.X():.4f} {pnt1.Y():.4f} {pnt1.Z():.4f}, ...)"

    elif geometry_type == "Point":
        vertex = topods.Vertex(shape)
        pnt = BRep_Tool.Pnt(vertex)
        return f"POINT Z ({pnt.X():.4f} {pnt.Y():.4f} {pnt.Z():.4f})"
        
    else:
        raise ValueError(f"Geometry type {geometry_type} routing not implemented.")

if __name__ == "__main__":
    import glob
    from ai_to_ogc import extract_ogc_boundaries
    from cad_engine import process_and_group_blueprint, resolve_internal_overlaps
    from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakePolygon, BRepBuilderAPI_MakeFace
    from OCC.Core.gp import gp_Pnt, gp_Vec
    from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakePrism

    image_paths = glob.glob("data/floor_plans/images/val/*")
    if image_paths:
        print("\n--- 🚀 STARTING OGC -> POSTGIS PIPELINE ---")
        image_path = image_paths[0]
        
        # 1. Extract 2D boundaries via AI
        raw_units = extract_ogc_boundaries(image_path)
        if raw_units:
            solids = []
            for unit in raw_units:
                poly = unit["polygon"]
                coords = list(poly.exterior.coords)
                makepoly = BRepBuilderAPI_MakePolygon()
                for x, y in coords[:-1]:
                    makepoly.Add(gp_Pnt(float(x), float(y), 0.0))
                try:
                    wire = makepoly.Wire()
                    face = BRepBuilderAPI_MakeFace(wire).Face()
                    prism = BRepPrimAPI_MakePrism(face, gp_Vec(0, 0, 3.0))
                    solids.append(prism.Shape())
                except Exception as e:
                    print(f"⚠️ Failed to extrude solid: {e}")

            # 2. Resolve internal overlaps using OpenCASCADE Boolean cuts
            resolved_solids = resolve_internal_overlaps(solids)
            
            # 3. Connect to PostGIS and register each unit individually
            try:
                db = CadastreDatabaseEngine()
                
                for i, solid in enumerate(resolved_solids, start=1):
                    unit_id = f"AI_Unit_{i}"
                    wkt_string = occ_to_wkt(solid, geometry_type="Solid")
                    
                    # Only register if the shape actually contains vertices
                    if wkt_string and "()" not in wkt_string:
                        db.register_property(unit_id, wkt_string)
                    else:
                        print(f"⚠️ Skipping {unit_id}: Shape was completely resolved/cut away by overlaps.\n")
                    
                print("🎉 All OpenCASCADE solids successfully registered into PostGIS!")
                
            except Exception as e:
                print(f"❌ Database Registration Failed: {e}")
        else:
            print("❌ No geometries extracted from image.")
    else:
        print("❌ No validation images found.")