import os
import psycopg2
import trimesh
import numpy as np
import re

def export_postgis_to_glb(output_filename="approved_cadastre.glb"):
    print("🌍 Connecting to PostGIS to query registered legal entities...")
    try:
        # FIX: previously hardcoded (dbname="cadastre_db", user="postgres",
        # password="mayank9431", host="localhost"), unlike every other
        # file in this pipeline (db_engine.py, lidar_indexer.py,
        # z_engine.py), which all read PG_DBNAME / PG_USER / PG_PASSWORD /
        # PG_HOST from the environment. If you ever point the rest of the
        # pipeline at a different host, user, password, or database via
        # those env vars, export_ledger.py would silently keep connecting
        # to the old default and either fail outright or export an empty/
        # stale ledger, with no obvious link back to the env var change.
        conn = psycopg2.connect(
            dbname=os.environ.get("PG_DBNAME", "cadastre_db"),
            user=os.environ.get("PG_USER", "postgres"),
            password=os.environ.get("PG_PASSWORD", "mayank9431"),
            host=os.environ.get("PG_HOST", "localhost"),
        )
        cursor = conn.cursor()
        
        cursor.execute("SELECT unit_id, ST_AsText(boundary) FROM property_registry;")
        rows = cursor.fetchall()
        
        if not rows:
            print("⚠️ No registered properties found in the database ledger.")
            return

        scene = trimesh.Scene()
        
        for unit_id, wkt_str in rows:
            print(f"📦 Extracting True 3D geometry for legal title: {unit_id}")
            
            # Find all coordinate blocks
            faces_match = re.findall(r'\(\((.*?)\)\)', wkt_str)
            if not faces_match:
                print(f"   ⚠️ Could not parse 3D mesh for {unit_id}")
                continue
                
            vertices = []
            faces = []
            vertex_map = {}
            v_idx = 0
            
            # --- THE FIX: GLOBAL SHIFT FOR WEBGL PRECISION ---
            global_offset = None 
            
            for face_str in faces_match:
                pts = face_str.split(',')
                face_v_indices = []
                
                for pt_str in pts[:3]:
                    clean_pt = pt_str.replace('(', '').replace(')', '').strip()
                    raw_coords = tuple(map(float, clean_pt.split()))
                    
                    # Capture the very first point to use as our (0,0,0) anchor
                    if global_offset is None:
                        global_offset = (raw_coords[0], raw_coords[1], 0.0)
                        
                    # Subtract the massive UTM numbers so WebGL renders them near zero
                    coords = (
                        raw_coords[0] - global_offset[0],
                        raw_coords[1] - global_offset[1],
                        raw_coords[2]
                    )
                    
                    if coords not in vertex_map:
                        vertex_map[coords] = v_idx
                        vertices.append(coords)
                        v_idx += 1
                        
                    face_v_indices.append(vertex_map[coords])
                    
                faces.append(face_v_indices)
            
            mesh = trimesh.Trimesh(vertices=np.array(vertices), faces=np.array(faces))
            
            # 🎨 Color coding based on Vertical Tier
            if "METRO" in unit_id:
                mesh.visual.face_colors = [100, 100, 255, 200]  # Translucent Blue Tunnel
            elif "Floor_2" in unit_id:
                mesh.visual.face_colors = [255, 165, 0, 200]    # Translucent Orange 2nd Floor
            else:
                mesh.visual.face_colors = [100, 255, 100, 255]  # Solid Green Surface
            
            scene.add_geometry(mesh, node_name=unit_id)

        scene.export(output_filename)
        print(f"✨ Successfully exported legal cadastre ledger to WebGL file: {output_filename}")
        
    except Exception as e:
        print(f"❌ Export Failed: {e}")
    finally:
        if 'cursor' in locals(): cursor.close()
        if 'conn' in locals(): conn.close()

if __name__ == "__main__":
    export_postgis_to_glb()