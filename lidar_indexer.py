import os
import laspy
import psycopg2
from shapely.geometry import box

# FIX: must match CADASTRE_SRID in db_engine.py / z_engine.py. This table
# was previously always created (and every tile inserted) as SRID 4326,
# regardless of what SRID the rest of the pipeline actually uses. Once
# you populate real tiles here, z_engine.get_intersecting_lidar_tiles()
# would hit the exact same "mixed SRID geometries" error against this
# table that property_registry/property_spatial_shards hit.
CADASTRE_SRID = int(os.environ.get("CADASTRE_SRID", "32610"))


def index_lidar_directory(lidar_dir="data/raw_lidar"):
    print(f"🌍 Connecting to PostGIS to build Spatial LiDAR Index...")
    conn = psycopg2.connect(
        dbname=os.environ.get("PG_DBNAME", "cadastre_db"),
        user=os.environ.get("PG_USER", "postgres"),
        password=os.environ.get("PG_PASSWORD", "mayank9431"),
        host=os.environ.get("PG_HOST", "localhost"),
    )
    cursor = conn.cursor()

    # Create the tile tracking table and a high-speed GiST Spatial Index.
    # SRID now driven by CADASTRE_SRID instead of a hardcoded 4326.
    cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS lidar_tile_index (
            id SERIAL PRIMARY KEY,
            file_path TEXT UNIQUE,
            tile_boundary GEOMETRY(POLYGON, {CADASTRE_SRID})
        );
        CREATE INDEX IF NOT EXISTS lidar_spatial_idx ON lidar_tile_index USING GIST (tile_boundary);
    """)

    # Same migration safety net as db_engine.py: if this table already
    # existed from an earlier run (hardcoded 4326), CREATE TABLE IF NOT
    # EXISTS above is a no-op and the column stays stuck at the old SRID.
    # Retag it (not reproject -- these were always real-world LAS header
    # bounds, never actual WGS84 lon/lat) if it doesn't match.
    cursor.execute("SELECT Find_SRID('public', 'lidar_tile_index', 'tile_boundary');")
    current_srid = cursor.fetchone()[0]
    if current_srid != CADASTRE_SRID:
        print(f"   🔧 Migrating lidar_tile_index.tile_boundary: stored SRID {current_srid} -> {CADASTRE_SRID}")
        cursor.execute(f"""
            ALTER TABLE lidar_tile_index
            ALTER COLUMN tile_boundary TYPE geometry(POLYGON, {CADASTRE_SRID})
            USING ST_SetSRID(tile_boundary, {CADASTRE_SRID});
        """)
    conn.commit()

    if not os.path.exists(lidar_dir):
        os.makedirs(lidar_dir)
        print(f"⚠️ Directory {lidar_dir} created. Please add .laz files here.")
        cursor.close()
        conn.close()
        return

    print(f"🔍 Scanning directory {lidar_dir} for LiDAR tiles...")

    for filename in os.listdir(lidar_dir):
        if filename.endswith(".laz") or filename.endswith(".las"):
            file_path = os.path.join(lidar_dir, filename)
            abs_path = os.path.abspath(file_path).replace("\\", "/")  # Normalize for Windows/DB

            # Read ONLY the header metadata to extract the OGC bounding box
            with laspy.open(file_path) as fh:
                header = fh.header
                minx, miny, _ = header.mins
                maxx, maxy, _ = header.maxs

                # Create a flat 2D footprint for spatial intersection
                tile_poly = box(minx, miny, maxx, maxy)
                wkt_geom = f"SRID={CADASTRE_SRID};{tile_poly.wkt}"

                # Register the tile in the PostGIS ledger
                cursor.execute("""
                    INSERT INTO lidar_tile_index (file_path, tile_boundary)
                    VALUES (%s, ST_GeomFromEWKT(%s))
                    ON CONFLICT (file_path) DO UPDATE SET tile_boundary = EXCLUDED.tile_boundary;
                """, (abs_path, wkt_geom))
                print(f"   ✅ Indexed: {filename} (Bounds: X:{minx:.1f}->{maxx:.1f}, Y:{miny:.1f}->{maxy:.1f})")

    conn.commit()
    cursor.close()
    conn.close()
    print("✨ Dynamic LiDAR Spatial Indexing Complete!")


if __name__ == "__main__":
    index_lidar_directory()