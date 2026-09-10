import os
import psycopg2
import laspy
import numpy as np
from shapely.geometry import Polygon
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge

# Must match CADASTRE_SRID in db_engine.py. `lidar_tile_index.tile_boundary`
# needs to be stored in this same SRID for ST_Intersects to mean anything.
CADASTRE_SRID = int(os.environ.get("CADASTRE_SRID", "32610"))


def get_intersecting_lidar_tiles(poly):
    """
    Queries the PostGIS spatial index to find exactly which
    LiDAR tiles overlap with the AI-detected building footprint.

    IMPORTANT: `poly` must already be in real-world / global coordinates
    (the same CADASTRE_SRID as lidar_tile_index.tile_boundary), NOT raw
    image-pixel coordinates. Calling this with a pixel-space polygon will
    silently return zero tiles every time (pixel values like 0-1000
    interpreted as SRID coordinates will almost never intersect real
    tile boundaries), which is exactly why every roof used to come out
    flat regardless of what LiDAR was indexed. main.py now converts the
    footprint to global coordinates via the GNSS affine BEFORE calling
    calculate_z_bounds, specifically to satisfy this requirement.
    """
    try:
        conn = psycopg2.connect(
            dbname=os.environ.get("PG_DBNAME", "cadastre_db"),
            user=os.environ.get("PG_USER", "postgres"),
            password=os.environ.get("PG_PASSWORD", "mayank9431"),
            host=os.environ.get("PG_HOST", "localhost"),
        )
        cursor = conn.cursor()

        # Convert the shapely polygon into an OGC standard EWKT format,
        # tagged with the SAME SRID the tile index actually uses.
        wkt_geom = f"SRID={CADASTRE_SRID};{poly.wkt}"

        # Utilize the GiST spatial index to instantly find overlapping point cloud tiles
        cursor.execute("""
            SELECT file_path FROM lidar_tile_index
            WHERE ST_Intersects(tile_boundary, ST_GeomFromEWKT(%s));
        """, (wkt_geom,))

        tiles = [row[0] for row in cursor.fetchall()]

        cursor.close()
        conn.close()
        return tiles
    except Exception as e:
        print(f"❌ Spatial Index Query Failed: {e}")
        return []


def calculate_z_bounds(poly, lidar_path=None):
    """
    Dynamically fetches LAZ files based on the spatial footprint, extracts points,
    and fits a True 3D polynomial surface for domes, slants, and irregular Z-axis curves.

    `poly` must be in real-world / global coordinates (see note above).

    `lidar_path`: optional. If provided (e.g. run_drone.py passing an
    explicit .laz path), that single file is used directly instead of
    querying the PostGIS spatial index. This is what makes the extra
    argument run_drone.py passes actually do something, instead of being
    silently dropped.
    """
    if lidar_path:
        if not os.path.exists(lidar_path):
            print(f"   ⚠️ Explicit LiDAR path '{lidar_path}' not found. Falling back to flat roof.")
            return 0.0, None, None
        laz_paths = [lidar_path]
    else:
        # 1. Fetch relevant tiles dynamically from the PostGIS spatial index
        laz_paths = get_intersecting_lidar_tiles(poly)
        if not laz_paths:
            print("   ⚠️ No indexed LiDAR tiles intersect with this footprint. Falling back to flat roof.")
            return 0.0, None, None

    all_x, all_y, all_z = [], [], []
    minx, miny, maxx, maxy = poly.bounds

    # 2. Extract and merge points from ALL relevant tiles
    for laz_path in laz_paths:
        try:
            with laspy.open(laz_path) as fh:
                las = fh.read()

                # Broad Phase Bounding Box Crop (Massively faster than checking every point!)
                # OpenCASCADE handles the strict OGC boundary clipping later in the pipeline.
                mask = (las.x >= minx) & (las.x <= maxx) & (las.y >= miny) & (las.y <= maxy)

                all_x.extend(las.x[mask])
                all_y.extend(las.y[mask])
                all_z.extend(las.z[mask])
        except Exception as e:
            print(f"   ⚠️ Could not read tile {laz_path}: {e}")

    if not all_x:
        print("   ⚠️ LiDAR files loaded, but bounding box was empty. Falling back to flat roof.")
        return 0.0, None, None

    pts_x = np.array(all_x)
    pts_y = np.array(all_y)
    pts_z = np.array(all_z)

    # We use percentiles instead of np.min to ignore laser scatter or pits
    z_base = float(np.percentile(pts_z, 5))

    # ==============================================================
    # 3. Fit True 3D Polynomial Surface (Handles curves, slants, domes)
    # ==============================================================
    xy = np.column_stack((pts_x, pts_y))

    # Upgraded to Degree 3 to capture highly irregular architectural curves
    poly_features = PolynomialFeatures(degree=3)
    xy_poly = poly_features.fit_transform(xy)

    # Upgraded to Ridge Regression to handle matrix instability when merging multi-tile point clouds
    model = Ridge(alpha=1.0)
    model.fit(xy_poly, pts_z)

    return z_base, model, poly_features


# --- Quick Test ---
if __name__ == "__main__":
    import shapely.geometry as sg

    print(f"🔍 Testing Dynamic LiDAR Spatial Indexing...")

    # Create a dummy polygon in real-world coordinates matching your
    # CADASTRE_SRID (this example assumes UTM meters, not lon/lat degrees --
    # adjust to whatever CADASTRE_SRID is set to).
    dummy_room = sg.Polygon([
        (552000.0, 4182000.0), (552000.0, 4182010.0),
        (552010.0, 4182010.0), (552010.0, 4182000.0)
    ])

    z_dem, poly_model, poly_features = calculate_z_bounds(dummy_room)

    if poly_model:
        print(f"✅ Z-Axis Fusion Complete!")
        print(f"   Ground Level (z_dem): {z_dem:.2f}m")
        print(f"   Degree 3 Ridge Polynomial Model Trained Successfully!")
    else:
        print(f"⚠️ Test failed. Did you run `python lidar_indexer.py` first to build the database index?")