"""
lidar_indexer.py -- broad-phase spatial index of LiDAR FILES (not geometry).

TRUE-3D CONTRACT
----------------
* `lidar_tile_index.tile_boundary` is a 2D XY bounding box taken from each LAS/LAZ
  header. Its ONLY purpose is to locate which tile files may contain points near
  an area of interest (broad-phase spatial lookup).
* It must NEVER create, clip, position, or determine building geometry, and no Z,
  height, or elevation is derived from it. The column is a 2D POLYGON, so it
  cannot carry Z, and the header Z range is deliberately not stored.
* A tile box is only a candidate filter: it is header-declared (not measured from
  points) and may contain empty area. Authoritative geometry must come from the
  raw measured XYZ points read from the tile files themselves.
* CRS: each tile's header XY bounds are in the file's OWN CRS (read from the
  header via `parse_crs()`). Only X/Y are transformed into CADASTRE_SRID before
  storing. A tile with no valid source CRS, or whose transform fails, aborts
  the run; a CRS is never guessed. Z is never read, transformed or stored.
"""
import math
import os
import laspy
import psycopg2
from pyproj import CRS, Transformer
from shapely.geometry import box
import config

# FIX: must match CADASTRE_SRID in db_engine.py / z_engine.py. Now
# sourced from config.py instead of its own env-var read, so it can't
# drift out of sync with the rest of the pipeline.
CADASTRE_SRID = config.CADASTRE_SRID


def _tile_xy_bounds_in_cadastre_srid(header, filename):
    """
    Returns (minx, miny, maxx, maxy) of the tile header's XY bounds transformed
    from the file's own CRS into CADASTRE_SRID. Only X/Y are used; Z is never
    touched. Raises RuntimeError if the source CRS is missing/invalid or the
    transform fails -- never guesses a CRS.
    """
    try:
        src_crs = header.parse_crs()
    except Exception as e:
        raise RuntimeError(f"{filename}: could not parse source CRS from LAS/LAZ header: {e}") from e
    if src_crs is None:
        raise RuntimeError(f"{filename}: no CRS found in LAS/LAZ header; refusing to guess one")

    # Use only the horizontal part of a compound (horizontal + vertical) CRS.
    if src_crs.is_compound:
        src_crs = src_crs.sub_crs_list[0]

    minx, miny, _ = header.mins   # Z deliberately discarded
    maxx, maxy, _ = header.maxs

    try:
        transformer = Transformer.from_crs(src_crs, CRS.from_epsg(int(CADASTRE_SRID)), always_xy=True)
        # Densified edges so the result still covers the tile under projection distortion.
        tminx, tminy, tmaxx, tmaxy = transformer.transform_bounds(minx, miny, maxx, maxy, densify_pts=21)
    except Exception as e:
        raise RuntimeError(
            f"{filename}: transform from source CRS ({src_crs.to_string()}) to EPSG:{CADASTRE_SRID} failed: {e}"
        ) from e

    if not all(math.isfinite(v) for v in (tminx, tminy, tmaxx, tmaxy)):
        raise RuntimeError(
            f"{filename}: transform from source CRS ({src_crs.to_string()}) to EPSG:{CADASTRE_SRID} "
            f"produced non-finite coordinates"
        )
    return tminx, tminy, tmaxx, tmaxy


def index_lidar_directory(lidar_dir="data/raw_lidar"):
    """
    Registers each LAS/LAZ tile's header XY bounding box in `lidar_tile_index`
    as a broad-phase file locator ONLY (see module contract). The box is never
    building geometry, and no Z is stored or inferred.
    """
    print(f"🌍 Connecting to PostGIS to build Spatial LiDAR Index...")
    conn = psycopg2.connect(**config.PG_DSN_KWARGS)
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

    # Keep the contract attached to the data itself, visible to any consumer.
    cursor.execute("""
        COMMENT ON TABLE lidar_tile_index IS
            'Broad-phase LiDAR file locator only. Not building geometry.';
        COMMENT ON COLUMN lidar_tile_index.tile_boundary IS
            'Header XY bounding box for locating tile files. Never creates, clips, positions or determines building geometry; no Z. Authoritative geometry comes from raw measured XYZ points.';
    """)
    conn.commit()

    if not os.path.exists(lidar_dir):
        os.makedirs(lidar_dir)
        print(f"⚠️ Directory {lidar_dir} created. Please add .laz files here.")
        cursor.close()
        conn.close()
        return

    print(f"🔍 Scanning directory {lidar_dir} for LiDAR tiles...")

    try:
        for filename in os.listdir(lidar_dir):
            if filename.endswith(".laz") or filename.endswith(".las"):
                file_path = os.path.join(lidar_dir, filename)
                abs_path = os.path.abspath(file_path).replace("\\", "/")  # Normalize for Windows/DB

                # Read ONLY the header metadata to extract the OGC bounding box
                with laspy.open(file_path) as fh:
                    header = fh.header
                    # Header XY bounds are in the file's own CRS: transform X/Y only into
                    # CADASTRE_SRID (aborts if the source CRS is missing or transform fails).
                    minx, miny, maxx, maxy = _tile_xy_bounds_in_cadastre_srid(header, filename)

                    # Flat 2D box: broad-phase file lookup ONLY, never building geometry
                    tile_poly = box(minx, miny, maxx, maxy)
                    wkt_geom = f"SRID={CADASTRE_SRID};{tile_poly.wkt}"

                    # Register the tile in the PostGIS ledger
                    cursor.execute("""
                        INSERT INTO lidar_tile_index (file_path, tile_boundary)
                        VALUES (%s, ST_GeomFromEWKT(%s))
                        ON CONFLICT (file_path) DO UPDATE SET tile_boundary = EXCLUDED.tile_boundary;
                    """, (abs_path, wkt_geom))
                    print(f"   ✅ Indexed: {filename} (EPSG:{CADASTRE_SRID} bounds: X:{minx:.1f}->{maxx:.1f}, Y:{miny:.1f}->{maxy:.1f})")
    except Exception:
        # Abort: discard any tiles registered in this run so no partial/incorrect index is committed.
        conn.rollback()
        cursor.close()
        conn.close()
        print("❌ LiDAR indexing aborted; no tiles from this run were committed.")
        raise

    conn.commit()
    cursor.close()
    conn.close()
    print("✨ Dynamic LiDAR Spatial Indexing Complete!")


if __name__ == "__main__":
    index_lidar_directory()