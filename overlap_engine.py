import os
import psycopg2
import shapely.geometry as sg
from shapely.strtree import STRtree

# ==========================================
# FIX: must match CADASTRE_SRID in db_engine.py, or every query in
# GlobalOverlapEngine will hit the same "mixed SRID geometries" /
# "does not match column SRID" errors that main.py's pipeline hit.
# This file previously hardcoded 4326 independently of db_engine.py's
# SRID, so the two modules could silently disagree even after
# db_engine.py was fixed.
# ==========================================
CADASTRE_SRID = int(os.environ.get("CADASTRE_SRID", "32610"))


# ==========================================
# 1. LOCAL IN-MEMORY ENGINE (Pre-processing)
# ==========================================
def detect_local_encroachments(rooms):
    """
    Checks for overlaps WITHIN a local batch of rooms (e.g., a single floor plan)
    before sending them to the database.
    """
    print(f"🌲 Building Local R-Tree index for {len(rooms)} property units...")
    polygons = [room["polygon"] for room in rooms]
    tree = STRtree(polygons)
    encroachments = []

    for i, room_a in enumerate(rooms):
        poly_a = room_a["polygon"]
        candidate_indices = tree.query(poly_a)

        for j in candidate_indices:
            if i >= j:
                continue

            room_b = rooms[j]
            poly_b = room_b["polygon"]

            # Z-Axis Overlap Check
            z_overlap = not (room_a["z_top"] <= room_b["z_base"] or room_a["z_base"] >= room_b["z_top"])

            if z_overlap and poly_a.intersects(poly_b):
                intersection_geom = poly_a.intersection(poly_b)
                if intersection_geom.area > 0.05:
                    encroachments.append({
                        "unit_1": room_a["id"],
                        "unit_2": room_b["id"],
                        "overlap_area_sqm": round(intersection_geom.area, 2)
                    })
    return encroachments


# ==========================================
# 2. GLOBAL POSTGIS ENGINE (Database Audit)
# ==========================================
class GlobalOverlapEngine:
    def __init__(self):
        # FIX: was hardcoded to dbname="cadastre_db", user="postgres",
        # password="mayank9431", host="localhost" regardless of
        # environment. db_engine.py reads these from PG_DBNAME/PG_USER/
        # PG_PASSWORD/PG_HOST env vars -- if those are ever set to
        # something other than the hardcoded defaults (a different host,
        # a different DB for staging vs prod, etc.), this class would
        # silently connect to a DIFFERENT database than the rest of the
        # pipeline and every query here would look at the wrong data.
        self.conn = psycopg2.connect(
            dbname=os.environ.get("PG_DBNAME", "cadastre_db"),
            user=os.environ.get("PG_USER", "postgres"),
            password=os.environ.get("PG_PASSWORD", "mayank9431"),
            host=os.environ.get("PG_HOST", "localhost"),
        )

    def convert_to_ewkt_3d(self, shapely_poly, z_base, z_top):
        """
        Converts a 2D Shapely polygon into a PostGIS 3D EWKT Solid/Extrusion
        base, tagged with CADASTRE_SRID.
        """
        coords = list(shapely_poly.exterior.coords)
        coord_strings = [f"{x} {y} {z_base}" for x, y in coords]
        poly_z_wkt = f"SRID={CADASTRE_SRID};POLYGON Z (({', '.join(coord_strings)}))"
        return poly_z_wkt, (z_top - z_base)

    def check_global_encroachment(self, ewkt_3d_polygon, height):
        """
        Queries the Multi-Cell Sharded Database using the fast R-Tree (&&&)
        and exact 3D volume intersection.

        FIX: previously built the query with raw f-string interpolation
        of the WKT and an inline `ST_GeomFromText(..., 4326)` call --
        both the hardcoded SRID and the lack of parameterization were
        bugs. Now uses a parameterized query and derives the geometry
        once from `ST_GeomFromEWKT`, whose SRID comes from the EWKT
        string itself (CADASTRE_SRID, set in convert_to_ewkt_3d), and
        computes the extrusion once in SQL via a CTE instead of
        duplicating the ST_Extrude(...) call in both the && and
        ST_3DIntersects clauses.
        """
        query = """
            WITH target AS (
                SELECT ST_Extrude(ST_GeomFromEWKT(%s), 0, 0, %s) AS geom
            )
            SELECT DISTINCT pr.unit_id
            FROM property_spatial_shards pss
            JOIN property_registry pr ON pss.ulpin = pr.ulpin
            CROSS JOIN target
            WHERE
                pss.shard_geom &&& target.geom
                AND ST_3DIntersects(pss.shard_geom, target.geom);
        """

        with self.conn.cursor() as cursor:
            cursor.execute(query, (ewkt_3d_polygon, height))
            conflicts = [row[0] for row in cursor.fetchall()]

        if conflicts:
            return False, conflicts
        return True, []

    def test_rtree_index(self):
        """Runs EXPLAIN ANALYZE to verify the GiST R-Tree is active."""
        print("\n📊 Running Global R-Tree Diagnostic...")
        try:
            with self.conn.cursor() as cursor:
                # Force index usage for testing even if the table is currently empty/small
                cursor.execute("SET enable_seqscan = OFF;")

                test_ewkt = f"SRID={CADASTRE_SRID};POLYGON Z ((0 0 0, 0 4 0, 4 4 0, 4 0 0, 0 0 0))"

                query = """
                    EXPLAIN ANALYZE
                    SELECT DISTINCT pr.unit_id
                    FROM property_spatial_shards pss
                    JOIN property_registry pr ON pss.ulpin = pr.ulpin
                    WHERE
                        pss.shard_geom &&& ST_Extrude(ST_GeomFromEWKT(%s), 0, 0, 3);
                """
                cursor.execute(query, (test_ewkt,))

                plan = cursor.fetchall()
                for row in plan:
                    line = row[0]
                    if "idx_shards_3d" in line:
                        print(f"✅ R-TREE CONFIRMED ACTIVE: {line}")
                    else:
                        print(f"   {line}")

        except Exception as e:
            print(f"⚠️ Diagnostic Error: {e}")
        finally:
            # Restore normal planner behavior for this connection instead
            # of leaving enable_seqscan permanently OFF for the session.
            try:
                with self.conn.cursor() as cursor:
                    cursor.execute("SET enable_seqscan = ON;")
            except Exception:
                pass


# ==========================================
# 3. TEST SCRIPT (Runs when you execute the file)
# ==========================================
if __name__ == "__main__":
    test_rooms = [
        {"id": "ULPIN_1001", "polygon": sg.Polygon([(0, 0), (0, 4), (4, 4), (4, 0)]), "z_base": 0.0, "z_top": 3.0},
        {"id": "ULPIN_1002", "polygon": sg.Polygon([(3, 0), (3, 4), (7, 4), (7, 0)]), "z_base": 0.0, "z_top": 3.0},
    ]

    # 1. Local Check
    local_conflicts = detect_local_encroachments(test_rooms)
    print("\n🚨 Local Spatial Conflicts Detected:")
    for conflict in local_conflicts:
        print(f" - {conflict['unit_1']} encroaches on {conflict['unit_2']} by {conflict['overlap_area_sqm']} sq meters.")

    # 2. Global Check & R-Tree Diagnostic
    print("\n🌍 Connecting to PostGIS Global Database for verification...")
    try:
        global_engine = GlobalOverlapEngine()

        # Test the database collision logic
        room_to_test = test_rooms[0]
        ewkt_poly, height = global_engine.convert_to_ewkt_3d(
            room_to_test["polygon"],
            room_to_test["z_base"],
            room_to_test["z_top"]
        )

        is_clear, db_conflicts = global_engine.check_global_encroachment(ewkt_poly, height)

        if is_clear:
            print(f"✅ {room_to_test['id']} is clear to be registered in the global cadastre.")
        else:
            print(f"❌ {room_to_test['id']} collides with existing global properties: {db_conflicts}")

        # RUN THE R-TREE DIAGNOSTIC
        global_engine.test_rtree_index()

    except Exception as e:
        print(f"⚠️ Could not test global engine (Make sure PostgreSQL is running): {e}")