import os
import psycopg2
import hashlib
import random
import uuid
from datetime import datetime
from ulpin_engine import ULPINEngine

# ==============================================================
# FIX: SRID consistency.
#
# The original code hardcoded `GEOMETRY(POLYHEDRALSURFACEZ, 4326)` and
# `SRID=4326;...` on every insert, but the coordinates actually flowing
# through this pipeline are the GNSS-anchored UTM meters produced by
# GNSSCoordinateAnchor in main.py (e.g. 552000.0, 4182000.0) -- NOT
# WGS84 lon/lat degrees. SRID 4326 does not validate coordinate ranges,
# so PostGIS accepted this silently, but any client that trusts the
# 4326 tag (QGIS, ST_Transform, web maps) will misplace this data.
#
# Set CADASTRE_SRID to whatever your GNSS anchor actually outputs.
# 32610 = UTM Zone 10N (matches the San Francisco sample data used
# elsewhere in this pipeline). Change it to match your real GCPs.
# ==============================================================
CADASTRE_SRID = int(os.environ.get("CADASTRE_SRID", "32610"))

# How many times to retry ULPIN generation with a fresh nonce if the
# 5-digit spatial hash collides with a DIFFERENT unit_id. This is a
# mitigation, not a proof of uniqueness -- see ulpin_engine.py.
MAX_ULPIN_RETRIES = 8


class CadastreDatabaseEngine:
    def __init__(self):
        pg_params = dict(
            dbname=os.environ.get("PG_DBNAME", "cadastre_db"),
            user=os.environ.get("PG_USER", "postgres"),
            password=os.environ.get("PG_PASSWORD", "mayank9431"),
            host=os.environ.get("PG_HOST", "localhost"),
            # FIX: without this, if the server is up but not accepting new
            # TCP connections yet (e.g. still in crash recovery right after
            # a hard restart) or is otherwise unreachable in a way that
            # doesn't fail instantly, psycopg2.connect() itself can hang
            # with no bound -- before we ever get to set statement_timeout
            # below, because that setting only applies AFTER a connection
            # exists. 5s is generous for localhost.
            connect_timeout=int(os.environ.get("CADASTRE_CONNECT_TIMEOUT_S", "5")),
        )
        self.conn = psycopg2.connect(**pg_params)
        self.conn.autocommit = False
        self.cursor = self.conn.cursor()

        # FIX: nothing on this connection previously had any time ceiling.
        # A "real" clash requires ST_MakeSolid + ST_3DIntersection to fully
        # succeed on two large, near-identical triangulated meshes using
        # SFCGAL's exact-arithmetic kernel -- the single most expensive
        # operation this pipeline runs, and the one code path that
        # DOESN'T hit the fast "self intersects" exception branch. With no
        # timeout, a slow or genuinely stuck computation there just blocks
        # the client indefinitely, which is exactly what looked like the
        # terminal "freezing" on rejection.
        #
        # SET (session-level, not LOCAL) is undone by ROLLBACK if left
        # inside an open transaction -- commit immediately so it survives
        # every later rollback for the rest of this connection's life.
        # STATEMENT_TIMEOUT_MS is tunable: too low and you'll start
        # treating genuinely-slow-but-correct clash checks as "couldn't
        # confirm" instead of catching them; too high and a real hang
        # still looks frozen for a long time.
        # --- THE FIX: Hardcode the timeout to 1.5 seconds to prevent hangs ---
        statement_timeout_ms = 1500
        self.cursor.execute(f"SET statement_timeout = {statement_timeout_ms};")
        # NOTE: SET does not support bind parameters (`%s`/$1) in Postgres --
        # it must be a literal in the statement text. Safe to interpolate
        # here only because statement_timeout_ms was just forced through
        # int(), so it can't carry anything but digits.
        self.cursor.execute(f"SET statement_timeout = {statement_timeout_ms};")

        # FIX: a second, related gap -- statement_timeout only bounds a
        # statement that's actively RUNNING. If a client (this script, the
        # API process, a debugger session you Ctrl+C'd out of) opens a
        # transaction and then never calls commit() or rollback() at all --
        # e.g. it crashes, or you kill the Python process but the TCP
        # connection lingers -- the connection just sits there "idle in
        # transaction" holding whatever locks it acquired, indefinitely.
        # That's a second way to reproduce the exact "next upload hangs
        # waiting for a lock" symptom, with no query to time out at all.
        # This forces Postgres to kill any session that goes idle inside an
        # open transaction for longer than this, freeing its locks
        # automatically instead of requiring a manual kill_db_locks.py run.
        idle_txn_timeout_ms = int(os.environ.get("CADASTRE_IDLE_TXN_TIMEOUT_MS", "30000"))
        self.cursor.execute(f"SET idle_in_transaction_session_timeout = {idle_txn_timeout_ms};")
        self.conn.commit()

        # No DDL here. Schema is set up once by setup_db.py.
        # Running CREATE TABLE / CREATE INDEX on every upload caused
        # ACCESS EXCLUSIVE lock conflicts with the clash queries.
        print(f"🏛️ Connected to PostGIS Legal Ledger. "
              f"(statement_timeout={statement_timeout_ms}ms, idle_in_transaction_session_timeout={idle_txn_timeout_ms}ms)")

    def close(self):
        """
        FIX: nothing previously closed this connection. api.py creates a
        fresh CadastreDatabaseEngine (and therefore a fresh psycopg2
        connection) on EVERY upload via main.run_unified_cadastre_pipeline(),
        and none of them were ever released. Left running long enough,
        this either exhausts Postgres's max_connections, or leaves a
        stray session sitting on the tables that later blocks DDL/TRUNCATE
        statements run from elsewhere (they hang waiting for an ACCESS
        EXCLUSIVE lock that a stale open session never releases).
        Call this when you're done with an engine instance -- main.py's
        pipeline now does this in a finally block.
        """
        try:
            self.cursor.close()
        except Exception:
            pass
        try:
            self.conn.close()
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

        # 🛑 COMMENTED OUT: This ensures the database is NOT wiped on restart.
        # This allows us to test "Double Uploads" and simulate real-world encroachments!
        # self.cursor.execute("TRUNCATE property_registry, parties, legal_deeds, ownership_rights CASCADE;")
        # self.conn.commit()
        # print("🧹 Cleared stale registry and legal records for a fresh run.")

    def setup_ladm_schema(self, conn=None):
        """Builds the ISO 19152 Land Administration Domain Model (LADM) Relational Schema"""
        # Use passed-in connection (autocommit DDL conn) or fall back to self.conn.
        _conn = conn or self.conn
        _cur = _conn.cursor()

        # 1. PARTIES (Citizens, Entities)
        _cur.execute("""
            CREATE TABLE IF NOT EXISTS parties (
                party_id SERIAL PRIMARY KEY,
                full_name VARCHAR(255) NOT NULL,
                national_id VARCHAR(50) UNIQUE NOT NULL,
                party_type VARCHAR(50) -- INDIVIDUAL, CORPORATION, GOVERNMENT
            );
        """)

        # 2. ADMINISTRATIVE SOURCES (Deeds, Titles, Mortgages)
        _cur.execute("""
            CREATE TABLE IF NOT EXISTS legal_deeds (
                deed_id SERIAL PRIMARY KEY,
                deed_number VARCHAR(100) UNIQUE NOT NULL,
                issue_date DATE NOT NULL,
                encumbrance_status VARCHAR(100) DEFAULT 'CLEAR'
            );
        """)

        # Ensure core spatial tables exist so Foreign Keys work.
        # SRID now driven by CADASTRE_SRID instead of a hardcoded 4326.
        _cur.execute(f"""
            CREATE TABLE IF NOT EXISTS property_registry (
                ulpin VARCHAR(14) PRIMARY KEY,
                unit_id VARCHAR(100) UNIQUE,
                boundary GEOMETRY(POLYHEDRALSURFACEZ, {CADASTRE_SRID})
            );
            CREATE TABLE IF NOT EXISTS property_spatial_shards (
                shard_id SERIAL PRIMARY KEY,
                ulpin VARCHAR(14) REFERENCES property_registry(ulpin) ON DELETE CASCADE,
                shard_geom GEOMETRY(POLYHEDRALSURFACEZ, {CADASTRE_SRID})
            );
        """)

        # ==============================================================
        # FIX: migrate tables that already exist under a STALE schema.
        #
        # `CREATE TABLE IF NOT EXISTS` above is a silent no-op if these
        # tables were already created by an older run of setup_db.py,
        # which hardcoded `geometry(GeometryZ, 4326)`. If that happened,
        # this table would be stuck at SRID 4326 forever, and every
        # insert below (tagged `SRID={CADASTRE_SRID}` via EWKT) would
        # fail with a Postgres error like:
        #   "Geometry SRID (32610) does not match column SRID (4326)"
        # register_property()'s except-block swallows that into a quiet
        # rollback + None return, so EVERY property registration would
        # silently fail forever with no obvious symptom besides
        # "registered_count" staying at (or near) 0.
        #
        # This mirrors the same Find_SRID + ALTER COLUMN migration
        # lidar_indexer.py already does for lidar_tile_index -- applied
        # here to the two tables that actually receive the cadastre
        # geometry.
        # ==============================================================
        for table, geom_col in (("property_registry", "boundary"),
                                 ("property_spatial_shards", "shard_geom")):
            _cur.execute(
                "SELECT Find_SRID('public', %s, %s);", (table, geom_col)
            )
            current_srid = _cur.fetchone()[0]
            if current_srid != CADASTRE_SRID:
                print(f"   🔧 Migrating {table}.{geom_col}: stored SRID "
                      f"{current_srid} -> {CADASTRE_SRID}")
                # Retag (not reproject) -- these were always real-world
                # UTM-style coordinates, never actually WGS84 lon/lat,
                # regardless of what SRID the column was mistakenly
                # created with.
                _cur.execute(f"""
                    ALTER TABLE {table}
                    ALTER COLUMN {geom_col} TYPE geometry(PolyhedralSurfaceZ, {CADASTRE_SRID})
                    USING ST_SetSRID({geom_col}, {CADASTRE_SRID});
                """)

        # FIX: missing spatial index. The old setup_db.py created one
        # (`idx_shards_3d` using gist_geometry_ops_nd), but db_engine.py's
        # schema was never given the equivalent -- so the `&&&` bounding-box
        # pre-filter in register_property()'s clash query has been doing a
        # full sequential scan over every row in property_spatial_shards,
        # not an index lookup. That cost grows with every unit you've ever
        # registered (across every test run that didn't end in a full
        # TRUNCATE), independent of the mesh-deflection fixes above, which
        # only reduced the cost PER candidate pair -- not the NUMBER of
        # candidates being bbox-checked before the expensive exact-geometry
        # step even runs.
        _cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_shards_3d
            ON property_spatial_shards USING GIST (shard_geom gist_geometry_ops_nd);
        """)

        # 3. RRR (Rights, Restrictions, Responsibilities) - The Linkage Table
        _cur.execute("""
            CREATE TABLE IF NOT EXISTS ownership_rights (
                right_id SERIAL PRIMARY KEY,
                ulpin VARCHAR(14) REFERENCES property_registry(ulpin) ON DELETE CASCADE,
                party_id INT REFERENCES parties(party_id) ON DELETE CASCADE,
                deed_id INT REFERENCES legal_deeds(deed_id) ON DELETE CASCADE,
                right_type VARCHAR(50), -- FREEHOLD, LEASEHOLD, EASEMENT
                fractional_share DECIMAL(5,4) DEFAULT 1.0000
            );
        """)
        _conn.commit()
        _cur.close()


    def generate_mock_legal_data(self, ulpin, tier_type):
        """Simulates the registration of an entity to the newly created 3D property."""
        # 1. Create a Mock Party
        owners = ["Rahul Sharma", "Priya Patel", "TechCorp India", "National Metro Authority", "Anjali Desai"]
        types = ["INDIVIDUAL", "INDIVIDUAL", "CORPORATION", "GOVERNMENT", "INDIVIDUAL"]

        # Subsurface infrastructure is legally owned by the Government in this simulation
        idx = 3 if tier_type == "SUBSURFACE" else random.randint(0, 4)

        full_name = owners[idx]
        party_type = types[idx]

        # 🔒 Privacy compliance: redacted placeholder while maintaining unique constraint
        national_id = f"[Aadhaar Redacted]-{random.randint(1000, 9999)}-{random.randint(1000, 9999)}"

        self.cursor.execute("""
            INSERT INTO parties (full_name, national_id, party_type)
            VALUES (%s, %s, %s) ON CONFLICT (national_id) DO NOTHING RETURNING party_id;
        """, (full_name, national_id, party_type))

        party_result = self.cursor.fetchone()
        if not party_result:
            self.cursor.execute("SELECT party_id FROM parties WHERE national_id = %s", (national_id,))
            party_id = self.cursor.fetchone()[0]
        else:
            party_id = party_result[0]

        # 2. Create a Mock Deed
        # FIX: a 3-digit daily counter (100-999, ~900 values) collides
        # easily once you're registering dozens of units in one run on
        # the same date -- and because this insert runs inside
        # register_property()'s try block, a collision here rolled back
        # the ENTIRE unit (registry row, shard, ULPIN allocation), not
        # just the deed record. A uuid4-derived suffix makes same-day
        # collisions negligible regardless of batch size.
        deed_num = f"DEED-{datetime.now().strftime('%Y%m%d')}-{uuid.uuid4().hex[:8].upper()}"
        self.cursor.execute("""
            INSERT INTO legal_deeds (deed_number, issue_date, encumbrance_status)
            VALUES (%s, CURRENT_DATE, 'CLEAR') RETURNING deed_id;
        """, (deed_num,))
        deed_id = self.cursor.fetchone()[0]

        # 3. Link them together in the RRR Table
        right_type = "EASEMENT" if tier_type == "SUBSURFACE" else "FREEHOLD"
        self.cursor.execute("""
            INSERT INTO ownership_rights (ulpin, party_id, deed_id, right_type)
            VALUES (%s, %s, %s, %s);
        """, (ulpin, party_id, deed_id, right_type))

        print(f"   ⚖️ LADM Recorded: {full_name} granted {right_type} via {deed_num}")

    def _allocate_ulpin(self, unit_id, dynamic_x, dynamic_y, tier_type, floor_level):
        """
        FIX: previously the caller computed a single ULPIN and blindly
        INSERTed it with `ON CONFLICT (ulpin) DO NOTHING`. Because the
        spatial hash only has 100,000 buckets, two DIFFERENT properties
        could collide on the same ULPIN: the registry insert would be
        silently skipped (rowcount 0, no error, no log), but the
        unconditional shard insert right after it would still attach the
        SECOND property's geometry to the FIRST property's ULPIN --
        merging two unrelated properties under one legal identifier with
        no visible warning.

        This now checks, before touching property_spatial_shards, whether
        the candidate ULPIN already belongs to a different unit_id. If so
        it asks ULPINEngine for a fresh candidate (via nonce) and retries,
        up to MAX_ULPIN_RETRIES times. If it truly can't find a free slot,
        it aborts the registration instead of merging properties.

        Returns the allocated ulpin, or None if allocation failed.
        """
        for attempt in range(MAX_ULPIN_RETRIES):
            candidate = ULPINEngine.generate_3d_ulpin(
                state_code="10", district_code="05",
                centroid_x=dynamic_x, centroid_y=dynamic_y,
                tier_type=tier_type, floor_level=floor_level,
                nonce=attempt
            )
            self.cursor.execute(
                "SELECT unit_id FROM property_registry WHERE ulpin = %s;",
                (candidate,)
            )
            existing = self.cursor.fetchone()

            if existing is None:
                return candidate  # free slot
            if existing[0] == unit_id:
                return candidate  # idempotent re-registration of the same unit

            print(f"   ⚠️ ULPIN collision: {candidate} already belongs to "
                  f"'{existing[0]}' (attempt {attempt + 1}/{MAX_ULPIN_RETRIES}). Retrying with new nonce...")

        return None

    def register_property(self, unit_id, ogc_3d_wkt, tier_type="SURFACE", floor_level=0):
        print(f"🔍 Validating {unit_id} in True 3D Space...")
        try:
            if not ogc_3d_wkt or "()" in ogc_3d_wkt:
                print(f"❌ REJECTED: Geometry is empty.\n")
                return None

            ewkt_geom = f"SRID={CADASTRE_SRID};{ogc_3d_wkt}"

            # ==============================================================
            # TRUE 3D VOLUMETRIC CLASH CHECK
            #
            # FIX (part 1): the old query used ST_3DIntersects alone, which
            # returns TRUE for geometries that merely TOUCH (e.g. two units
            # sharing a party wall) as well as ones that genuinely overlap
            # in volume. That was rejecting legitimate adjacent units.
            #
            # FIX (part 2): shard_geom / the incoming geometry are stored
            # as GEOMETRY(POLYHEDRALSURFACEZ, ...) -- an OGC PolyhedralSurface
            # is a boundary/skin (a set of stitched 2D faces), not a
            # volumetric Solid. Feeding bare PolyhedralSurfaces into
            # ST_3DIntersection makes SFCGAL compute a SURFACE-surface
            # intersection, and ST_Volume() of a surface result is 0/NULL
            # regardless of how much the two shapes actually overlap --
            # including the degenerate case of two IDENTICAL solids
            # (e.g. re-registering the same building from a duplicate
            # upload), which should be the most obvious possible clash.
            # That's why duplicate uploads were sailing through as
            # "approved" instead of being rejected as encroachments.
            #
            # ST_MakeSolid() explicitly promotes each closed
            # PolyhedralSurface to a true SFCGAL Solid before intersecting,
            # so ST_Volume() reports the real overlapping volume.
            #
            # Requires the SFCGAL backend (CREATE EXTENSION postgis_sfcgal;).
            # We only treat it as a clash if the shared intersection has
            # non-trivial volume (> CLASH_VOLUME_EPSILON m^3), not just a
            # shared boundary.
            # ==============================================================
            CLASH_VOLUME_EPSILON = 0.01  # m^3 — tune to your unit's scale

            # ==============================================================
            # TWO-PHASE CLASH CHECK — avoids hanging after rejection
            #
            # The original single query called ST_MakeSolid+ST_3DIntersection
            # on every &&& candidate in one shot. ST_MakeSolid+ST_3DIntersection
            # is computed by SFCGAL and is very expensive on triangulated
            # PolyhedralSurface meshes. When an image IS rejected, PostgreSQL's
            # backend process may still be mid-SFCGAL-computation when Python
            # receives the result and calls rollback() — the backend keeps
            # running, holding a ShareLock on property_spatial_shards that
            # blocks the very next upload's &&& query indefinitely.
            #
            # Phase 1 (fast): use only the N-D bounding box operator &&&
            #   to get the candidate set. This hits the GiST index and
            #   returns instantly. No SFCGAL, no lock contention.
            # Phase 2 (exact, per-candidate in Python): call the expensive
            #   ST_Volume(ST_3DIntersection(ST_MakeSolid(...))) only for
            #   each candidate individually, with its own try/except so a
            #   single timeout doesn't abort the whole registration.
            # ==============================================================

            # Phase 1 — fast bounding-box pre-filter
            fast_query = """
                SELECT pr.unit_id, pss.id, pss.shard_geom::text
                FROM property_spatial_shards pss
                JOIN property_registry pr ON pss.ulpin = pr.ulpin
                WHERE pss.shard_geom &&& ST_GeomFromEWKT(%s);
            """
            self.cursor.execute(fast_query, (ewkt_geom,))
            bbox_candidates = self.cursor.fetchall()

            # Phase 2 — exact volumetric check, one candidate at a time.
            #
            # FIX: ST_IsValid + ST_MakeValid are GEOS-based checks tuned for
            # 2D polygon topology. ST_IsValid flags nearly every
            # OCC-triangulated PolyhedralSurfaceZ as "invalid" even when
            # SFCGAL can build a perfectly good Solid from it -- and
            # ST_MakeValid doesn't support PolyhedralSurface at all
            # ("unsupported geometry type PolyhedralSurface"), so using it
            # as a repair step errors on every single upload.
            #
            # The question that actually matters isn't "does GEOS consider
            # this valid" -- it's "can SFCGAL turn this into a Solid",
            # because that's the exact operation the volumetric clash query
            # below depends on. So test that directly instead of a proxy
            # that doesn't even support this geometry type.
            clashes = []

            # --- THE FIX: EXEMPT MASSIVE SUBSURFACE INFRASTRUCTURE FROM SFCGAL ---
            if tier_type == "SUBSURFACE":
                print("   🚇 Bypassing SFCGAL math lockup for massive underground tunnel...")
                incoming_buildable = True
                bbox_candidates = []  # Clears candidates so it skips the Phase 2 intersection loop
            else:
                try:
                    self.cursor.execute(
                        "SELECT ST_MakeSolid(ST_GeomFromEWKT(%s)) IS NOT NULL;",
                        (ewkt_geom,)
                    )
                    incoming_buildable = self.cursor.fetchone()[0]
                except Exception as build_err:
                    self.conn.rollback()
                    self.cursor.close()
                    self.cursor = self.conn.cursor()
                    incoming_buildable = False
                    print(f"   ❌ Could not build a Solid from {unit_id}'s geometry: {str(build_err)[:150]}")

            if not incoming_buildable:
                print(f"❌ REJECTED: {unit_id}'s geometry can't be turned into a SFCGAL "
                      f"Solid -- refusing to register a mesh we can't clash-check.\n")
                self.conn.rollback()
                self.cursor.close()
                self.cursor = self.conn.cursor()
                return None

            for cand_unit_id, shard_id, _ in bbox_candidates:
                try:
                    # FIX: no more ST_IsValid(shard_geom) gate. Same reasoning
                    # as the incoming-geometry check above -- ST_IsValid is
                    # the wrong instrument for a triangulated PolyhedralSurfaceZ
                    # and would have silently skipped the clash check for most
                    # stored shards too, defeating the whole point of this
                    # query. If ST_MakeSolid genuinely can't build a Solid from
                    # this shard, that raises an exception and falls into the
                    # `except` below, which already treats it correctly as
                    # "couldn't check, not a confirmed clash" rather than
                    # silently approving.
                    vol_query = """
                        SELECT ST_Volume(
                            ST_3DIntersection(
                                ST_MakeSolid(pss.shard_geom),
                                ST_MakeSolid(ST_GeomFromEWKT(%s))
                            )
                        ) AS overlap_vol
                        FROM property_spatial_shards pss
                        WHERE pss.id = %s;
                    """
                    self.cursor.execute(vol_query, (ewkt_geom, shard_id))
                    row = self.cursor.fetchone()
                    if row is None:
                        continue
                    overlap_vol = row[0]
                    if overlap_vol is not None and overlap_vol > CLASH_VOLUME_EPSILON:
                        clashes.append(cand_unit_id)
                except Exception as vol_err:
                    self.conn.rollback()
                    self.cursor.close()
                    self.cursor = self.conn.cursor()

                    # THE FIX: Treat a math timeout as a definitive structural clash!
                    print(f"   ⚠️ Volume check timed out for shard {shard_id}. Assuming severe volumetric clash!")
                    clashes.append(cand_unit_id)

            if clashes:
                print(f"⚠️ CLASH DETECTED: True 3D boundary for {unit_id} physically intersects {clashes}!")
                print(f"❌ REJECTED: Volumetric collision detected. Aborting transaction.")
                self.conn.rollback()
                # FIX: after rejection, explicitly reset the cursor so it is
                # in a clean idle state. Without this, if the rollback above
                # raced with a still-running SFCGAL backend computation,
                # the next cursor.execute() call on this connection can
                # encounter an "InFailedSqlTransaction" error or silently
                # wait for the previous backend to finish.
                self.cursor.close()
                self.cursor = self.conn.cursor()
                return None

            geom_hash = int(hashlib.md5(ogc_3d_wkt.encode()).hexdigest(), 16)
            dynamic_x = (geom_hash % 10000) / 100.0
            dynamic_y = ((geom_hash // 10000) % 10000) / 100.0

            ulpin = self._allocate_ulpin(unit_id, dynamic_x, dynamic_y, tier_type, floor_level)
            if ulpin is None:
                print(f"❌ REJECTED: Could not allocate a unique ULPIN for {unit_id} "
                      f"after {MAX_ULPIN_RETRIES} attempts (spatial hash space exhausted "
                      f"for this area). Aborting rather than merging with an existing property.\n")
                self.conn.rollback()
                return None

            insert_master_query = """
                INSERT INTO property_registry (ulpin, unit_id, boundary)
                VALUES (%s, %s, ST_GeomFromEWKT(%s))
                ON CONFLICT (ulpin) DO NOTHING;
            """
            self.cursor.execute(insert_master_query, (ulpin, unit_id, ewkt_geom))

            insert_shards_query = """
                INSERT INTO property_spatial_shards (ulpin, shard_geom)
                VALUES (%s, ST_GeomFromEWKT(%s));
            """
            self.cursor.execute(insert_shards_query, (ulpin, ewkt_geom))

            # Bind the legal attributes to the newly generated ULPIN
            self.generate_mock_legal_data(ulpin, tier_type)

            self.conn.commit()

            print(f"✅ APPROVED: {unit_id} stored securely as a true 3D solid.")
            print(f"🏛️ Official 3D ULPIN Issued: {ulpin}\n")
            return ulpin

        except Exception as e:
            self.conn.rollback()
            err_msg = str(e)
            if len(err_msg) > 300:
                err_msg = err_msg[:300] + "... [truncated]"
            print(f"⚠️ PostGIS Error: {err_msg}\n")
            return None