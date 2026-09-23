import os
import time
import psycopg2
import psycopg2.errors
import random
import uuid
from datetime import datetime
from ulpin_engine import ULPINEngine
import config

# ==============================================================
# SRID Consistency
# Set CADASTRE_SRID to whatever your GNSS anchor actually outputs.
#
# 7755 = "WGS 84 / India NSF LCC", a Lambert Conformal Conic covering
# the ENTIRE country of India with a single consistent SRID (meters).
#
# This used to default to 32610 (UTM Zone 10N -- San Francisco). UTM
# zones are only ~6 degrees wide, so any Indian coordinate (which spans
# UTM zones 42N-47N) silently reprojected into the wrong zone, producing
# geometrically "valid" but physically nonsensical coordinates -- no
# crash, just corrupted data. A single national LCC avoids the
# per-request "which UTM zone is this point in" problem entirely, which
# matters because this system is meant to accept coordinates from
# anywhere in India, not one fixed city.
#
# FIX: now sourced from config.py, which requires PG_* to be explicitly
# set (no silent fallback to a dev password/host) -- see config.py.
# ==============================================================
CADASTRE_SRID = config.CADASTRE_SRID
MAX_ULPIN_RETRIES = 8
MAX_SERIALIZATION_RETRIES = 4


class CadastreDatabaseEngine:
    def __init__(self, use_pool=True):
        """
        `use_pool=True` (default, used by api.py under FastAPI request
        handling): borrows a connection from config.get_pool() instead
        of opening a brand-new TCP connection to Postgres per request.
        `use_pool=False`: standalone connection, for one-off scripts
        (setup_db.py-style tooling, tests, CLI diagnostics) that aren't
        running inside the pooled API process.
        """
        self._use_pool = use_pool
        if use_pool:
            self._pool = config.get_pool()
            self.conn = self._pool.getconn()
            self.conn.autocommit = False
        else:
            self.conn = psycopg2.connect(**config.PG_DSN_KWARGS)
            self.conn.autocommit = False
        self.cursor = self.conn.cursor()

        # Schema DDL (CREATE TABLE / ALTER TABLE / CREATE INDEX) is NOT run
        # here: constructing an engine performs no DDL. The schema is created
        # once, by setup_db.py, through setup_ladm_schema() below -- the single
        # authoritative schema definition. (DDL must not run under the strict
        # statement_timeout set next, which only guards runtime SFCGAL
        # volume-intersection queries in register_property().)

        # Hardcode the timeout to 1.5 seconds to prevent SFCGAL math hangs
        # -- applies to runtime queries only.
        statement_timeout_ms = 1500
        self.cursor.execute(f"SET statement_timeout = {statement_timeout_ms};")

        # Kill any session that goes idle inside an open transaction
        idle_txn_timeout_ms = int(os.environ.get("CADASTRE_IDLE_TXN_TIMEOUT_MS", "30000"))
        self.cursor.execute(f"SET idle_in_transaction_session_timeout = {idle_txn_timeout_ms};")
        self.conn.commit()

        print(f"🏛️ Connected to True 3D PostGIS Legal Ledger. "
              f"(statement_timeout={statement_timeout_ms}ms, idle_txn_timeout={idle_txn_timeout_ms}ms)")

    def close(self):
        try:
            self.cursor.close()
        except Exception:
            pass
        if self._use_pool:
            try:
                # Rolling back before returning to the pool guards
                # against handing back a connection that's mid-transaction
                # because an earlier caller forgot to commit/rollback.
                self.conn.rollback()
            except Exception:
                pass
            try:
                self._pool.putconn(self.conn)
            except Exception:
                pass
        else:
            try:
                self.conn.close()
            except Exception:
                pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def setup_ladm_schema(self, conn=None):
        """
        Builds the ISO 19152 Land Administration Domain Model (LADM) Relational
        Schema. This is the SINGLE authoritative schema definition; it is run
        once by setup_db.py and is NOT called when an engine is constructed.
        """
        _conn = conn or self.conn
        _cur = _conn.cursor()

        # 1. PARTIES
        _cur.execute("""
            CREATE TABLE IF NOT EXISTS parties (
                party_id SERIAL PRIMARY KEY,
                full_name VARCHAR(255) NOT NULL,
                national_id VARCHAR(50) UNIQUE NOT NULL,
                party_type VARCHAR(50)
            );
        """)

        # 2. LEGAL DEEDS
        _cur.execute("""
            CREATE TABLE IF NOT EXISTS legal_deeds (
                deed_id SERIAL PRIMARY KEY,
                deed_number VARCHAR(100) UNIQUE NOT NULL,
                issue_date DATE NOT NULL,
                encumbrance_status VARCHAR(100) DEFAULT 'CLEAR'
            );
        """)

        # 3. TRUE 3D SPATIAL REGISTRY 
        # Upgraded to inherently track Tier and Floor Level alongside Geometry
        _cur.execute(f"""
            CREATE TABLE IF NOT EXISTS property_registry (
                ulpin VARCHAR(14) PRIMARY KEY,
                unit_id VARCHAR(100) UNIQUE,
                tier_type VARCHAR(50),
                floor_level INT,
                boundary GEOMETRY(POLYHEDRALSURFACEZ, {CADASTRE_SRID}),
                -- TRUE 3D: these three are DERIVED FROM `boundary` by PostGIS
                -- at insert time, never supplied by the caller. They exist so
                -- the ledger can be queried and audited by volume and vertical
                -- extent without re-running SFCGAL on every read. If they ever
                -- disagree with the geometry, THE GEOMETRY WINS -- that is the
                -- whole point of storing a solid rather than a footprint plus a
                -- height attribute.
                volume_m3 DOUBLE PRECISION,
                z_min DOUBLE PRECISION,
                z_max DOUBLE PRECISION,
                -- User-provided location: METADATA / REFERENCE ONLY. Stored
                -- exactly as supplied (WGS84 lat/lon); never used to build,
                -- move or measure the geometry, and never a source of Z.
                location_latitude DOUBLE PRECISION,
                location_longitude DOUBLE PRECISION,
                location_accuracy_m DOUBLE PRECISION,
                location_source VARCHAR(100),
                location_crs VARCHAR(50) DEFAULT 'EPSG:4326',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            
            CREATE TABLE IF NOT EXISTS property_spatial_shards (
                shard_id SERIAL PRIMARY KEY,
                ulpin VARCHAR(14) REFERENCES property_registry(ulpin) ON DELETE CASCADE,
                shard_geom GEOMETRY(POLYHEDRALSURFACEZ, {CADASTRE_SRID})
            );
        """)

        # Schema Migration Safety (Ensures SRID matches)
        for table, geom_col in (("property_registry", "boundary"),
                                 ("property_spatial_shards", "shard_geom")):
            _cur.execute(
                "SELECT Find_SRID('public', %s, %s);", (table, geom_col)
            )
            current_srid = _cur.fetchone()[0]
            if current_srid != CADASTRE_SRID:
                # FAIL SAFE: Find_SRID() returns 0 when the column's SRID is
                # unknown/unconstrained (and None/negative for legacy or
                # broken metadata). Without a known source CRS there is no
                # valid transform, and guessing one (or relabelling with
                # ST_SetSRID) would silently corrupt every stored
                # coordinate. Abort before touching the column.
                if not current_srid or current_srid <= 0:
                    raise RuntimeError(
                        f"Cannot migrate {table}.{geom_col} to SRID {CADASTRE_SRID}: "
                        f"its stored SRID is unknown ({current_srid!r}). Refusing to "
                        f"guess a source CRS. Set the column's true SRID explicitly "
                        f"(after verifying what CRS the existing coordinates are in) "
                        f"and re-run."
                    )
                print(f"   🔧 Migrating {table}.{geom_col}: reprojecting stored SRID "
                      f"{current_srid} -> {CADASTRE_SRID} (ST_Transform)")
                # ST_Transform reprojects X/Y from the geometry's own stored
                # SRID (== current_srid, enforced by the column typmod) into
                # CADASTRE_SRID and carries Z through. The target typmod
                # PolyhedralSurfaceZ still enforces type and Z, so a result
                # that lost Z or the surface type aborts the ALTER instead
                # of being stored.
                _cur.execute(f"""
                    ALTER TABLE {table}
                    ALTER COLUMN {geom_col} TYPE geometry(PolyhedralSurfaceZ, {CADASTRE_SRID})
                    USING ST_Transform({geom_col}, {CADASTRE_SRID});
                """)

        # Schema Migration Safety (Ensures columns match too, not just
        # SRID) -- property_registry / property_spatial_shards may have
        # been first created by an older/other schema (e.g. setup_db.py,
        # which only ever defines ulpin/unit_id/boundary/created_at on
        # property_registry, and names the shard PK "id" instead of
        # "shard_id"). CREATE TABLE IF NOT EXISTS above is a no-op once
        # the table exists, so without an explicit migration these
        # columns silently never appear, and every register_property()
        # INSERT (which references tier_type/floor_level) and every
        # clash query (which references shard_id) fails forever with
        # UndefinedColumn -- caught, rolled back, and reported as a
        # normal "no clash" / None return rather than a visible error.
        _cur.execute("""
            ALTER TABLE property_registry
                ADD COLUMN IF NOT EXISTS tier_type VARCHAR(50),
                ADD COLUMN IF NOT EXISTS floor_level INT,
                ADD COLUMN IF NOT EXISTS volume_m3 DOUBLE PRECISION,
                ADD COLUMN IF NOT EXISTS z_min DOUBLE PRECISION,
                ADD COLUMN IF NOT EXISTS z_max DOUBLE PRECISION,
                ADD COLUMN IF NOT EXISTS location_latitude DOUBLE PRECISION,
                ADD COLUMN IF NOT EXISTS location_longitude DOUBLE PRECISION,
                ADD COLUMN IF NOT EXISTS location_accuracy_m DOUBLE PRECISION,
                ADD COLUMN IF NOT EXISTS location_source VARCHAR(100),
                ADD COLUMN IF NOT EXISTS location_crs VARCHAR(50) DEFAULT 'EPSG:4326',
                ADD COLUMN IF NOT EXISTS created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP;
        """)

        # property_spatial_shards: rename the legacy `id` PK to
        # `shard_id` if that's what an older schema created it as.
        # Every clash-detection query in this file reads/writes
        # shard_id, not id.
        _cur.execute("""
            DO $$
            BEGIN
                IF NOT EXISTS (
                    SELECT 1 FROM information_schema.columns
                    WHERE table_name = 'property_spatial_shards' AND column_name = 'shard_id'
                ) AND EXISTS (
                    SELECT 1 FROM information_schema.columns
                    WHERE table_name = 'property_spatial_shards' AND column_name = 'id'
                ) THEN
                    ALTER TABLE property_spatial_shards RENAME COLUMN id TO shard_id;
                END IF;
            END $$;
        """)

        # Build 3D N-Dimensional GiST Index (The &&& operator relies on this)
        _cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_shards_3d
            ON property_spatial_shards USING GIST (shard_geom gist_geometry_ops_nd);
        """)

        # Vertical-extent index. Makes "which titles occupy 12m-15m over this
        # plot" an indexed range scan over values DERIVED from the solids,
        # rather than a scan over a user-entered floor_level attribute.
        _cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_registry_z_extent
            ON property_registry (z_min, z_max);
        """)


        # 4. RRR (Rights, Restrictions, Responsibilities)
        _cur.execute("""
            CREATE TABLE IF NOT EXISTS ownership_rights (
                right_id SERIAL PRIMARY KEY,
                ulpin VARCHAR(14) REFERENCES property_registry(ulpin) ON DELETE CASCADE,
                party_id INT REFERENCES parties(party_id) ON DELETE CASCADE,
                deed_id INT REFERENCES legal_deeds(deed_id) ON DELETE CASCADE,
                right_type VARCHAR(50), 
                fractional_share DECIMAL(5,4) DEFAULT 1.0000
            );
        """)
        _conn.commit()
        _cur.close()

    def generate_mock_legal_data(self, ulpin, tier_type):
        """Simulates the registration of an entity to the newly created 3D property."""
        owners = ["Rahul Sharma", "Priya Patel", "TechCorp India", "National Metro Authority", "Anjali Desai"]
        types = ["INDIVIDUAL", "INDIVIDUAL", "CORPORATION", "GOVERNMENT", "INDIVIDUAL"]
        
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

        deed_num = f"DEED-{datetime.now().strftime('%Y%m%d')}-{uuid.uuid4().hex[:8].upper()}"
        self.cursor.execute("""
            INSERT INTO legal_deeds (deed_number, issue_date, encumbrance_status)
            VALUES (%s, CURRENT_DATE, 'CLEAR') RETURNING deed_id;
        """, (deed_num,))
        deed_id = self.cursor.fetchone()[0]

        right_type = "EASEMENT" if tier_type == "SUBSURFACE" else "FREEHOLD"
        self.cursor.execute("""
            INSERT INTO ownership_rights (ulpin, party_id, deed_id, right_type)
            VALUES (%s, %s, %s, %s);
        """, (ulpin, party_id, deed_id, right_type))

        print(f"   ⚖️ LADM Recorded: {full_name} granted {right_type} via {deed_num}")

    def _allocate_ulpin(self, unit_id, centroid_x, centroid_y, centroid_z,
                        tier_type, floor_level):
        """
        TRUE 3D: the centroid (x, y, z) is the exact volumetric centroid of
        the actual solid, derived from its faces inside PostGIS (see
        _measure_solid), not from a hash of the WKT string, user input, or
        a Z-extent midpoint. Z is treated identically to X and Y -- all
        three are coordinates of the same geometry-derived 3D point.

        The identifier (stored in the `ulpin` column, kept for compatibility)
        is this cadastre's custom 3D property ID from
        ULPINEngine.generate_3d_property_id(): derived solely from the
        measured centroid plus the collision-retry nonce. It carries no
        state/district code, tier or floor, and it is NOT the official
        ULPIN/PNIU.
        """
        for attempt in range(MAX_ULPIN_RETRIES):
            candidate = ULPINEngine.generate_3d_property_id(
                centroid_x=centroid_x, centroid_y=centroid_y,
                centroid_z=centroid_z,
                tier_type=tier_type, floor_level=floor_level,
                nonce=attempt,
            )
            self.cursor.execute("SELECT unit_id FROM property_registry WHERE ulpin = %s;", (candidate,))
            existing = self.cursor.fetchone()

            if existing is None:
                return candidate
            if existing[0] == unit_id:
                return candidate 

            print(f"   ⚠️ 3D property ID collision: {candidate} already belongs to "
                  f"'{existing[0]}' (attempt {attempt + 1}/{MAX_ULPIN_RETRIES}). Retrying...")

        return None


    # ==============================================================
    # TRUE 3D GATEKEEPING
    # ==============================================================

    MIN_REGISTRABLE_VOLUME_M3 = 0.5
    MIN_REGISTRABLE_Z_EXTENT_M = 0.05
    # SFCGAL volume math on a large swept tunnel genuinely can exceed the
    # 1.5s runtime guard. Raising the budget for that one statement is the
    # correct answer; skipping the check is not (see _clash_against_shard).
    HEAVY_GEOMETRY_TIMEOUT_MS = 12000

    def _measure_solid(self, ewkt_geom):
        """
        Asks PostGIS/SFCGAL what this geometry ACTUALLY is in 3D, and returns
        (volume_m3, z_min, z_max, centroid_x, centroid_y, centroid_z)
        or None if it cannot be made into a solid.

        TRUE 3D CENTROID
        ----------------
        PostGIS/SFCGAL has no 3D centroid function (ST_Centroid is 2D and
        raises on a POLYHEDRALSURFACE), and (ST_ZMin + ST_ZMax) / 2 is a
        bounding-box midpoint, not a centroid. The centroid here is the
        exact volumetric centroid of the closed shell, measured from the
        geometry's own faces by the divergence theorem: ST_Tesselate is used
        read-only to enumerate the faces as triangles, and each triangle
        contributes the signed tetrahedron it spans with a reference point.
        The reference point (the shell's min corner) is only a numeric
        offset that cancels out of the result. Nothing is stored, repaired,
        simplified, or reconstructed from the tessellation.

        The centroid is used for the 3D property ID's spatial hash, where Z is treated
        identically to X and Y -- all three are geometry-derived
        coordinates of the same 3D point, not user-supplied attributes. The
        overlap/clash check itself uses the full solid geometry
        (ST_3DIntersection + ST_Volume), never the centroid or z extents.

        VALIDATION (reject only -- geometry is never repaired)
        ------------------------------------------------------
        The input must be a 3D POLYHEDRALSURFACE that is closed, has a real
        vertical extent and volume, and whose face-summed volume agrees
        with ST_Volume. Validity itself is decided by SFCGAL's own solid
        validation, which ST_Volume(ST_MakeSolid(...)) in the probe below
        runs on the submitted shell and which raises on open shells,
        inconsistent face orientation, self-intersection, disconnected
        shells, and degenerate faces (rings that are zero-area or
        self-intersecting). That error is a rejection here, not something
        to fix up. There is deliberately NO minimum face/triangle area:
        finely tessellated curved or complex solids legitimately contain
        tiny triangles, and size is not a validity criterion.

        This is the gate that keeps 2.5D data out of a 3D ledger. Before this
        existed, `boundary` was typed POLYHEDRALSURFACEZ but nothing verified
        the surface enclosed anything: a flat footprint whose vertices all sat
        at the same Z, or an unclosed shell, would be accepted and stored as a
        "3D property". The registry would then contain a polygon wearing a Z
        coordinate -- height as an attribute, which is exactly the failure mode
        we are trying to eliminate -- and every downstream volumetric clash
        check against it would silently return zero overlap, because a surface
        with no interior cannot intersect anything volumetrically.
        """
        # TRUE 3D: exact volumetric centroid + validity facts, all measured
        # from the submitted shell (see docstring). ST_Volume(ST_MakeSolid())
        # is the SFCGAL solid-validity gate: an invalid shell raises here and
        # is rejected below. Coordinates are taken
        # relative to the shell's min corner purely to avoid cancellation
        # error at large projected coordinates; the offset is added back.
        probe = """
            WITH g AS (SELECT ST_GeomFromEWKT(%s) AS geom),
            o AS (
                SELECT geom, ST_XMin(geom) AS ox, ST_YMin(geom) AS oy,
                       ST_ZMin(geom) AS oz
                FROM g
            ),
            tri AS (
                SELECT o.ox, o.oy, o.oz,
                       ST_X(ST_PointN(ST_ExteriorRing(d.geom), 1)) - o.ox AS ax,
                       ST_Y(ST_PointN(ST_ExteriorRing(d.geom), 1)) - o.oy AS ay,
                       ST_Z(ST_PointN(ST_ExteriorRing(d.geom), 1)) - o.oz AS az,
                       ST_X(ST_PointN(ST_ExteriorRing(d.geom), 2)) - o.ox AS bx,
                       ST_Y(ST_PointN(ST_ExteriorRing(d.geom), 2)) - o.oy AS by,
                       ST_Z(ST_PointN(ST_ExteriorRing(d.geom), 2)) - o.oz AS bz,
                       ST_X(ST_PointN(ST_ExteriorRing(d.geom), 3)) - o.ox AS cx,
                       ST_Y(ST_PointN(ST_ExteriorRing(d.geom), 3)) - o.oy AS cy,
                       ST_Z(ST_PointN(ST_ExteriorRing(d.geom), 3)) - o.oz AS cz
                FROM o, LATERAL ST_Dump(ST_Tesselate(o.geom)) AS d
            ),
            tv AS (
                SELECT *, (ax*(by*cz - bz*cy) - ay*(bx*cz - bz*cx)
                           + az*(bx*cy - by*cx)) / 6.0 AS v
                FROM tri
            ),
            mom AS (
                SELECT MIN(ox) AS ox, MIN(oy) AS oy, MIN(oz) AS oz,
                       SUM(v) AS v_sum,
                       SUM(v * (ax + bx + cx) / 4.0) AS mx,
                       SUM(v * (ay + by + cy) / 4.0) AS my,
                       SUM(v * (az + bz + cz) / 4.0) AS mz
                FROM tv
            )
            SELECT
                GeometryType(g.geom),
                ST_NDims(g.geom),
                ST_IsClosed(g.geom),
                ST_ZMin(g.geom),
                ST_ZMax(g.geom),
                ST_Volume(ST_MakeSolid(g.geom)),
                m.v_sum,
                m.ox + m.mx / NULLIF(m.v_sum, 0),
                m.oy + m.my / NULLIF(m.v_sum, 0),
                m.oz + m.mz / NULLIF(m.v_sum, 0)
            FROM g, mom m;
        """
        self.cursor.execute("SAVEPOINT solid_probe;")
        try:
            self.cursor.execute(f"SET LOCAL statement_timeout = {self.HEAVY_GEOMETRY_TIMEOUT_MS};")
            self.cursor.execute(probe, (ewkt_geom,))
            (geom_type, ndims, is_closed, z_min, z_max, volume,
             face_volume, cx, cy, cz) = self.cursor.fetchone()
            self.cursor.execute("RELEASE SAVEPOINT solid_probe;")
            # SET LOCAL is transaction-scoped, not statement-scoped, and
            # survives RELEASE -- restore the strict guard so the widened
            # budget applies only to the heavy geometry statement above.
            self.cursor.execute("SET LOCAL statement_timeout = 1500;")
        except Exception as e:
            # Savepoint rollback, NOT a full transaction rollback -- the old
            # code called self.conn.rollback() inside the clash loop, which
            # silently discarded the SERIALIZABLE snapshot the clash check
            # depended on and let the subsequent INSERT commit under a fresh,
            # unchecked transaction.
            self.cursor.execute("ROLLBACK TO SAVEPOINT solid_probe;")
            self.cursor.execute("RELEASE SAVEPOINT solid_probe;")
            print(f"   ❌ Not a constructible 3D solid: {str(e)[:150]}")
            return None

        if geom_type != "POLYHEDRALSURFACE" or ndims != 3:
            print(f"   ❌ REJECTED: expected a 3D POLYHEDRALSURFACE, got {geom_type} "
                  f"({ndims}D).")
            return None
        if not is_closed:
            print("   ❌ REJECTED: shell is not closed -- an open surface bounds no volume.")
            return None
        if z_min is None or z_max is None or (z_max - z_min) < self.MIN_REGISTRABLE_Z_EXTENT_M:
            print(f"   ❌ REJECTED: vertical extent {(z_max or 0) - (z_min or 0):.3f}m is degenerate. "
                  f"This is a flat footprint, not a volume.")
            return None
        if volume is None or volume < self.MIN_REGISTRABLE_VOLUME_M3:
            print(f"   ❌ REJECTED: enclosed volume {volume or 0:.3f}m³ is below "
                  f"{self.MIN_REGISTRABLE_VOLUME_M3}m³ -- nothing volumetric to register.")
            return None

        # The face-summed volume behind the centroid must agree with SFCGAL's
        # volume; if not, the centroid is not trustworthy and nothing is
        # substituted for it.
        if (face_volume is None or cx is None or cy is None or cz is None
                or abs(face_volume - volume) > 1e-6 * max(abs(volume), 1.0)):
            print("   ❌ REJECTED: no consistent volumetric centroid can be measured "
                  "from this shell.")
            return None

        return (float(volume), float(z_min), float(z_max),
                float(cx), float(cy), float(cz))

    def _clash_against_shard(self, shard_id, ewkt_geom, epsilon):
        """
        Exact volumetric clash test for ONE candidate shard.

        Every failure mode here resolves CONSERVATIVELY to "clash". A timeout
        or SFCGAL error means we do not know whether two legal volumes overlap,
        and issuing a title on the strength of a check that did not complete is
        the worse error. Uses a savepoint so a failure on one candidate leaves
        the surrounding serializable transaction intact.
        """
        vol_query = """
            SELECT ST_Volume(
                ST_3DIntersection(
                    ST_MakeSolid(pss.shard_geom),
                    ST_MakeSolid(ST_GeomFromEWKT(%s))
                )
            )
            FROM property_spatial_shards pss
            WHERE pss.shard_id = %s;
        """
        self.cursor.execute("SAVEPOINT shard_check;")
        try:
            self.cursor.execute(f"SET LOCAL statement_timeout = {self.HEAVY_GEOMETRY_TIMEOUT_MS};")
            self.cursor.execute(vol_query, (ewkt_geom, shard_id))
            row = self.cursor.fetchone()
            self.cursor.execute("RELEASE SAVEPOINT shard_check;")
            self.cursor.execute("SET LOCAL statement_timeout = 1500;")
            if row is None or row[0] is None:
                return False
            return row[0] > epsilon
        except psycopg2.errors.SerializationFailure:
            raise
        except Exception as e:
            self.cursor.execute("ROLLBACK TO SAVEPOINT shard_check;")
            self.cursor.execute("RELEASE SAVEPOINT shard_check;")
            print(f"   ⚠️ Exact volume check failed on shard {shard_id} ({str(e)[:100]}). "
                  f"Treating as a clash -- an incomplete check is not a clearance.")
            return True

    def register_property(self, unit_id, ogc_3d_wkt, state_code, district_code,
                          tier_type="SURFACE", floor_level=0, is_demolition=False,
                          location_latitude=None, location_longitude=None,
                          location_accuracy_m=None, location_source=None,
                          location_crs='EPSG:4326'):
        """
        `location_*`: optional user-provided location, stored verbatim in
        property_registry as metadata/reference ONLY. It is never used to
        create, move, scale, trim or measure the geometry, never generates Z,
        and nothing is defaulted or invented when it is not supplied (NULL).


        `state_code`/`district_code`: accepted for caller compatibility only.
        They are NOT part of the 3D property ID (which is derived solely from
        the measured 3D centroid) and are not used by this method.

        FIX (race condition / TOCTOU): the actual clash check + insert
        below is "query existing shards, decide clear, then insert" as
        separate statements -- not atomic. Under READ COMMITTED (the
        Postgres default), two concurrent registrations whose volumes
        genuinely overlap could each run their clash query *before*
        either one's INSERT commits, each see zero conflicts, and both
        get written -- a real overlap that the volumetric check was
        supposed to prevent.
        Fix: run the whole check+insert as one SERIALIZABLE transaction.
        Postgres's serializable snapshot isolation detects exactly this
        pattern (both transactions read the same row range, both write
        into it) as write skew and aborts the loser with a
        `SerializationFailure` (SQLSTATE 40001) instead of letting it
        commit. We catch that specific error and retry with backoff --
        NOT the broad `except Exception` a few lines down, which would
        otherwise swallow it as an ordinary rejection.
        """
        for attempt in range(1, MAX_SERIALIZATION_RETRIES + 1):
            try:
                return self._register_property_once(unit_id, ogc_3d_wkt, state_code, district_code,
                                                     tier_type, floor_level, is_demolition,
                                                     location_latitude, location_longitude,
                                                     location_accuracy_m, location_source,
                                                     location_crs)
            except psycopg2.errors.SerializationFailure:
                self.conn.rollback()
                self.cursor.close()
                self.cursor = self.conn.cursor()
                if attempt == MAX_SERIALIZATION_RETRIES:
                    print(f"❌ REJECTED: {unit_id} lost {MAX_SERIALIZATION_RETRIES} consecutive concurrency "
                          f"races against other in-flight registrations near the same volume. "
                          f"Caller should retry the whole request.\n")
                    return None
                backoff_s = 0.05 * (2 ** (attempt - 1))
                print(f"   ⏳ Concurrent registration conflict near {unit_id}'s volume "
                      f"(attempt {attempt}/{MAX_SERIALIZATION_RETRIES}) — retrying in {backoff_s:.2f}s...")
                time.sleep(backoff_s)

    def _register_property_once(self, unit_id, ogc_3d_wkt, state_code, district_code,
                                tier_type="SURFACE", floor_level=0, is_demolition=False,
                                location_latitude=None, location_longitude=None,
                                location_accuracy_m=None, location_source=None,
                                location_crs='EPSG:4326'):
        print(f"🔍 Validating {unit_id} in True 3D Space (Tier: {tier_type}, Floor: {floor_level})...")
        # Scoped to THIS transaction only (resets to the session default,
        # READ COMMITTED, on commit/rollback) -- other methods on this
        # same connection (get_property_record, audit_infrastructure_clash)
        # don't pay the serializable-conflict-retry cost.
        self.cursor.execute("SET TRANSACTION ISOLATION LEVEL SERIALIZABLE;")
        try:
            if not ogc_3d_wkt or "()" in ogc_3d_wkt:
                print(f"❌ REJECTED: Geometry is empty.\n")
                return None

            ewkt_geom = f"SRID={CADASTRE_SRID};{ogc_3d_wkt}"
            CLASH_VOLUME_EPSILON = 0.01  # m^3

            # Phase 1 — fast bounding-box pre-filter (&&& operator)
            fast_query = """
                SELECT pr.unit_id, pss.shard_id, pss.shard_geom::text
                FROM property_spatial_shards pss
                JOIN property_registry pr ON pss.ulpin = pr.ulpin
                WHERE pss.shard_geom &&& ST_GeomFromEWKT(%s);
            """
            self.cursor.execute(fast_query, (ewkt_geom,))
            bbox_candidates = self.cursor.fetchall()

            clashes = []

            # ----------------------------------------------------------
            # Phase 1b — prove the incoming geometry is a real VOLUME.
            #
            # FIX (the biggest 2.5D hole in this file): the old code only
            # asked `ST_MakeSolid(...) IS NOT NULL`, which is true for
            # essentially any polyhedral surface, including a flat footprint
            # with a constant Z and including an unclosed shell. Such a
            # geometry was registered as a "3D property" while enclosing no
            # volume at all -- height present as a coordinate but carrying no
            # volumetric meaning, i.e. 2.5D data sitting in a 3D ledger. Worse,
            # it poisoned the ledger: every later clash check against that row
            # returns zero overlap, because a surface with no interior cannot
            # volumetrically intersect anything, so it became an invisible
            # parcel that nothing could ever collide with.
            # ----------------------------------------------------------
            measurement = self._measure_solid(ewkt_geom)
            if measurement is None:
                print(f"❌ REJECTED: {unit_id} is not a registrable 3D volume.\n")
                self.conn.rollback()
                return None
            volume_m3, z_min, z_max, cx, cy, cz = measurement
            print(f"   📐 Verified solid: {volume_m3:.2f}m³, Z {z_min:.2f}m → {z_max:.2f}m, "
                  f"volume centroid ({cx:.2f}, {cy:.2f}, {cz:.2f})")

            # ----------------------------------------------------------
            # Phase 2 — exact volumetric clash check.
            #
            # FIX: the SUBSURFACE branch above used to set bbox_candidates = []
            # and skip clash detection ENTIRELY for every tunnel, pipeline and
            # subsurface easement. The stated reason was SFCGAL slowness, but
            # the effect was that a metro tunnel could be driven straight
            # through a registered basement and the ledger would approve it
            # without looking -- the single largest correctness hole in a
            # system whose whole claim is volumetric conflict detection, and
            # precisely the check that distinguishes 3D cadastre from a 2D map
            # with a depth label. Subsurface assets are now checked like
            # everything else; the slowness is addressed where it belongs, by
            # giving the heavy statement a larger timeout budget
            # (HEAVY_GEOMETRY_TIMEOUT_MS) rather than by not asking.
            #
            # Note this is not a policy change about subsurface RIGHTS. If a
            # jurisdiction grants tunnels an easement below a given depth,
            # encode that as an explicit depth rule evaluated against z_min /
            # z_max -- a reviewable legal rule -- not as a silent skip.
            # ----------------------------------------------------------
            for cand_unit_id, shard_id, _ in bbox_candidates:
                if self._clash_against_shard(shard_id, ewkt_geom, CLASH_VOLUME_EPSILON):
                    clashes.append(cand_unit_id)

            if clashes:
                print(f"⚠️ CLASH DETECTED: True 3D boundary for {unit_id} physically intersects {clashes}!")

                if is_demolition:
                    # Demolition authorized: retire the conflicting legal
                    # records instead of aborting. ON DELETE CASCADE on
                    # property_spatial_shards / ownership_rights takes
                    # care of their dependent rows.
                    print(f"🏗️ DEMOLITION AUTHORIZED: Clearing {len(clashes)} "
                          f"conflicting propert{'y' if len(clashes) == 1 else 'ies'} "
                          f"to make way for {unit_id}: {clashes}")
                    self.cursor.execute(
                        "DELETE FROM property_registry WHERE unit_id = ANY(%s);",
                        (clashes,)
                    )
                else:
                    print(f"❌ REJECTED: Volumetric collision detected. Aborting transaction.\n")
                    self.conn.rollback()
                    self.cursor.close()
                    self.cursor = self.conn.cursor()
                    return None

            # TRUE 3D: the 3D property ID is spatially hashed from the exact volumetric
            # centroid (cx, cy, cz) measured by PostGIS from the solid's faces,
            # not from a hash of the WKT string. Z is treated identically to
            # X and Y -- it is a coordinate, not an attribute. Two units at
            # the same (X, Y) but different Z (stacked flats, basement vs.
            # ground) produce different centroid_z values and therefore
            # different spatial hashes, without any bounding-box reasoning.
            ulpin = self._allocate_ulpin(unit_id, cx, cy, cz,
                                        tier_type, floor_level)
            if ulpin is None:
                print(f"❌ REJECTED: Could not allocate a unique 3D property ID. Aborting.\n")
                self.conn.rollback()
                return None

            # True 3D Modification: Store the specific tier_type and floor_level
            # volume_m3 / z_min / z_max come from _measure_solid() -- i.e. from
            # PostGIS reading the geometry -- never from a caller-supplied
            # height. The solid remains the authoritative record; these are a
            # queryable projection of it.
            insert_master_query = """
                INSERT INTO property_registry
                    (ulpin, unit_id, tier_type, floor_level, boundary, volume_m3, z_min, z_max,
                     location_latitude, location_longitude, location_accuracy_m,
                     location_source, location_crs)
                VALUES (%s, %s, %s, %s, ST_GeomFromEWKT(%s), %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (ulpin) DO NOTHING;
            """
            # location_* are stored verbatim (metadata only); they are not
            # part of the geometry, the volume/z extents, or the clash checks.
            self.cursor.execute(
                insert_master_query,
                (ulpin, unit_id, tier_type, floor_level, ewkt_geom, volume_m3, z_min, z_max,
                 location_latitude, location_longitude, location_accuracy_m,
                 location_source, location_crs)
            )

            insert_shards_query = """
                INSERT INTO property_spatial_shards (ulpin, shard_geom)
                VALUES (%s, ST_GeomFromEWKT(%s));
            """
            self.cursor.execute(insert_shards_query, (ulpin, ewkt_geom))

            self.generate_mock_legal_data(ulpin, tier_type)
            self.conn.commit()

            print(f"✅ APPROVED: {unit_id} stored as a verified 3D solid "
                  f"({volume_m3:.2f}m³ enclosed, Z {z_min:.2f}m → {z_max:.2f}m).")
            print(f"🏛️ 3D property ID issued: {ulpin}\n")
            return ulpin

        except psycopg2.errors.SerializationFailure:
            # Postgres SSI most commonly reports the write-skew conflict
            # right here, at COMMIT. Let it propagate to register_property()'s
            # retry loop instead of being reported as a normal rejection.
            raise
        except Exception as e:
            self.conn.rollback()
            err_msg = str(e)[:300]
            print(f"⚠️ PostGIS Error: {err_msg}...\n")
            return None

    # ==============================================================
    # TRUE 3D DATA RETRIEVAL & AUDITING CAPABILITIES 
    # ==============================================================
    
    def get_property_record(self, ulpin):
        """Retrieves the full 3D spatial and legal record for a given ULPIN."""
        query = """
            SELECT 
                pr.unit_id,
                pr.tier_type,
                pr.floor_level,
                ST_AsEWKT(pr.boundary) as boundary_wkt,
                pr.volume_m3,
                pr.z_min,
                pr.z_max,
                p.full_name,
                p.party_type,
                ld.deed_number,
                ld.encumbrance_status,
                oright.right_type,
                oright.fractional_share,
                pr.location_latitude,
                pr.location_longitude,
                pr.location_accuracy_m,
                pr.location_source,
                pr.location_crs
            FROM property_registry pr
            LEFT JOIN ownership_rights oright ON pr.ulpin = oright.ulpin
            LEFT JOIN parties p ON oright.party_id = p.party_id
            LEFT JOIN legal_deeds ld ON oright.deed_id = ld.deed_id
            WHERE pr.ulpin = %s;
        """
        try:
            self.cursor.execute(query, (ulpin,))
            row = self.cursor.fetchone()
            if row:
                return {
                    "ulpin": ulpin,
                    "unit_id": row[0],
                    "tier": row[1],
                    "floor": row[2],
                    "spatial_boundary": row[3],
                    # Measured from the stored solid, not asserted by whoever
                    # filed it. A record that cannot report a volume is not a
                    # 3D record.
                    "volume_m3": float(row[4]) if row[4] is not None else None,
                    "z_min": float(row[5]) if row[5] is not None else None,
                    "z_max": float(row[6]) if row[6] is not None else None,
                    "height_m": (float(row[6]) - float(row[5]))
                                 if row[5] is not None and row[6] is not None else None,
                    "legal_owner": row[7],
                    "entity_type": row[8],
                    "deed_number": row[9],
                    "status": row[10],
                    "rights": row[11],
                    "share": float(row[12]) if row[12] else 1.0,
                    # User-provided location metadata, exactly as stored
                    # (reference only; not derived from or used for geometry).
                    "location_latitude": row[13],
                    "location_longitude": row[14],
                    "location_accuracy_m": row[15],
                    "location_source": row[16],
                    "location_crs": row[17],
                }
            return None
        except Exception as e:
            print(f"⚠️ Retrieval Error: {e}")
            self.conn.rollback()
            return None

    def audit_infrastructure_clash(self, ewkt_proposed_geometry):
        """
        Auditing Tool: Takes a proposed 3D volume (e.g. a new elevated highway)
        and returns all existing ULPINs that it volumetrically intersects with.
        """
        query = """
            SELECT DISTINCT pr.ulpin, pr.unit_id, pr.tier_type, p.full_name
            FROM property_spatial_shards pss
            JOIN property_registry pr ON pss.ulpin = pr.ulpin
            LEFT JOIN ownership_rights oright ON pr.ulpin = oright.ulpin
            LEFT JOIN parties p ON oright.party_id = p.party_id
            WHERE 
                pss.shard_geom &&& ST_GeomFromEWKT(%s)
                AND ST_Volume(
                    ST_3DIntersection(
                        ST_MakeSolid(pss.shard_geom),
                        ST_MakeSolid(ST_GeomFromEWKT(%s))
                    )
                ) > 0.1;
        """
        try:
            self.cursor.execute(query, (ewkt_proposed_geometry, ewkt_proposed_geometry))
            conflicts = self.cursor.fetchall()
            results = []
            for row in conflicts:
                results.append({
                    "ulpin": row[0],
                    "unit_id": row[1],
                    "tier": row[2],
                    "owner": row[3]
                })
            return results
        except Exception as e:
            print(f"⚠️ Auditing Error: {e}")
            self.conn.rollback()
            return []