import os
import re
import psycopg2
import config

# ==========================================
# FIX: must match CADASTRE_SRID in db_engine.py, or every query in
# GlobalOverlapEngine will hit the same "mixed SRID geometries" /
# "does not match column SRID" errors that main.py's pipeline hit.
# Now sourced from config.py (single source of truth) instead of its
# own independent env-var read, so the two modules can't disagree.
# ==========================================
CADASTRE_SRID = config.CADASTRE_SRID


# ==========================================
# 0. AUTHORITATIVE 3D INPUT
# ==========================================
# Every geometry this module handles is an ALREADY-BUILT 3D solid (a closed
# POLYHEDRALSURFACE Z, as EWKT or WKT-Z): measured or generated upstream.
# _as_ewkt_3d() re-validates every input as a genuine closed, valid 3D solid
# via PostGIS/SFCGAL before it is used, and rejects it otherwise. Nothing here
# builds, extrudes, positions, trims, pairs or repairs geometry. z_base /
# z_top / Z ranges / 2D footprints / floor counts or heights are not inputs
# to any geometry decision in this module.
_SOLID_WKT_RE = re.compile(r"^\s*POLYHEDRALSURFACE\s*Z\b", re.IGNORECASE)


def _validate_solid_3d(ewkt, conn):
    """
    Read-only PostGIS/SFCGAL validity gate for a submitted 3D solid. Raises
    ValueError if it is not a genuine closed, valid, positively-oriented 3D
    POLYHEDRALSURFACE. The geometry is only measured: it is never
    repaired, made into a solid, reoriented, simplified or clipped, and
    nothing is returned but the verdict.

    What decides it:
      - SFCGAL's own surface validation, run by ST_Tesselate (used read-only
        to list the faces): raises on inconsistent face orientation,
        self-intersection, disconnected shells, and degenerate faces.
      - ST_IsClosed: rejects open shells.
      - The signed volume of the face-listed shell (divergence theorem, coords
        offset to the shell's min corner purely to limit float error) must be
        > 0. A zero volume is degenerate; a NEGATIVE one is an inside-out
        shell, which the ST_MakeSolid/ST_3DIntersection test would score as
        negative overlap and so silently clear.
    """
    probe = """
        WITH g AS (SELECT ST_GeomFromEWKT(%s) AS geom),
        o AS (
            SELECT geom, ST_XMin(geom) AS ox, ST_YMin(geom) AS oy,
                   ST_ZMin(geom) AS oz
            FROM g
        ),
        tri AS (
            SELECT ST_X(ST_PointN(ST_ExteriorRing(d.geom), 1)) - o.ox AS ax,
                   ST_Y(ST_PointN(ST_ExteriorRing(d.geom), 1)) - o.oy AS ay,
                   ST_Z(ST_PointN(ST_ExteriorRing(d.geom), 1)) - o.oz AS az,
                   ST_X(ST_PointN(ST_ExteriorRing(d.geom), 2)) - o.ox AS bx,
                   ST_Y(ST_PointN(ST_ExteriorRing(d.geom), 2)) - o.oy AS by,
                   ST_Z(ST_PointN(ST_ExteriorRing(d.geom), 2)) - o.oz AS bz,
                   ST_X(ST_PointN(ST_ExteriorRing(d.geom), 3)) - o.ox AS cx,
                   ST_Y(ST_PointN(ST_ExteriorRing(d.geom), 3)) - o.oy AS cy,
                   ST_Z(ST_PointN(ST_ExteriorRing(d.geom), 3)) - o.oz AS cz
            FROM o, LATERAL ST_Dump(ST_Tesselate(o.geom)) AS d
        )
        SELECT GeometryType(g.geom), ST_NDims(g.geom), ST_IsClosed(g.geom),
               (SELECT SUM((ax*(by*cz - bz*cy) - ay*(bx*cz - bz*cx)
                            + az*(bx*cy - by*cx)) / 6.0) FROM tri)
        FROM g;
    """
    # A failed statement aborts the caller's transaction; isolate it in a
    # savepoint (not possible, and not needed, on an autocommit connection).
    use_savepoint = not conn.autocommit
    with conn.cursor() as cur:
        if use_savepoint:
            cur.execute("SAVEPOINT solid_validate;")
        try:
            cur.execute(probe, (ewkt,))
            geom_type, ndims, is_closed, signed_volume = cur.fetchone()
            if use_savepoint:
                cur.execute("RELEASE SAVEPOINT solid_validate;")
        except Exception as e:
            if use_savepoint:
                cur.execute("ROLLBACK TO SAVEPOINT solid_validate;")
                cur.execute("RELEASE SAVEPOINT solid_validate;")
            raise ValueError(f"Not a valid 3D solid: {str(e)[:150]}") from e

    if geom_type != "POLYHEDRALSURFACE" or ndims != 3:
        raise ValueError(
            f"Not a valid 3D solid: expected a 3D POLYHEDRALSURFACE, got "
            f"{geom_type} ({ndims}D).")
    if not is_closed:
        raise ValueError("Not a valid 3D solid: shell is not closed.")
    if signed_volume is None or signed_volume <= 0:
        raise ValueError(
            "Not a valid 3D solid: shell is degenerate (zero volume) or "
            "inside-out (negative volume).")


def _as_ewkt_3d(geometry_3d, conn=None):
    """
    Validates that `geometry_3d` is a genuine closed, valid 3D solid --
    a POLYHEDRALSURFACE Z (EWKT or WKT-Z) -- and returns it as EWKT.

    Syntax is checked first, then the shell itself is validated in
    PostGIS/SFCGAL (see _validate_solid_3d). Invalid, open, self-intersecting,
    inside-out or degenerate input raises; it is never repaired. Only a
    missing SRID tag is added (CADASTRE_SRID); the coordinates are passed
    through untouched. Anything else -- a 2D polygon, a shapely object, a
    footprint plus heights -- is rejected, never converted.

    `conn`: open psycopg2 connection to validate on (left open; its
    transaction is left to the caller). If None, a short-lived one is opened
    from config.PG_DSN_KWARGS.
    """
    if not isinstance(geometry_3d, str) or not geometry_3d.strip():
        raise TypeError(
            "3D geometry must be an EWKT / WKT-Z POLYHEDRALSURFACE string of an "
            "already-built solid. 2D footprints with z_base/z_top are not "
            "accepted and are never extruded.")
    text = geometry_3d.strip()
    body = text.split(";", 1)[1] if text.upper().startswith("SRID=") else text
    if "()" in body or not _SOLID_WKT_RE.match(body):
        raise ValueError(
            "Expected a non-empty POLYHEDRALSURFACE Z solid (EWKT or WKT-Z).")
    ewkt = text if text.upper().startswith("SRID=") else f"SRID={CADASTRE_SRID};{text}"

    own_conn = conn is None
    if own_conn:
        conn = psycopg2.connect(**config.PG_DSN_KWARGS)
    try:
        _validate_solid_3d(ewkt, conn)
    finally:
        if own_conn:
            conn.rollback()
            conn.close()
    return ewkt


# ==========================================
# 1. LOCAL ENGINE (Pre-processing, true 3D)
# ==========================================
def detect_local_encroachments(rooms, min_overlap_volume_m3=0.05, conn=None):
    """
    Checks for overlaps WITHIN a local batch of units before sending them to
    the registry, by real 3D intersection of their solids.

    `rooms`: iterable of {"id": ..., "geometry_3d": <EWKT / WKT-Z
    POLYHEDRALSURFACE of the built solid>}. Extra keys are ignored.

    FIX (strict 3D): this used to treat each unit as a vertical prism and
    report footprint-intersection area x shared Z thickness, i.e. it
    reconstructed the volume from a 2D outline and a height range. It now
    measures the actual solid: `&&&` is only the broad-phase filter for
    candidate pairs, and ST_Volume(ST_3DIntersection(...)) on the two solids
    is the authoritative overlap volume. No footprint, z_base or z_top is read.

    Runs in PostGIS/SFCGAL on transient parameters (no table is touched or
    written). `conn`: optional open psycopg2 connection (left open, and its
    transaction left to the caller); if None, one is opened from
    config.PG_DSN_KWARGS for the call.

    Every input solid is first validated by _as_ewkt_3d (closed, valid,
    non-degenerate); an invalid one raises ValueError naming the room.

    Returns a list of {"unit_1", "unit_2", "overlap_volume_m3"}, largest
    overlap first. Fails safe like db_engine._clash_against_shard: if SFCGAL
    cannot evaluate a candidate pair (e.g. timeout), the pair is NOT cleared -- it is reported with
    "overlap_volume_m3": None and "unverified": True, listed first. The
    geometry is never repaired to make the test pass.
    """
    rooms = list(rooms)
    for room in rooms:
        if "geometry_3d" not in room:
            raise ValueError(
                f"room {room.get('id')!r} has no 'geometry_3d'. A 2D 'polygon' "
                f"with 'z_base'/'z_top' is not geometry and is not extruded here.")
    print(f"🧊 True-3D overlap check for {len(rooms)} property solids...")

    broad_phase = """
        WITH s AS (
            SELECT idx, ST_GeomFromEWKT(ewkt) AS geom
            FROM unnest(%s::text[]) WITH ORDINALITY AS t(ewkt, idx)
        )
        SELECT a.idx, b.idx
        FROM s a JOIN s b ON a.idx < b.idx AND a.geom &&& b.geom;
    """
    exact = """
        SELECT ST_Volume(ST_3DIntersection(
            ST_MakeSolid(ST_GeomFromEWKT(%s)),
            ST_MakeSolid(ST_GeomFromEWKT(%s))));
    """

    own_conn = conn is None
    if own_conn:
        conn = psycopg2.connect(**config.PG_DSN_KWARGS)
    encroachments = []
    try:
        # Every solid is validated (closed, valid, non-degenerate) before it
        # takes part in any test; an invalid one is rejected, not repaired.
        ewkts = []
        for room in rooms:
            try:
                ewkts.append(_as_ewkt_3d(room["geometry_3d"], conn))
            except (TypeError, ValueError) as e:
                raise type(e)(f"room {room.get('id')!r}: {e}") from e

        with conn.cursor() as cur:
            cur.execute(broad_phase, (ewkts,))
            candidates = cur.fetchall()

            for ia, ib in candidates:
                room_a, room_b = rooms[ia - 1], rooms[ib - 1]
                entry = {"unit_1": room_a["id"], "unit_2": room_b["id"]}
                cur.execute("SAVEPOINT overlap_pair;")
                try:
                    cur.execute(exact, (ewkts[ia - 1], ewkts[ib - 1]))
                    volume = cur.fetchone()[0]
                    cur.execute("RELEASE SAVEPOINT overlap_pair;")
                except Exception as e:
                    cur.execute("ROLLBACK TO SAVEPOINT overlap_pair;")
                    cur.execute("RELEASE SAVEPOINT overlap_pair;")
                    print(f"   ⚠️ 3D intersection failed for {entry['unit_1']} / "
                          f"{entry['unit_2']} ({str(e)[:100]}). Not cleared.")
                    entry.update({"overlap_volume_m3": None, "unverified": True})
                    encroachments.append(entry)
                    continue

                if volume is not None and volume > min_overlap_volume_m3:
                    entry["overlap_volume_m3"] = round(volume, 3)
                    encroachments.append(entry)
    finally:
        if own_conn:
            conn.rollback()
            conn.close()

    # Unverified pairs first (an incomplete check is not a clearance), then
    # worst encroachment first -- by volume, which is the thing in dispute.
    encroachments.sort(key=lambda e: (e["overlap_volume_m3"] is not None,
                                      -(e["overlap_volume_m3"] or 0)))
    return encroachments


# ==========================================
# Literal, already-built test solids (closed POLYHEDRALSURFACE Z). They are
# fixtures, not generated from footprints or heights.
# ==========================================
FIXTURE_A = "POLYHEDRALSURFACE Z (((0 0 0,0 4 0,4 4 0,4 0 0,0 0 0)),((0 0 3,4 0 3,4 4 3,0 4 3,0 0 3)),((0 0 0,4 0 0,4 0 3,0 0 3,0 0 0)),((4 0 0,4 4 0,4 4 3,4 0 3,4 0 0)),((4 4 0,0 4 0,0 4 3,4 4 3,4 4 0)),((0 4 0,0 0 0,0 0 3,0 4 3,0 4 0)))"
FIXTURE_B = "POLYHEDRALSURFACE Z (((3 0 0,3 4 0,7 4 0,7 0 0,3 0 0)),((3 0 3,7 0 3,7 4 3,3 4 3,3 0 3)),((3 0 0,7 0 0,7 0 3,3 0 3,3 0 0)),((7 0 0,7 4 0,7 4 3,7 0 3,7 0 0)),((7 4 0,3 4 0,3 4 3,7 4 3,7 4 0)),((3 4 0,3 0 0,3 0 3,3 4 3,3 4 0)))"
FIXTURE_C = "POLYHEDRALSURFACE Z (((0 0 3,0 4 3,4 4 3,4 0 3,0 0 3)),((0 0 6,4 0 6,4 4 6,0 4 6,0 0 6)),((0 0 3,4 0 3,4 0 6,0 0 6,0 0 3)),((4 0 3,4 4 3,4 4 6,4 0 6,4 0 3)),((4 4 3,0 4 3,0 4 6,4 4 6,4 4 3)),((0 4 3,0 0 3,0 0 6,0 4 6,0 4 3)))"
FIXTURE_D = "POLYHEDRALSURFACE Z (((0 0 2.8,0 4 2.8,4 4 2.8,4 0 2.8,0 0 2.8)),((0 0 5,4 0 5,4 4 5,0 4 5,0 0 5)),((0 0 2.8,4 0 2.8,4 0 5,0 0 5,0 0 2.8)),((4 0 2.8,4 4 2.8,4 4 5,4 0 5,4 0 2.8)),((4 4 2.8,0 4 2.8,0 4 5,4 4 5,4 4 2.8)),((0 4 2.8,0 0 2.8,0 0 5,0 4 5,0 4 2.8)))"


# ==========================================
# 2. GLOBAL POSTGIS ENGINE (Database Audit)
# ==========================================
class GlobalOverlapEngine:
    def __init__(self):
        # FIX: was hardcoded to dbname="cadastre_db", user="postgres",
        # password="mayank9431", host="localhost" regardless of
        # environment. Now shares config.py's single source of DB
        # credentials with the rest of the pipeline, so this class can't
        # silently drift onto a different database than db_engine.py.
        self.conn = psycopg2.connect(**config.PG_DSN_KWARGS)

    def convert_to_ewkt_3d(self, geometry_3d, z_base=None, z_top=None):
        """
        DISABLED as a geometry-construction path. This used to extrude a 2D
        shapely polygon between z_base and z_top; that built 3D geometry from
        a footprint and a height range and is gone. It now only accepts an
        already-built 3D solid (EWKT / WKT-Z POLYHEDRALSURFACE), tags a
        missing SRID with CADASTRE_SRID, and returns the EWKT string
        unchanged otherwise.

        Breaking change, deliberately loud: the old return value was
        (ewkt, height). It is now just the EWKT string, and passing a
        polygon or z_base/z_top raises instead of extruding.
        """
        if z_base is not None or z_top is not None:
            raise TypeError(
                "z_base/z_top are no longer accepted: 3D geometry is never "
                "built from a footprint and a height range.")
        return _as_ewkt_3d(geometry_3d, self.conn)

    def check_global_encroachment(self, ewkt_3d_solid, height=None, min_overlap_volume_m3=0.01):
        """
        Queries the registry with a conservative 3D bounding-box (&&&) broad
        phase on property_registry.boundary, then decides each candidate by
        exact 3D volume intersection.

        `ewkt_3d_solid`: the already-built solid (EWKT / WKT-Z
        POLYHEDRALSURFACE). `height` is retained only for signature
        compatibility and must be None: nothing is extruded.

        FIX (strict 3D): the target used to be built in SQL with
        ST_Extrude(polygon, 0, 0, height). The submitted solid is now used as
        given. ST_3DIntersects was also dropped from the WHERE clause -- a
        second, non-authoritative predicate between the index and the exact
        test. `&&&` only narrows candidates; ST_Volume(ST_3DIntersection(...))
        > epsilon is the sole clash decision, matching db_engine.py. A
        candidate SFCGAL cannot evaluate raises (nothing is cleared silently).

        AUTHORITY / BROAD-PHASE INVARIANT: both the `&&&` candidate filter and
        the exact test use property_registry.boundary, the full authoritative
        solid; its 3D bounding box conservatively covers all of it, so no
        clash can be missed at the filter. property_spatial_shards.shard_geom
        is NOT used here: nothing guarantees shards cover the whole boundary,
        and a filter on shards that omit any part of it could drop a real
        clash. Shards may only ever be used as a filter if they are guaranteed
        to fully cover the boundary (they may partition it for indexing, never
        omit any of it), and they never decide the overlap result.

        Shared volume, not contact, is what counts: stacked storeys and
        adjacent flats legitimately touch, and that intersects in zero
        volume.
        """
        if height is not None:
            raise TypeError(
                "height is no longer accepted: the solid is used as given and "
                "is never extruded.")
        ewkt_3d_solid = _as_ewkt_3d(ewkt_3d_solid, self.conn)

        query = """
            WITH target AS (
                SELECT ST_GeomFromEWKT(%s) AS geom
            )
            SELECT DISTINCT pr.unit_id
            FROM property_registry pr
            CROSS JOIN target
            WHERE
                pr.boundary &&& target.geom
                AND ST_Volume(
                        ST_3DIntersection(
                            ST_MakeSolid(pr.boundary),
                            ST_MakeSolid(target.geom)
                        )
                    ) > %s;
        """

        with self.conn.cursor() as cursor:
            cursor.execute(query, (ewkt_3d_solid, min_overlap_volume_m3))
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

                # Diagnostic probe only: a literal, already-built solid.
                test_ewkt = f"SRID={CADASTRE_SRID};{FIXTURE_A}"

                query = """
                    EXPLAIN ANALYZE
                    SELECT DISTINCT pr.unit_id
                    FROM property_spatial_shards pss
                    JOIN property_registry pr ON pss.ulpin = pr.ulpin
                    WHERE
                        pss.shard_geom &&& ST_GeomFromEWKT(%s);
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
        {"id": "ULPIN_1001", "geometry_3d": FIXTURE_A},
        # Side-by-side with 1001: shares a genuine slab of volume.
        {"id": "ULPIN_1002", "geometry_3d": FIXTURE_B},
        # Directly above 1001: shares a slab plane only -> zero shared volume.
        {"id": "ULPIN_1003", "geometry_3d": FIXTURE_C},
        # A shallow mezzanine genuinely intruding into 1001's airspace.
        {"id": "ULPIN_1004", "geometry_3d": FIXTURE_D},
    ]

    # 1. Local Check
    try:
        local_conflicts = detect_local_encroachments(test_rooms)
        print("\n🚨 Local Spatial Conflicts Detected:")
        if not local_conflicts:
            print(" (none)")
        for conflict in local_conflicts:
            vol = conflict["overlap_volume_m3"]
            vol_txt = "UNVERIFIED (3D intersection could not be evaluated)" if vol is None else f"{vol} m³"
            print(f" - {conflict['unit_1']} encroaches on {conflict['unit_2']}: {vol_txt}.")
    except Exception as e:
        print(f"⚠️ Could not run local 3D check (Make sure PostgreSQL/SFCGAL is available): {e}")

    # 2. Global Check & R-Tree Diagnostic
    print("\n🌍 Connecting to PostGIS Global Database for verification...")
    try:
        global_engine = GlobalOverlapEngine()

        # Test the database collision logic
        room_to_test = test_rooms[0]
        ewkt_solid = global_engine.convert_to_ewkt_3d(room_to_test["geometry_3d"])

        is_clear, db_conflicts = global_engine.check_global_encroachment(ewkt_solid)

        if is_clear:
            print(f"✅ {room_to_test['id']} is clear to be registered in the global cadastre.")
        else:
            print(f"❌ {room_to_test['id']} collides with existing global properties: {db_conflicts}")

        # RUN THE R-TREE DIAGNOSTIC
        global_engine.test_rtree_index()

    except Exception as e:
        print(f"⚠️ Could not test global engine (Make sure PostgreSQL is running): {e}")