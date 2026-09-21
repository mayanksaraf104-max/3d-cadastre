import math
import os
import types
from typing import NamedTuple, Optional
import psycopg2
import laspy
import numpy as np
from pyproj import CRS, Transformer
from shapely.geometry import Polygon, Point, box
from shapely.prepared import prep
import config
# 3D-FIX (roof evidence): PolynomialFeatures/StandardScaler/Ridge/Pipeline
# used to fit a single Z = f(X, Y) roof surface in calculate_z_bounds().
# That model is gone -- see this module's ROOF EVIDENCE note near
# calculate_z_bounds() -- so these imports have no remaining use in this
# file. If you're looking for the old roof-fit code, it is not hiding
# elsewhere: it has been replaced by returning the measured points
# themselves.

# Must match CADASTRE_SRID in db_engine.py. `lidar_tile_index.tile_boundary`
# needs to be stored in this same SRID for ST_Intersects to mean anything.
# Sourced from config.py -- see that module's docstring.
CADASTRE_SRID = config.CADASTRE_SRID

# ASPRS LAS standard classification code for terrain/ground returns.
#
# TERRAIN IS CONTEXT, NOT FLOOR GEOMETRY. This code is used ONLY by
# get_terrain_context_points(), which exists for site/grade context
# (e.g. reporting how far a measured ground-floor slab sits above
# surrounding grade). It is deliberately unreachable from any floor
# extraction path in this module. The ground under a building is a
# different physical surface from the building's ground-floor slab:
# it sits below the foundation, footings, fill and screed, it follows
# site grading rather than the finished floor level, and it is measured
# by a sensor that never saw the slab at all. Substituting it for a
# ground-floor slab is fabricated elevation wearing a measurement's
# name -- the same class of bug as the old hardcoded z_base=0.0.
GROUND_CLASSIFICATION_CODE = 2

# Project-specific LAS classification code reserved for MEASURED
# interior/structural floor-slab returns (e.g. from an indoor/TLS
# mobile-mapping scan that can see through openings to hit an actual
# slab surface, or a point-cloud classifier that explicitly labels slab
# returns as such). This is deliberately NOT the generic ASPRS "Building"
# class (6) -- that code lumps together roof, facade, balcony, ledge and
# parapet returns, so accepting it wholesale for slab evidence would let
# exactly those substitutes back in.
#
# This is now the ONLY source of floor geometry in this module, for
# EVERY floor including the ground floor (floor_index == 0). Sourced
# from config.py; if your point-cloud classifier doesn't produce a
# dedicated slab code, leave this unset (config.py simply won't define
# it) and every floor lookup -- ground and upper alike -- will correctly
# return None instead of fabricating a slab out of terrain.
SLAB_CLASSIFICATION_CODE = getattr(config, "SLAB_CLASSIFICATION_CODE", None)

# Optional, explicitly configured authoritative per-point semantic codes for
# ROOF and WALL returns, used ONLY to fill metadata["structure_semantic_labels"].
# Unset (the default) means that label is never produced: roof/wall are never
# inferred from Z, height, normals, orientation, position, proximity,
# connectedness or "whatever is left over", and the generic ASPRS "Building"
# class (6) does NOT automatically mean roof or wall. They do not change which
# points are in roof_points_xyz / structure_points_xyz.
ROOF_CLASSIFICATION_CODE = getattr(config, "ROOF_CLASSIFICATION_CODE", None)
WALL_CLASSIFICATION_CODE = getattr(config, "WALL_CLASSIFICATION_CODE", None)

# ---------------------------------------------------------------------
# FLOOR IDENTITY. A classification code says "this point is a slab
# return". It does NOT say WHICH storey's slab. Separating floors
# therefore requires explicit, recorded floor identity -- never an
# inference from elevation.
#
# There are exactly three admissible sources, all of them measurements
# or provenance recorded by whoever captured the scan:
#
#   1. FLOOR_ID_DIMENSION -- the name of a per-point LAS/LAZ dimension
#      (usually an extra-bytes field, e.g. "floor_id" / "storey")
#      written by the capture or classification stage, carrying the
#      storey each point was observed on. Highest fidelity: it survives
#      tiles that span several floors.
#   2. TILE_FLOOR_COLUMN -- a column on `lidar_tile_index` recording
#      which storey a given tile/scan file covers. Every point read from
#      that file inherits that floor id. Use when a capture produced one
#      file per floor and the index records it.
#   3. An explicit per-floor `floor_scan_path` passed by the caller,
#      asserting "this file is floor N's scan".
#
# What is NOT admissible, and is deliberately impossible to express
# anywhere in this module: splitting one cloud's slab returns into
# storeys by Z range, storey height, cumulative height, expected_z, or
# clustering/histogram peaks in elevation. Any of those would manufacture
# floor identity out of the very number the floor is supposed to supply
# -- circular, and the fabricated-elevation bug in its subtlest form. If
# none of the three sources above is available, floor identity is
# UNKNOWN, and an unknown floor returns None rather than every slab in
# the building.
FLOOR_ID_DIMENSION = getattr(config, "FLOOR_ID_DIMENSION", None)
TILE_FLOOR_COLUMN = getattr(config, "TILE_FLOOR_COLUMN", "floor_index")

# Optional {floor_index: value_stored_in_the_data} map, for captures
# whose floor labels aren't plain storey integers (e.g. {0: "G", 1: "01"}
# or a basement encoded as 100). Identity mapping when unset -- the
# stored value is compared to floor_index directly.
FLOOR_ID_MAP = getattr(config, "FLOOR_ID_MAP", None)

# ---------------------------------------------------------------------
# ACQUISITION BROAD-PHASE (a performance device, never a boundary).
#
# The blueprint / 2D footprint is a DRAFTED outline. It is not a
# measurement, and it must never decide which measured returns survive.
# Roofs and slabs overhang, cantilever, carry eaves, jetties, balconies
# and setbacks past the drafted wall line, so neither a crop at
# `poly.bounds` nor a tile lookup that only accepts tiles touching
# `poly` may be allowed to define extent.
#
# The footprint therefore seeds only a LiDAR XY acquisition window, and
# that window is GROWN FROM THE DATA, not sized by a constant:
#
#   * ROOF_ACQUISITION_MARGIN_M is the grow STEP (metres in the
#     CADASTRE_SRID's units). Starting at the footprint's axis-aligned
#     bounds, the window widens one step at a time for as long as the
#     ring just outside it still contains measured evidence (any return
#     not classified ground -- or any return at all when the cloud has no
#     classification). It stops at the first step whose ring is empty:
#     a real gap in the measurements, not a number chosen in advance.
#     A cantilever longer than the step is therefore still captured, as
#     long as the measured points are continuous to within one step.
#   * ACQUISITION_MAX_EXPANSIONS is a memory/performance SAFETY VALVE on
#     how many steps that growth may take (points and tiles are only
#     pulled off disk out to footprint + (max+1) steps). It is not a
#     claim about building size. If growth is still finding evidence when
#     the valve trips, that is reported loudly (a printed warning and
#     metadata["acquisition_truncated"] = True) rather than silently
#     treating the valve as the building's edge. Raise it (or the step)
#     when that warning appears.
#   * UNBOUNDED MODE: pass `acquisition_max_expansions=math.inf` (or set
#     config.ACQUISITION_MAX_EXPANSIONS = math.inf) and no valve exists:
#     collection re-reads with a doubling valve until the ring test finds
#     a genuinely empty ring, so extent is decided by the measurements
#     alone. Costs memory/IO proportional to the connected evidence.
#   * When a finite valve does trip, the returned cloud ends at the
#     window edge. That edge is an ACQUISITION LIMIT, not a measured
#     building edge: metadata["acquisition_extent_complete"] is False and
#     nothing may treat the window boundary as geometry.
#
# No per-point test against the footprint polygon is applied to roof or
# slab evidence anywhere in this module. The PostGIS tile lookup keeps
# using ST_Intersects against the (larger) acquisition box. Overrides per
# call: `acquisition_margin_m=`, `acquisition_max_expansions=`.
#
# Trade-off, stated plainly: a window that follows the data also follows
# touching neighbours (terraces, adjoining roofs) out to the safety
# valve. Separating them is the job of the downstream 3D segmentation,
# working from measured points -- not of an XY rectangle.
# ---------------------------------------------------------------------
ROOF_ACQUISITION_MARGIN_M = float(getattr(config, "ROOF_ACQUISITION_MARGIN_M", 3.0))
_cfg_max_exp = getattr(config, "ACQUISITION_MAX_EXPANSIONS", 8)
# math.inf = unbounded acquisition (no safety valve); otherwise an int >= 1.
ACQUISITION_MAX_EXPANSIONS = (_cfg_max_exp if _cfg_max_exp == math.inf
                              else int(_cfg_max_exp))


def _acquisition_bbox(poly, margin_m=None):
    """
    Axis-aligned XY window (minx, miny, maxx, maxy): the footprint's
    bounds expanded by `margin_m` on every side. Used for the read
    window (which tiles to open, which points to pull off disk) and as
    the grow step in _grow_window_to_evidence. PERFORMANCE ONLY -- never
    a boundary, never a clip for measured geometry.
    """
    if margin_m is None:
        margin_m = ROOF_ACQUISITION_MARGIN_M
    margin_m = float(margin_m)
    if margin_m < 0:
        raise ValueError(
            f"acquisition margin must be >= 0, got {margin_m}. A negative "
            f"margin would let the footprint trim measured roof geometry."
        )
    minx, miny, maxx, maxy = poly.bounds
    return (minx - margin_m, miny - margin_m, maxx + margin_m, maxy + margin_m)


def _bbox_mask(x, y, bb):
    minx, miny, maxx, maxy = bb
    return (x >= minx) & (x <= maxx) & (y >= miny) & (y <= maxy)


def _grow_window_to_evidence(pts_x, pts_y, pts_cls, poly, step_m, max_steps):
    """
    Data-driven acquisition window (see ACQUISITION BROAD-PHASE above).

    Starts at the footprint's bounds and widens by `step_m` while the
    ring just outside the window still holds measured evidence. Stops at
    the first empty ring. `pts_*` must already be read out to at least
    footprint + (max_steps + 1) steps so the ring test is meaningful at
    the valve.

    Evidence = any return not classified ground; with no classification,
    every return (ground can't be told from building, so growth is then
    limited only by the safety valve -- reported, not hidden).

    Returns (window_bbox, steps_taken, truncated). `truncated` is True
    only when the safety valve stopped growth while evidence was still
    present just outside the window.
    """
    if pts_cls is None:
        ex, ey = pts_x, pts_y
    else:
        keep = pts_cls != GROUND_CLASSIFICATION_CODE
        ex, ey = pts_x[keep], pts_y[keep]

    minx, miny, maxx, maxy = poly.bounds
    steps = 0
    while True:
        inner = (minx - steps * step_m, miny - steps * step_m,
                 maxx + steps * step_m, maxy + steps * step_m)
        outer = (inner[0] - step_m, inner[1] - step_m,
                 inner[2] + step_m, inner[3] + step_m)
        ring_has_evidence = bool(np.any(_bbox_mask(ex, ey, outer)
                                        & ~_bbox_mask(ex, ey, inner)))
        if not ring_has_evidence:
            return inner, steps, False
        if steps >= max_steps:
            return inner, steps, True
        steps += 1


def roof_xy_for_visualization(roof_points_xyz):
    """
    2D projection of measured roof points, for PLOTTING / 2D VALIDATION
    ONLY (e.g. overlaying evidence on the blueprint to eyeball
    alignment).

    NOT a geometry source. Do not build a hull, outline, boundary,
    footprint, or any reconstruction input from this: the projection
    throws away Z, which is exactly what collapses overhangs, undercuts
    and multi-layer roofs into one indistinguishable blob. Boundary
    generation and reconstruction must consume `roof_points_xyz`.

    Returns a read-only (M, 2) view of the same array (no copy), or None.
    """
    if roof_points_xyz is None:
        return None
    view = np.asarray(roof_points_xyz)[:, :2].view()
    view.setflags(write=False)
    return view


def _resolve_floor_id(floor_index):
    """
    Translates a caller's `floor_index` into the value actually recorded
    in the data, via config.FLOOR_ID_MAP. Identity when no map is
    configured. Raises KeyError if a map exists but has no entry for
    this floor -- an unmapped floor is an unknown floor, and guessing
    that floor 3 is stored as `3` when the map says otherwise is exactly
    the kind of assumption this module refuses to make.
    """
    if FLOOR_ID_MAP is None:
        return floor_index
    if floor_index not in FLOOR_ID_MAP:
        raise KeyError(
            f"floor_index {floor_index!r} is not present in config.FLOOR_ID_MAP "
            f"({sorted(FLOOR_ID_MAP)!r}). Add it rather than assuming its "
            f"stored label."
        )
    return FLOOR_ID_MAP[floor_index]


def _floor_id_match_mask(pts_floor, floor_index):
    """
    Boolean mask of which points carry THIS floor's recorded id.
    Comparison is on the recorded label, not on elevation. Handles both
    numeric and string/bytes floor labels.
    """
    target = _resolve_floor_id(floor_index)
    pts_floor = np.asarray(pts_floor)

    if pts_floor.dtype.kind in "US":
        return pts_floor == str(target)
    if pts_floor.dtype.kind == "S":
        return pts_floor == str(target).encode()

    try:
        return pts_floor == type(pts_floor.flat[0])(target)
    except (TypeError, ValueError):
        return pts_floor == target


class ZEvidence(NamedTuple):
    """
    Return type of calculate_z_bounds(): measured evidence, plus metadata
    that is kept strictly apart from it.

    floor_points_xyz : (N, 3) raw measured slab points for the ONE storey
                       named by `floor_index`, or None. Storey attribution
                       needs recorded floor identity; it is a subset view
                       of structure_points_xyz, not the way to get slabs.
    roof_points_xyz  : (M, 3) raw measured CANDIDATE / exposed-surface
                       points, or None: the non-ground, non-slab returns.
                       It is NOT roof-semantic (it can include facade,
                       balcony, ledge, parapet or any other exposed
                       surface) unless metadata["roof_points_semantics"]
                       == "roof", in which case it contains ONLY points
                       carrying the explicitly configured
                       ROOF_CLASSIFICATION_CODE. Excludes ground-
                       and slab-classified returns BY DEFINITION; those
                       slabs are NOT lost -- they are in
                       structure_points_xyz.
    structure_points_xyz : (K, 3) EVERY measured return in the acquisition
                       window that is not classified ground: roof, the
                       slab returns of ALL storeys (ground, intermediate,
                       mezzanine, partial), and anything else measured.
                       Raw XYZ, unclipped, not attributed to any storey and
                       not split by Z -- this is the cloud a true 3D
                       segmentation should consume, and it is a superset
                       of floor_points_xyz and roof_points_xyz (do not
                       add them to it). With no classification at all it
                       is every return (ground can't be told apart).
    metadata         : read-only mapping of diagnostics (see
                       calculate_z_bounds). NOT geometry: nothing in it may
                       be used to build a plane, cap, cutting level,
                       extrusion, floor height, or any elevation. It also
                       carries the semantic-contract keys
                       "structure_semantic_labels" (the authoritative
                       per-point "slab"/"roof"/"wall" source, from
                       explicitly configured classification codes only),
                       "orientation_up" and "roof_points_semantics" (each
                       None unless authoritative source semantics exist).

    There is deliberately no scalar-height, model, feature, or 2D-outline
    field on this type. The old 6-tuple contract
    (floor, z_roof, model, poly_features, roof_xy, roof_xyz) is gone; code
    that still unpacks six values fails loudly with ValueError instead of
    silently receiving a geometry-shaped scalar. Likewise, code written for
    the 3-field version (floor, roof, metadata) fails loudly on unpack:
    read `structure_points_xyz` explicitly.
    """
    floor_points_xyz: Optional[np.ndarray]
    roof_points_xyz: Optional[np.ndarray]
    structure_points_xyz: Optional[np.ndarray]
    metadata: "types.MappingProxyType"


def get_intersecting_lidar_tiles(poly, conn=None, include_floor_ids=False,
                                 search_geom=None):
    """
    Queries the PostGIS spatial index to find exactly which
    LiDAR tiles overlap with the AI-detected building footprint.

    `search_geom`: optional shapely geometry used for the tile lookup
    INSTEAD of `poly`. Roof acquisition passes the expanded LiDAR
    acquisition box here so a tile that holds only an overhang/eave
    outside the drafted footprint is still found. Defaults to `poly`
    (original behaviour) so existing callers are unaffected.

    IMPORTANT: `poly` must already be in real-world / global coordinates
    (the same CADASTRE_SRID as lidar_tile_index.tile_boundary), NOT raw
    image-pixel coordinates. Calling this with a pixel-space polygon will
    silently return zero tiles every time (pixel values like 0-1000
    interpreted as SRID coordinates will almost never intersect real
    tile boundaries), which is exactly why every roof used to come out
    flat regardless of what LiDAR was indexed. main.py now converts the
    footprint to global coordinates via the GNSS affine BEFORE calling
    calculate_z_bounds / get_floor_points_xyz, specifically to satisfy
    this requirement.

    `include_floor_ids`: when True, returns a list of
    (file_path, floor_id) tuples instead of bare paths, reading
    `TILE_FLOOR_COLUMN` (default "floor_index") from lidar_tile_index --
    the recorded storey each scan file covers. floor_id is None for any
    tile where the column is NULL, and for EVERY tile if the column
    doesn't exist in the schema at all (older indexes). None means
    "floor identity not recorded", which downstream is treated as
    unknown, never as "any floor". Default False keeps the original
    list-of-paths contract for existing callers.

    FIX: previously opened its own raw psycopg2.connect() per call,
    creating a fresh TCP connection to Postgres for a single spatial index
    query and then immediately closing it. Under batch registration
    (run_batch_cadastre_pipeline: N buildings × M footprints), that was
    N×M connection round-trips. Now uses config.get_pool() (or an
    externally provided connection) and returns the connection to the pool
    properly, even on error paths.
    """
    owns_conn = conn is None
    try:
        if owns_conn:
            pool = config.get_pool()
            conn = pool.getconn()

        cursor = conn.cursor()
        try:
            # Convert the shapely polygon into an OGC standard EWKT format,
            # tagged with the SAME SRID the tile index actually uses.
            lookup_geom = search_geom if search_geom is not None else poly
            wkt_geom = f"SRID={CADASTRE_SRID};{lookup_geom.wkt}"

            if include_floor_ids:
                # The floor column is optional (older indexes predate it),
                # so try it first and fall back to paths-only with an
                # explicit "identity not recorded" marker. The SAVEPOINT
                # keeps a failed statement from poisoning a caller-supplied
                # transaction that may still have work to do.
                cursor.execute("SAVEPOINT z_engine_floor_col;")
                try:
                    cursor.execute(f"""
                        SELECT file_path, "{TILE_FLOOR_COLUMN}"
                        FROM lidar_tile_index
                        WHERE ST_Intersects(tile_boundary, ST_GeomFromEWKT(%s));
                    """, (wkt_geom,))
                    rows = [(row[0], row[1]) for row in cursor.fetchall()]
                    cursor.execute("RELEASE SAVEPOINT z_engine_floor_col;")
                    return rows
                except Exception as col_err:
                    cursor.execute("ROLLBACK TO SAVEPOINT z_engine_floor_col;")
                    cursor.execute("RELEASE SAVEPOINT z_engine_floor_col;")
                    print(f"   ℹ️ lidar_tile_index has no usable "
                          f"'{TILE_FLOOR_COLUMN}' column ({col_err}); tile-level "
                          f"floor identity unavailable.")

            # Utilize the GiST spatial index to instantly find overlapping point cloud tiles
            cursor.execute("""
                SELECT file_path FROM lidar_tile_index
                WHERE ST_Intersects(tile_boundary, ST_GeomFromEWKT(%s));
            """, (wkt_geom,))

            tiles = [row[0] for row in cursor.fetchall()]
            if include_floor_ids:
                return [(path, None) for path in tiles]
            return tiles
        finally:
            # FIX: previously cursor.close() sat after fetchall() on the
            # success path only, so a query-time exception (caught below)
            # left the cursor open on a connection that then got handed
            # straight back to the pool.
            cursor.close()
    except Exception as e:
        print(f"❌ Spatial Index Query Failed: {e}")
        return []
    finally:
        if owns_conn and conn is not None:
            try:
                pool.putconn(conn)
            except Exception:
                try:
                    conn.close()
                except Exception:
                    pass


class LidarCRSError(RuntimeError):
    """
    A LAS/LAZ tile's source CRS is missing/invalid, or coordinates could not
    be transformed between it and CADASTRE_SRID. Deliberately NOT swallowed
    by the tile-read loop: skipping such a tile would look like "no LiDAR
    coverage" and silently hide a CRS problem. A CRS is never guessed.
    """


def _tile_source_horizontal_crs(header, laz_path):
    """
    The tile's own CRS from its LAS/LAZ header (`header.parse_crs()`),
    reduced to its horizontal component when it is a compound
    (horizontal + vertical) CRS. Raises LidarCRSError if the header carries
    no CRS or it cannot be parsed -- never guesses one.
    """
    try:
        src_crs = header.parse_crs()
    except Exception as e:
        raise LidarCRSError(
            f"{laz_path}: could not parse source CRS from LAS/LAZ header: {e}") from e
    if src_crs is None:
        raise LidarCRSError(
            f"{laz_path}: no CRS found in LAS/LAZ header; refusing to guess one")
    if src_crs.is_compound:
        src_crs = src_crs.sub_crs_list[0]
    return src_crs


def _read_tile_points(laz_path, minx, miny, maxx, maxy):
    """
    Reads one LAZ/LAS tile and crops it to the LiDAR XY acquisition box
    (minx, miny, maxx, maxy) -- a broad-phase fetch window, not a
    footprint clip and not a geometry boundary.

    COORDINATE SYSTEM: the box is given in CADASTRE_SRID, but the tile's
    raw X/Y are in the file's OWN CRS (`header.parse_crs()`, horizontal
    component only). The box is transformed INTO the tile CRS (densified
    edges, so it still covers the window under projection distortion) for a
    coarse raw-point prefilter; the surviving points' X/Y are then
    transformed into CADASTRE_SRID and cropped to the exact box. The
    returned x/y are therefore ALWAYS in CADASTRE_SRID (same CRS as the
    footprint and every downstream test -- CRSs are never mixed). Z is
    returned exactly as measured in the file: never transformed,
    normalised, estimated or derived. Classification is unchanged.
    Raises LidarCRSError if the tile has no valid CRS or a transform fails.

    Returns (x, y, z, classification, floor_ids) for the cropped points.
    `classification` is None when the file carries no classification
    dimension (not every LAS/LAZ file is ground-classified).
    `floor_ids` is the per-point recorded storey from
    FLOOR_ID_DIMENSION, or None when no such dimension is configured or
    the file doesn't carry it -- meaning this file records no per-point
    floor identity, NOT that its points belong to some default floor.
    """
    with laspy.open(laz_path) as fh:
        src_crs = _tile_source_horizontal_crs(fh.header, laz_path)
        dst_crs = CRS.from_epsg(int(CADASTRE_SRID))
        try:
            to_tile = Transformer.from_crs(dst_crs, src_crs, always_xy=True)
            to_cadastre = Transformer.from_crs(src_crs, dst_crs, always_xy=True)
            # Query window (CADASTRE_SRID) -> tile CRS. Edges are densified so
            # the axis-aligned result still encloses the window under
            # projection distortion / rotation. Coarse prefilter only.
            raw_bb = to_tile.transform_bounds(minx, miny, maxx, maxy, densify_pts=21)
        except Exception as e:
            raise LidarCRSError(
                f"{laz_path}: could not transform query window from "
                f"EPSG:{CADASTRE_SRID} to tile CRS ({src_crs.to_string()}): {e}") from e
        if not all(math.isfinite(v) for v in raw_bb):
            raise LidarCRSError(
                f"{laz_path}: query window transform from EPSG:{CADASTRE_SRID} to "
                f"tile CRS ({src_crs.to_string()}) produced non-finite bounds")

        las = fh.read()

        # Broad Phase Bounding Box Crop (Massively faster than checking every point!)
        # The box is the (generous) READ window, so measured geometry
        # outside the drafted footprint is kept. The final, data-driven
        # window is chosen afterwards in _collect_points_for_footprint.
        # This crop is a performance device, not a boundary.
        raw_x = np.asarray(las.x)
        raw_y = np.asarray(las.y)
        cand = np.nonzero(_bbox_mask(raw_x, raw_y, raw_bb))[0]

        # Only X/Y move into CADASTRE_SRID; Z is never touched.
        if cand.size:
            try:
                cx, cy = to_cadastre.transform(raw_x[cand], raw_y[cand], errcheck=True)
            except Exception as e:
                raise LidarCRSError(
                    f"{laz_path}: could not transform tile X/Y from "
                    f"({src_crs.to_string()}) to EPSG:{CADASTRE_SRID}: {e}") from e
            cx = np.asarray(cx, dtype=float)
            cy = np.asarray(cy, dtype=float)
        else:
            cx = cy = np.empty(0, dtype=float)

        # Exact crop in CADASTRE_SRID, the same box the caller asked for.
        exact = _bbox_mask(cx, cy, (minx, miny, maxx, maxy))
        mask = np.zeros(len(raw_x), dtype=bool)
        mask[cand[exact]] = True

        x = cx[exact]
        y = cy[exact]
        z = np.asarray(las.z[mask])

        try:
            classification = np.asarray(las.classification[mask])
        except AttributeError:
            classification = None

        floor_ids = None
        if FLOOR_ID_DIMENSION is not None:
            try:
                floor_ids = np.asarray(las[FLOOR_ID_DIMENSION][mask])
            except Exception:
                # Dimension configured but absent from this file. Stay
                # None: an absent label is unknown, not floor 0.
                floor_ids = None

    return x, y, z, classification, floor_ids


def _collect_points_bounded(poly, lidar_path=None, conn=None,
                            declared_floor_id=None,
                            acquisition_margin_m=None,
                            acquisition_max_expansions=None,
                            acq_info=None, _report_truncation=True):
    """
    Single finite-valve pass of _collect_points_for_footprint (see it and
    the ACQUISITION BROAD-PHASE block). Shared tile-lookup + point extraction, used by BOTH the roof fit
    (calculate_z_bounds) and the floor evidence extraction
    (get_floor_points_xyz), so roof and floor are always drawn from
    exactly the same point cloud for a given footprint.

    The footprint only SEEDS a LiDAR XY acquisition window, which is
    then grown from the data (see ACQUISITION BROAD-PHASE near the top of
    this module): `acquisition_margin_m` (default
    ROOF_ACQUISITION_MARGIN_M) is the grow step, and
    `acquisition_max_expansions` (default ACQUISITION_MAX_EXPANSIONS) a
    reported safety valve -- neither is a maximum building extent. The
    returned cloud is everything measured in the final window, including
    geometry outside the drafted footprint. Nothing here clips to the
    footprint polygon. Only get_terrain_context_points (context, not
    geometry) applies a polygon test afterwards.

    `acq_info`: optional dict, filled with "window" (final bbox),
    "expansions" (steps taken) and "truncated" (safety valve tripped
    with evidence still present) for the caller's metadata.

    Returns (pts_x, pts_y, pts_z, pts_cls, pts_floor, floor_id_source,
    laz_paths).

    pts_cls is None if any contributing tile lacked classification, since
    a partial classification array can't be trusted to line up with
    pts_x/y/z.

    pts_floor is the per-point RECORDED storey label, aligned with
    pts_x/y/z, assembled from (in priority order):
      - FLOOR_ID_DIMENSION, the per-point label in the file itself;
      - the tile's recorded TILE_FLOOR_COLUMN value, inherited by every
        point read from that file;
      - `declared_floor_id`, when the caller passed an explicit
        per-floor scan path and is asserting what it contains.
    It is None when NO contributing tile supplied floor identity by any
    of those routes, and any point whose identity is unknown is left as
    None inside the array rather than defaulted. floor_id_source is a
    short string naming which route was used, for logging and for
    downstream provenance.

    All of pts_x/pts_y/pts_z/pts_cls/pts_floor are None (and laz_paths is
    []) when there is no LiDAR coverage at all for this footprint.
    """
    tile_floor_ids = {}
    step_m = ROOF_ACQUISITION_MARGIN_M if acquisition_margin_m is None else float(acquisition_margin_m)
    if step_m <= 0:
        raise ValueError(
            f"acquisition margin (grow step) must be > 0, got {step_m}. "
            f"A zero step would let the footprint bounds define extent.")
    max_steps = (ACQUISITION_MAX_EXPANSIONS if acquisition_max_expansions is None
                 else int(acquisition_max_expansions))
    if max_steps < 1:
        raise ValueError(
            f"acquisition_max_expansions must be >= 1, got {max_steps}.")
    # Read window: footprint + (max_steps + 1) steps, so the ring test at
    # the safety valve can see whether evidence continues past it. It is
    # also the geometry handed to the PostGIS ST_Intersects tile lookup
    # (finite by necessity: an unbounded lookup would open every tile).
    acq_bbox = _acquisition_bbox(poly, step_m * (max_steps + 1))
    tile_lookup_geom = box(*acq_bbox)

    if lidar_path:
        if not os.path.exists(lidar_path):
            print(f"   ⚠️ Explicit LiDAR path '{lidar_path}' not found.")
            return None, None, None, None, None, None, []
        laz_paths = [lidar_path]
        if declared_floor_id is not None:
            tile_floor_ids[lidar_path] = declared_floor_id
    else:
        tiles = get_intersecting_lidar_tiles(
            poly, conn=conn, include_floor_ids=True,
            search_geom=tile_lookup_geom,
        )
        if not tiles:
            print("   ⚠️ No indexed LiDAR tiles intersect with this footprint's "
                  "acquisition window.")
            return None, None, None, None, None, None, []
        laz_paths = [path for path, _ in tiles]
        tile_floor_ids = {path: fid for path, fid in tiles if fid is not None}

    all_x, all_y, all_z, all_cls, all_floor = [], [], [], [], []
    have_cls = True
    any_floor_id = False
    floor_id_source = None
    minx, miny, maxx, maxy = acq_bbox

    for laz_path in laz_paths:
        try:
            x, y, z, cls, floor_ids = _read_tile_points(laz_path, minx, miny, maxx, maxy)
        except LidarCRSError:
            raise  # a CRS problem must abort, not look like "no coverage"
        except Exception as e:
            print(f"   ⚠️ Could not read tile {laz_path}: {e}")
            continue

        all_x.extend(x)
        all_y.extend(y)
        all_z.extend(z)

        if cls is None:
            have_cls = False
        else:
            all_cls.extend(cls)

        # Per-point labels beat the tile-level tag: a single file can
        # legitimately span storeys, and the point label knows that while
        # the file-level tag doesn't.
        if floor_ids is not None and len(floor_ids) == len(x):
            all_floor.extend(floor_ids.tolist())
            any_floor_id = True
            floor_id_source = f"per-point LAS dimension '{FLOOR_ID_DIMENSION}'"
        elif laz_path in tile_floor_ids:
            all_floor.extend([tile_floor_ids[laz_path]] * len(x))
            any_floor_id = True
            if floor_id_source is None:
                floor_id_source = (
                    f"declared per-floor scan path"
                    if (lidar_path and declared_floor_id is not None)
                    else f"lidar_tile_index.{TILE_FLOOR_COLUMN}"
                )
        else:
            # Unknown identity for this tile's points. Explicitly None --
            # never backfilled with a default storey.
            all_floor.extend([None] * len(x))

    if not all_x:
        return None, None, None, None, None, None, laz_paths

    pts_x = np.array(all_x)
    pts_y = np.array(all_y)
    pts_z = np.array(all_z)
    pts_cls = np.array(all_cls) if (have_cls and len(all_cls) == len(all_x)) else None
    pts_floor = np.array(all_floor, dtype=object) if any_floor_id else None

    # Narrow from the (generous) read window to the data-driven window:
    # keep everything out to the first empty ring, drop the rest. Pure
    # XY selection of measured points; no point is moved, averaged, or
    # given a new Z.
    window, n_steps, truncated = _grow_window_to_evidence(
        pts_x, pts_y, pts_cls, poly, step_m, max_steps)
    keep = _bbox_mask(pts_x, pts_y, window)
    if not np.all(keep):
        pts_x, pts_y, pts_z = pts_x[keep], pts_y[keep], pts_z[keep]
        if pts_cls is not None:
            pts_cls = pts_cls[keep]
        if pts_floor is not None:
            pts_floor = pts_floor[keep]
    if acq_info is not None:
        acq_info.update({"window": tuple(float(v) for v in window),
                         "expansions": n_steps, "truncated": truncated})
    if truncated and _report_truncation:
        if pts_cls is None:
            print(f"   ℹ️ No classification: ground can't be told from building, "
                  f"so the acquisition window grew to the safety valve "
                  f"({n_steps} x {step_m:g} m). Not a claim about building extent.")
        else:
            print(f"   ⚠️ Acquisition window hit its safety valve ({n_steps} x "
                  f"{step_m:g} m) with measured evidence still continuing just "
                  f"outside it. The extent was NOT found and the window edge is "
                  f"an acquisition limit, not geometry; raise "
                  f"acquisition_max_expansions / ACQUISITION_MAX_EXPANSIONS "
                  f"or pass math.inf for unbounded acquisition.")
    if not len(pts_x):
        return None, None, None, None, None, None, laz_paths

    return pts_x, pts_y, pts_z, pts_cls, pts_floor, floor_id_source, laz_paths


def _collect_points_for_footprint(poly, lidar_path=None, conn=None,
                                  declared_floor_id=None,
                                  acquisition_margin_m=None,
                                  acquisition_max_expansions=None,
                                  acq_info=None):
    """
    Public collector; same contract and return value as
    _collect_points_bounded. `acquisition_max_expansions` may be an int
    >= 1 (finite safety valve, default ACQUISITION_MAX_EXPANSIONS) or
    math.inf (unbounded: the valve is doubled and the collection redone
    until the data-driven window stops at a genuinely empty ring, so the
    valve can never become the building's extent).
    """
    limit = (ACQUISITION_MAX_EXPANSIONS if acquisition_max_expansions is None
             else acquisition_max_expansions)
    kwargs = dict(lidar_path=lidar_path, conn=conn,
                  declared_floor_id=declared_floor_id,
                  acquisition_margin_m=acquisition_margin_m)
    if limit != math.inf:
        return _collect_points_bounded(
            poly, acquisition_max_expansions=limit, acq_info=acq_info, **kwargs)

    valve = 8
    while True:
        info = {}
        result = _collect_points_bounded(
            poly, acquisition_max_expansions=valve, acq_info=info,
            _report_truncation=False, **kwargs)
        if not info.get("truncated"):
            break
        valve *= 2
    if acq_info is not None:
        acq_info.update(info)
    return result


def points_in_polygon_mask(pts_x, pts_y, poly):
    """
    Boolean mask of which (x, y) points lie inside `poly` ITSELF -- not
    inside its bounding box.

    This matters because the only spatial filter applied upstream is a
    bounding-box crop (_read_tile_points, "Broad Phase Bounding Box
    Crop"). It is used for terrain CONTEXT selection and for
    inside/outside DIAGNOSTICS on slab evidence -- never to delete
    measured roof or slab returns, which must not be trimmed by the
    drafted footprint. For any footprint that isn't a perfect axis-aligned
    rectangle -- an L-shape, a diagonal wing, a curved frontage -- the
    bbox sweeps in points that belong to the neighbouring building, the
    street, or the courtyard next door. Binning those as if they were
    this unit's floor produces a slab surface partly measured off
    someone else's structure.

    Interior rings (courtyards, atria, light wells -- see ai_to_ogc.py's
    note on holes) are honoured: a point inside a hole is NOT inside the
    polygon, and is correctly excluded. There is no slab over a light
    well, so its returns must not be binned into one.

    Boundary points count as inside: a slab return landing exactly on
    the wall line is real evidence for this unit's floor, and at
    floating-point precision the distinction is immaterial anyway.
    """
    pts_x = np.asarray(pts_x)
    pts_y = np.asarray(pts_y)

    if len(pts_x) == 0:
        return np.zeros(0, dtype=bool)

    # Fast path: shapely >= 2.0 vectorised predicate over the whole array.
    # intersects_xy (rather than contains_xy) keeps boundary points.
    try:
        import shapely
        return np.asarray(shapely.intersects_xy(poly, pts_x, pts_y), dtype=bool)
    except (ImportError, AttributeError):
        pass

    # Fallback for shapely 1.x: prepared geometry keeps the per-point
    # test cheap by building the edge index once instead of per call.
    prepared = prep(poly)
    return np.fromiter(
        (prepared.intersects(Point(px, py)) for px, py in zip(pts_x, pts_y)),
        dtype=bool,
        count=len(pts_x),
    )


# ---------------------------------------------------------------------
# REMOVED: _grid_bin(pts_x, pts_y, pts_z, poly, grid_cell_size,
# min_points_per_cell, cell_z_fn). Its only caller was
# _slab_floor_points()'s optional grid_cell_size path, which used it with
# cell_z_fn=np.median to replace each grid_cell_size x grid_cell_size
# cell's measured points with ONE (x, y, median-z) centroid. That is a
# 2.5D reduction of exactly the kind this module exists to keep out of
# floor geometry: a median-per-cell height collapses whatever real
# variation the slab actually has -- a ramp, a tilted deck, a slab with a
# genuine local step or drain fall, a partial slab's ragged measured
# edge -- into one flattened, XY-gridded height field, and does it before
# main.py's true 3D segmentation/topology pipeline ever sees the points.
# `min_points_per_cell` remains as a parameter name on the functions
# below, but it is now a flat "at least this many measured points, full
# stop" threshold with no cell/grid concept behind it at all -- see
# _slab_floor_points. Floor evidence is now ALWAYS the raw measured (x,
# y, z) points; there is no parameter anywhere in this module that can
# ask for it to be binned, gridded, or Z-collapsed.
# ---------------------------------------------------------------------


def get_terrain_context_points(poly, lidar_path=None, conn=None):
    """
    CONTEXT ONLY -- NOT FLOOR GEOMETRY. Never feed the result of this
    function into floor_points_xyz, SectionProfile, or anything that
    extrudes a solid.

    REMOVED: `_terrain_floor_points()`. That function used terrain as
    the ground floor's slab evidence (classified class-2 returns when
    available, and a "lowest return per grid cell" heuristic when not).
    Both were substitutes for a measurement nobody took:

      - Classified terrain is the ground OUTSIDE/BENEATH the building.
        It sits below foundation, footings, fill and screed, and it
        follows site grading, not finished floor level. It is not the
        slab, and the offset between the two is a construction
        parameter this module has no measurement of.
      - The low-percentile grid fallback was worse: "the lowest returns
        in this cell are roughly the ground" is a heuristic about open
        terrain. Applied under a building footprint it happily returns
        whatever the scanner happened to see lowest -- a shadowed patch
        of street, a step, a vehicle roof -- and stamps it on the
        cadastral record as a measured floor.

    Ground floors now go through exactly the same path as upper floors:
    explicit, measured slab XYZ evidence (SLAB_CLASSIFICATION_CODE) or
    None. A ground floor is not epistemically special -- it is just the
    floor an aerial sensor is most tempted to guess at.

    This helper remains because terrain is still genuinely useful as
    CONTEXT: comparing a measured slab against surrounding grade,
    QA-ing a suspicious slab elevation, reporting site levels. Those
    are all "here is the ground, separately", never "here is the
    floor".

    Returns an (N, 3) np.ndarray of classified terrain points clipped to
    `poly`, or None. There is no heuristic fallback here either -- if
    the cloud has no ground classification, this returns None rather
    than inventing terrain.
    """
    pts_x, pts_y, pts_z, pts_cls, _, _, _ = _collect_points_for_footprint(
        poly, lidar_path=lidar_path, conn=conn
    )

    if pts_x is None or pts_cls is None:
        return None

    mask = (pts_cls == GROUND_CLASSIFICATION_CODE) & points_in_polygon_mask(
        pts_x, pts_y, poly
    )

    if not np.any(mask):
        return None

    return np.column_stack((pts_x[mask], pts_y[mask], pts_z[mask]))


def _slab_floor_points(pts_x, pts_y, pts_z, pts_cls, pts_floor, poly,
                       floor_index, floor_id_source=None,
                       min_points_per_cell=3):
    """
    Floor evidence for ONE specific storey -- ground (floor_index == 0)
    and upper (floor_index > 0) alike -- built ONLY from points that are
    BOTH:

      (a) explicitly classified as a MEASURED slab return
          (SLAB_CLASSIFICATION_CODE), and
      (b) explicitly RECORDED as belonging to THIS floor_index (see the
          FLOOR IDENTITY block at the top of this module).

    This is the single floor-geometry path in this module. This function
    does NOT:

      - use floor height, cumulative height, expected_z, or any other
        elevation hint to decide where the slab is, OR to decide which
        storey a slab belongs to. Splitting one cloud's slab returns
        into storeys by Z range, by storey height, or by finding peaks
        in an elevation histogram would derive floor identity from the
        very elevation the floor is meant to supply. There is no
        parameter here that could express it.
      - return every slab return in the footprint when floor identity is
        unknown. That was the previous behaviour and it silently handed
        floor 1 the slab points of floors 2, 3 and 4, producing a single
        smeared "floor" spanning the whole building.
      - accept balcony, ledge, parapet, or any other generic building
        (ASPRS class 6) return as a slab stand-in.
      - accept ground/terrain-classified returns, for ANY floor,
        including the ground floor (see get_terrain_context_points).
      - CLIP to the blueprint. The footprint polygon never deletes a
        measured slab return: a cantilever, overhang, balcony, setback or
        irregular extent whose XY lies outside the drafted outline stays
        in the evidence exactly as measured. The blueprint may only
        attribute/validate -- the number of retained points inside vs
        outside it is logged so a misregistered footprint or a
        neighbour's slab is visible -- and never trims or defines
        geometry. (Trade-off: a neighbouring unit's slab that lies in the
        acquisition window and carries this floor's recorded id is also
        kept; recorded floor identity and the 3D pipeline downstream, not
        an XY rectangle, are what separate those.)
      - bin, grid, or Z-collapse the result in ANY way (3D-FIX: this
        function used to offer an optional `grid_cell_size` path that
        replaced each grid cell's measured points with one
        (x, y, median-z) centroid via the now-removed `_grid_bin()`.
        That parameter is gone -- see the REMOVED note above this
        function -- so there is no way to invoke it, accidentally or
        otherwise. Every measured point that passes the classification/
        identity/footprint tests below is returned AS MEASURED: a ramp,
        a tilted deck, a slab with a real local step, or a partial
        slab's ragged edge comes through with its own real variation
        intact, not flattened onto an XY grid's per-cell median.

    If there is no SLAB_CLASSIFICATION_CODE configured, or the point
    cloud carries no classification at all, or no floor identity is
    recorded, or no points carry BOTH this floor's id and the slab code,
    this returns None -- no evidence means no geometry, never an invented
    one and never another storey's slab borrowed in its place.

    `pts_floor`: per-point recorded storey labels aligned with
    pts_x/y/z, or None when nothing recorded floor identity. None is
    fatal for floor selection by design: a slab of unknown storey is not
    this floor's slab.

    `min_points_per_cell`: despite the name (kept for signature
    stability -- see the REMOVED note above), this is now a flat
    "at least this many measured points for this floor, full stop"
    threshold. There are no cells any more for it to be "per".

    Returns an (N, 3) np.ndarray of measured slab points for this floor,
    or None.
    """
    if SLAB_CLASSIFICATION_CODE is None:
        print("   ⚠️ No SLAB_CLASSIFICATION_CODE configured -- there is no "
              "classification code reserved for measured slab returns, so "
              "slab evidence can't be identified without guessing. "
              "floor_points_xyz = None.")
        return None

    if pts_cls is None:
        print("   ⚠️ Point cloud carries no classification data for this "
              "footprint -- can't isolate measured slab returns from "
              "balconies/ledges/parapets/terrain. floor_points_xyz = None.")
        return None

    if pts_floor is None:
        print(f"   ⚠️ No floor identity recorded for this footprint's points "
              f"(no per-point '{FLOOR_ID_DIMENSION}' dimension, no "
              f"lidar_tile_index.{TILE_FLOOR_COLUMN} value, no declared "
              f"per-floor scan path). Slab returns can't be attributed to "
              f"floor {floor_index} without assuming an elevation, so "
              f"floor_points_xyz = None rather than returning every storey's "
              f"slabs at once.")
        return None

    try:
        floor_mask = _floor_id_match_mask(pts_floor, floor_index)
    except KeyError as e:
        print(f"   ⚠️ {e} floor_points_xyz = None.")
        return None

    # Slab class AND recorded as this storey. NO footprint test here: the
    # blueprint must not delete measured XYZ (see the docstring).
    slab_mask = (pts_cls == SLAB_CLASSIFICATION_CODE) & floor_mask
    slab_count = int(np.count_nonzero(slab_mask))

    if slab_count < min_points_per_cell:
        other_floors = sorted({
            str(f) for f in np.asarray(pts_floor)[pts_cls == SLAB_CLASSIFICATION_CODE]
            if f is not None
        })
        print(f"   ⚠️ Only {slab_count} point(s) classified as measured slab "
              f"(class {SLAB_CLASSIFICATION_CODE}) AND recorded as floor "
              f"{floor_index} in the acquisition window -- not enough to "
              f"constitute real slab evidence. floor_points_xyz = None. "
              f"(Slab returns present for recorded floors: "
              f"{other_floors or 'none'} -- these are NOT substituted in.)")
        return None

    slab_x, slab_y, slab_z = pts_x[slab_mask], pts_y[slab_mask], pts_z[slab_mask]

    # Validation only (never filtering): how much of this measured slab
    # lies inside the drafted footprint vs outside it.
    n_inside = int(np.count_nonzero(points_in_polygon_mask(slab_x, slab_y, poly)))
    n_outside = len(slab_x) - n_inside
    provenance = f" via {floor_id_source}" if floor_id_source else ""

    floor_points_xyz = np.column_stack((slab_x, slab_y, slab_z))
    print(f"   🧱 Floor {floor_index} slab evidence: {len(floor_points_xyz)} measured "
          f"slab-classified points (LAS class {SLAB_CLASSIFICATION_CODE})"
          f"{provenance}, RAW (no binning or Z-collapsing, NOT clipped to the "
          f"blueprint: {n_inside} inside the footprint, {n_outside} outside "
          f"retained as measured).")
    return floor_points_xyz


def get_floor_points_xyz(poly, floor_index=0, lidar_path=None, conn=None,
                          floor_scan_path=None, min_points_per_cell=3):
    """
    Standalone entry point: extracts real slab evidence for ONE SPECIFIC
    storey of a footprint as an array of (x, y, z) points -- never a
    flattened scalar, never terrain, never balcony/ledge/parapet returns
    masquerading as a slab, and never another storey's slab.

    `poly` must be in real-world / global coordinates (same requirement
    as get_intersecting_lidar_tiles / calculate_z_bounds).

    `floor_index`: the storey to extract, ground (0) and upper (>0)
        treated identically. Points are selected by RECORDED floor
        identity matched against this value (see the FLOOR IDENTITY
        block at the top of this module, and config.FLOOR_ID_MAP when
        the data's labels aren't plain storey integers). There is
        deliberately no floor-height, cumulative-height, or
        expected-elevation parameter, and no code path that could infer
        a storey from a Z value.

    `floor_scan_path`: optional. An explicit path to a scan the caller
        asserts contains floor `floor_index`. Every slab return in that
        file is attributed to this floor on the caller's authority. Use
        this when a capture produced one file per storey but the index
        doesn't record it. This is provenance, supplied by whoever knows
        what was scanned -- not an inference made here.

    `lidar_path`: optional. A single file to read instead of querying
        the tile index. NOTE: unlike `floor_scan_path`, this does NOT
        assert floor membership -- points from it still need recorded
        floor identity (a per-point FLOOR_ID_DIMENSION) to be selected.
        A general aerial tile passed here will correctly yield None, not
        a floor's worth of borrowed slabs.

    `min_points_per_cell`: a flat "at least this many measured points"
        threshold (kept name for signature stability). See the
        3D-FIX (this pass): the removed `grid_cell_size` parameter used
        to route this function's result through `_grid_bin()`, replacing
        each grid cell's measured points with one (x, y, median-z)
        centroid. That parameter is GONE -- passing it now raises
        TypeError, deliberately, matching this module's established
        convention for a removed geometry-affecting parameter (see
        `floor_ground_percentile` above). There is no way left to invoke
        median-Z binning from this function; the result is always the
        raw measured points that passed classification/identity/
        footprint selection.

    FIX (per-floor selection): this function previously returned ALL
    slab-classified points in the footprint regardless of `floor_index`,
    which the docstring acknowledged as a caveat. In a multi-storey scan
    that meant floor 1 received floors 1-N's slab returns fused into one
    surface -- a floor spanning the whole building. Selection is now by
    recorded floor identity, and where identity is absent the answer is
    None. Do NOT "restore" the old behaviour by range-slicing Z or
    clustering elevations into storeys; that invents the identity it
    claims to read.

    Returns
    -------
    floor_points_xyz : np.ndarray of shape (N, 3), or None
        None when there is no usable measured-slab evidence recorded for
        THIS storey. That is the expected, correct result whenever the
        point cloud has no dedicated slab classification, or has slab
        returns but no recorded floor identity to attribute them by.
        Feed the array directly into SectionProfile -- do not reduce it
        to a scalar before doing so. Handle None as "no slab geometry
        for this floor", never as a cue to substitute terrain, grade,
        zero, or a neighbouring storey's slab.
    """
    if floor_scan_path:
        try:
            declared_floor_id = _resolve_floor_id(floor_index)
        except KeyError as e:
            print(f"   ⚠️ {e} floor_points_xyz = None.")
            return None
    else:
        declared_floor_id = None

    pts_x, pts_y, pts_z, pts_cls, pts_floor, floor_id_source, _ = (
        _collect_points_for_footprint(
            poly,
            lidar_path=floor_scan_path or lidar_path,
            conn=conn,
            declared_floor_id=declared_floor_id,
        )
    )

    if pts_x is None:
        print("   ⚠️ No LiDAR coverage for this footprint. floor_points_xyz = None.")
        return None

    return _slab_floor_points(
        pts_x, pts_y, pts_z, pts_cls, pts_floor, poly,
        floor_index=floor_index,
        floor_id_source=floor_id_source,
        min_points_per_cell=min_points_per_cell,
    )


def calculate_z_bounds(poly, lidar_path=None, conn=None,
                        floor_index=0, floor_scan_path=None,
                        floor_min_points_per_cell=3,
                        acquisition_margin_m=None,
                        acquisition_max_expansions=None):
    """
    Dynamically fetches LAZ files based on the spatial footprint and
    extracts real MEASURED evidence for both the roof and one floor/slab
    -- as points, never a fitted surface and never a scalar.

    `poly` must be in real-world / global coordinates (see note above).
    It seeds the LiDAR XY acquisition box only; it never defines, trims
    or discards measured roof geometry.

    `acquisition_margin_m`: optional override of the grow STEP (default
    config.ROOF_ACQUISITION_MARGIN_M, 3.0 m). The acquisition window
    widens by this much per step for as long as measured evidence
    continues just outside it. It is a performance/granularity knob, not
    a maximum building extent.

    `acquisition_max_expansions`: optional override of the growth safety
    valve (default config.ACQUISITION_MAX_EXPANSIONS, 8; int >= 1, or
    math.inf for unbounded acquisition with no valve). If a finite valve
    trips while evidence is still continuing,
    metadata["acquisition_truncated"] is True,
    metadata["acquisition_extent_complete"] is False, and a warning is
    printed; the window edge is then an acquisition limit, never geometry.

    `lidar_path`: optional. If provided (e.g. run_drone.py passing an
    explicit .laz path), that single file is used directly instead of
    querying the PostGIS spatial index.

    `conn`: optional. A pooled connection to reuse for the spatial index
    query. If None, a connection is borrowed from config.get_pool() for
    the duration of the tile lookup.

    `floor_index`: the storey whose slab to extract -- see
    get_floor_points_xyz. EVERY floor, ground (0) and upper (>0) alike,
    uses ONLY points explicitly classified as a measured slab return
    (SLAB_CLASSIFICATION_CODE) AND explicitly recorded as belonging to
    that storey. There is no terrain path, no floor-height /
    cumulative-height / expected-elevation parameter, and no way to
    separate storeys by Z. Call this once PER FLOOR/UNIT to get that
    floor's own evidence; floors are never assumed to share a Z, derived
    from one another, or handed each other's slab returns.

    `floor_scan_path`: optional explicit per-floor scan for
    `floor_index` -- see get_floor_points_xyz. The ROOF evidence
    deliberately keeps using the footprint's full indexed cloud (or
    `lidar_path`), not this interior scan; likewise structure_points_xyz
    comes from that indexed cloud only, so this scan's slab points are
    reported solely through floor_points_xyz.

    Returns
    -------
    ZEvidence(floor_points_xyz, roof_points_xyz, structure_points_xyz,
              metadata)

        floor_points_xyz : np.ndarray (N, 3) of real, measured slab
                            points for the storey `floor_index`, or None if
                            there's no usable evidence. Never a single
                            flattened number, never terrain (for ANY
                            floor, ground included), never a
                            balcony/ledge/parapet reused as a slab, never
                            computed from height, and never median-binned
                            onto a grid -- see _slab_floor_points. None is
                            the expected, correct result unless the point
                            cloud carries BOTH a dedicated slab
                            classification AND recorded floor identity for
                            that storey.

        roof_points_xyz  : np.ndarray (M, 3), or None -- CANDIDATE /
                            exposed-surface evidence, NOT automatically
                            roof: real MEASURED non-ground, non-slab
                            returns (facade, balcony, ledge or parapet
                            returns are not separated out) unless
                            metadata["roof_points_semantics"] == "roof",
                            set only when the array was built from points
                            carrying an explicitly configured
                            ROOF_CLASSIFICATION_CODE (then it contains ONLY
                            those points),
                            from the acquisition window
                            (a data-grown XY window seeded by the
                            footprint, NOT clipped to the footprint
                            polygon), with classified
                            ground/slab returns excluded where
                            classification exists. Overhangs, cantilevers,
                            eaves, undercuts, folds, vaults and any other
                            multi-valued-over-(x,y) shape are exactly as
                            present as in the raw scan, because nothing
                            here evaluates, fits, resamples or projects
                            them. Feed it to a true 3D segmentation /
                            topology pipeline; do not average, bin, or fit
                            a surface to it in this module. Slab-classified
                            returns are excluded here on purpose (they are
                            not roof) and are carried by
                            structure_points_xyz instead.

        structure_points_xyz : np.ndarray (K, 3), or None -- the FULL
                            measured structural cloud: every return in the
                            acquisition window not classified ground. That
                            is the candidate/exposed-surface returns (see
                            roof_points_xyz) PLUS the slab-classified
                            returns of EVERY storey, so intermediate slabs, mezzanines
                            and partial slabs reach a downstream 3D
                            segmentation even though only one storey is
                            attributable via `floor_index`. Nothing here is
                            attributed to a floor, split or filtered by Z
                            or floor height, clipped to the footprint,
                            binned, fitted or projected; it is the raw
                            measured XYZ. It is a SUPERSET of
                            floor_points_xyz and roof_points_xyz, so
                            consume it INSTEAD of concatenating them
                            (that would double-count). With no
                            classification it is every measured return.
                            The acquisition is only a broad-phase read, so
                            this cloud may include NEIGHBOURING structures
                            and is deliberately not clipped to the
                            blueprint; downstream 3D segmentation must
                            separate structures. When
                            metadata["acquisition_truncated"] is True the
                            cloud is INCOMPLETE and non-authoritative for
                            final geometry (metadata
                            "structure_cloud_incomplete" is True): the
                            window edge is an acquisition limit, never a
                            building boundary.
                            Points from an explicit `floor_scan_path` are
                            a separate acquisition and are NOT merged in;
                            they appear only in floor_points_xyz.

        metadata         : read-only mapping, DIAGNOSTICS ONLY:
                            "z_roof"        float or None -- a measured
                                            95th-percentile of Z, for
                                            logging / audit cross-checks /
                                            rough scale sanity. Never
                                            geometry (see below).
                            "z_roof_basis"  "roof_evidence" |
                                            "whole_crop" | None -- which
                                            points z_roof summarises.
                            "point_count", "roof_point_count",
                            "floor_point_count", "structure_point_count",
                            "slab_point_count" (all slab-classified
                            returns, any storey -- a count only),
                            "laz_paths",
                            "acquisition_window" (final XY bbox),
                            "acquisition_expansions",
                            "acquisition_truncated" (safety valve tripped
                            with evidence still continuing),
                            "acquisition_extent_complete" (TRI-STATE:
                            False when acquisition was explicitly
                            truncated -- the cloud ends at an acquisition
                            limit, not at the measured building edge;
                            True only when the collector explicitly
                            recorded a non-truncated, data-driven stop;
                            None when acquisition status is unavailable /
                            no LiDAR -- never inferred from a missing key;
                            "acquisition_truncated" and the *_incomplete
                            flags below follow the same True/False/None
                            rule),
                            "structure_cloud_incomplete" (True exactly when
                            acquisition_truncated: the structural cloud is
                            incomplete/non-authoritative for final
                            geometry; None when there was no LiDAR),
                            "roof_evidence_incomplete" /
                            "floor_evidence_incomplete" (True when the
                            array exists but comes from a truncated
                            acquisition: the points are real and kept for
                            diagnostics, but "measured points exist" is NOT
                            "measured extent is complete", so they must not
                            be consumed as complete geometry evidence;
                            False when acquisition was not truncated; None
                            when the array is None, or -- floor only --
                            comes from an explicit `floor_scan_path`, a
                            separate acquisition whose completeness this
                            function does not establish). No point is ever
                            cropped, extrapolated or replaced, and the
                            acquisition window edge is never geometry.
                            SEMANTIC CONTRACT (each None unless the source
                            data already states it; never inferred from Z,
                            height, normals, orientation, proximity, XY
                            overlap or position):
                            "structure_semantic_labels"  object array
                                            aligned 1:1 with
                                            structure_points_xyz: THE
                                            authoritative per-point
                                            semantic source for main.py.
                                            "slab" only from
                                            SLAB_CLASSIFICATION_CODE,
                                            "roof" only from a configured
                                            ROOF_CLASSIFICATION_CODE,
                                            "wall" only from a configured
                                            WALL_CLASSIFICATION_CODE, None
                                            (unlabelled) otherwise. Unset
                                            codes never produce a label
                                            (LAS class 6 "Building" is not
                                            roof or wall by itself), and
                                            the whole value is None when no
                                            point is labelled. It does not
                                            change roof_points_xyz, which
                                            stays candidate/exposed-surface
                                            evidence.
                            "orientation_up"             authoritative UP
                                            direction from the source, or
                                            None (the LAZ classification
                                            data carries none).
                            "roof_points_semantics"      "roof" only when
                                            ROOF_CLASSIFICATION_CODE is
                                            explicitly configured and
                                            roof-classified measured points
                                            exist, in which case
                                            roof_points_xyz holds ONLY
                                            those points; else None.

    When no LiDAR data is available at all, returns
    ZEvidence(None, None, None, metadata) with metadata["z_roof"] = None (there
    is no measurement, so there is no number -- not 0.0).

    z_roof may appear ONLY as metadata. Nothing in this module, and
    nothing downstream is entitled to, builds geometry from it: no flat
    roof plane, no cap or cutting level, no "z_roof - some_height" floor,
    no storey/Z-range pairing, no extrusion, no synthetic elevation of any
    kind, and no 2D-footprint-derived reconstruction.

    BREAKING CHANGE (this pass): the geometry-facing legacy contract is
    REMOVED. The old return was the 6-tuple
    (floor_points_xyz, z_roof, model, poly_features, roof_points_xy,
    roof_points_xyz). `model` and `poly_features` were permanently None
    (the degree-3 Ridge Z = f(X, Y) fit is long gone -- a height field
    cannot represent an overhang, an eave that curls back under itself, a
    vault, a dome, or anything multi-valued over (x, y)), and
    `roof_points_xy` was a disabled 2D projection. All three slots, and z_roof as a
    positional scalar, are dropped from the return type. Migrate callers:

        (floor_points_xyz, roof_points_xyz,
         structure_points_xyz, metadata) = calculate_z_bounds(poly)
        z_roof_diag = metadata["z_roof"]  # optional; logging only

    A leftover six-way unpack raises ValueError, deliberately.

    BREAKING CHANGE (this pass): `floor_grid_cell_size` is REMOVED from
    this signature. It used to route floor evidence through
    `_grid_bin()`'s median-per-cell binning (see the REMOVED note above
    that function). Passing it now raises TypeError, deliberately, the
    same convention this module already uses for `floor_ground_percentile`
    below. Floor evidence is always the raw measured points.

    BREAKING CHANGE (earlier): slab returns are selected by RECORDED
    floor identity, so `floor_index` actually selects a storey instead
    of being bookkeeping. A footprint whose points carry no recorded
    floor identity returns None for floor evidence. Supply identity via
    config.FLOOR_ID_DIMENSION, lidar_tile_index.{TILE_FLOOR_COLUMN}, or
    an explicit `floor_scan_path` -- never by slicing Z.

    BREAKING CHANGE (earlier): the ground floor no longer falls back
    to terrain. `_terrain_floor_points()` is removed as floor geometry --
    both its classified-ground branch and its lowest-return-per-cell
    heuristic. `floor_ground_percentile` is gone from the signature
    (passing it now raises TypeError, deliberately), and
    `floor_grid_cell_size` -- itself now also removed, see above --
    defaulted to None. Ground floors that used to come back with
    terrain-shaped "evidence" now come back None unless the cloud
    carries SLAB_CLASSIFICATION_CODE returns. That is the fix, not a
    regression: those values were the ground under the building, not its
    slab. Terrain remains available, separately and explicitly, via
    get_terrain_context_points() -- for context only.

    BREAKING CHANGE (earlier): this function used to return a scalar
    `z_base` (5th percentile of the whole point cloud) as element 0 of
    the tuple, and in a prior revision accepted an `expected_z` height
    hint for upper floors. Both are gone. It now returns
    `floor_points_xyz` (measured evidence or None) with no height-based
    fallback. Callers (e.g. main.py's per-unit loop, blueprint_metrology.py)
    must NOT collapse it back into `z_roof - some_height` or invent a Z
    when it's None -- either one reintroduces the exact fabricated-floor
    bug this change fixes. Feed floor_points_xyz/roof_points_xyz directly
    into SectionProfile, and handle None as "no [floor/roof] geometry
    here" rather than substituting a guess.
    """
    acq_info = {}
    pts_x, pts_y, pts_z, pts_cls, pts_floor, floor_id_source, laz_paths = (
        _collect_points_for_footprint(
            poly, lidar_path=lidar_path, conn=conn,
            acquisition_margin_m=acquisition_margin_m,
            acquisition_max_expansions=acquisition_max_expansions,
            acq_info=acq_info,
        )
    )

    if pts_x is None:
        print("   ⚠️ No LiDAR data available. No roof or floor evidence.")
        return ZEvidence(None, None, None, types.MappingProxyType({
            "z_roof": None, "z_roof_basis": None, "point_count": 0,
            "roof_point_count": 0, "floor_point_count": 0,
            "structure_point_count": 0, "slab_point_count": 0, "laz_paths": [],
            "acquisition_window": None, "acquisition_expansions": 0,
            "acquisition_truncated": None,
            "acquisition_extent_complete": None,
            "structure_cloud_incomplete": None,
            "roof_evidence_incomplete": None, "floor_evidence_incomplete": None,
            "structure_semantic_labels": None, "orientation_up": None,
            "roof_points_semantics": None,
        }))

    point_count = len(pts_z)

    if point_count < 20:
        print(f"   ⚠️ LiDAR coverage is sparse ({point_count} points). "
              f"Roof/floor evidence may be thin.")

    # Real slab evidence for THIS storey -- measured points only, never a
    # scalar, never terrain (ground floors included), never
    # balcony/ledge/parapet returns standing in for a slab, never another
    # floor's slabs, never derived from height, and (3D-FIX, this pass)
    # never grid-binned or Z-collapsed.
    #
    # An explicit per-floor scan is read separately from the roof cloud:
    # the interior scan describes this storey, the indexed tiles describe
    # the roof, and neither should be asked to stand in for the other.
    if floor_scan_path:
        floor_points_xyz = get_floor_points_xyz(
            poly, floor_index=floor_index, conn=conn,
            floor_scan_path=floor_scan_path,
            min_points_per_cell=floor_min_points_per_cell,
        )
    else:
        floor_points_xyz = _slab_floor_points(
            pts_x, pts_y, pts_z, pts_cls, pts_floor, poly,
            floor_index=floor_index,
            floor_id_source=floor_id_source,
            min_points_per_cell=floor_min_points_per_cell,
        )

    # ==================================================================
    # ROOF EVIDENCE (3D-FIX, this pass): measured points, not a fitted
    # surface. These are generic CANDIDATE / exposed-surface returns, NOT
    # authoritative roof geometry: roof is never inferred from highest Z,
    # normals, height, position or proximity, only from an explicit roof
    # semantic source (metadata["roof_points_semantics"], set only from an
    # explicitly configured ROOF_CLASSIFICATION_CODE).
    #
    # REMOVED: a degree-3 StandardScaler + PolynomialFeatures + Ridge
    # regression used to fit ONE Z = f(X, Y) surface through every point
    # in the footprint's indexed cloud -- terrain, roof, whatever the
    # tile happened to contain, all fit as if they were one function of
    # (x, y). A height field is the wrong representation for a roof on
    # its face: it cannot express an overhang or cantilever (there is
    # real slab material at an (x, y) with nothing below it -- a height
    # field must still assign that (x, y) exactly one Z), an eave or
    # soffit that curls back under the roofline (multi-valued over its
    # own (x, y): two real surfaces, one predicted value), a vault, dome,
    # or barrel roof (well-conditioned in 3D, degenerate the moment it's
    # asked to be a function of world X and Y once it passes 90 degrees
    # of curvature), or a roof with a genuine opening (a courtyard, a
    # skylight void) -- the fit interpolates straight across it because a
    # regression has no notion of "no measurement here". And because the
    # main.py pipeline this feeds (_detect_slab_surfaces /
    # _measured_surface_brep / _pair_floor_volumes) is built to segment
    # and ray-trace real 3D points, not evaluate a model, the fitted
    # surface was strictly less information than the points that trained
    # it -- regression performed only to be thrown away downstream.
    #
    # What replaces it: `roof_points_xyz` is CANDIDATE / exposed-surface
    # evidence (NOT asserted to be roof; see metadata["roof_points_semantics"]):
    # this footprint's collected cloud with classified ground/slab returns excluded where
    # classification is available -- those categories are already known,
    # by measurement, NOT to be roof (see GROUND_CLASSIFICATION_CODE /
    # SLAB_CLASSIFICATION_CODE above), so excluding them is reading the
    # classification, not modelling anything. No plane, patch, or
    # surface is fit to what remains; that is main.py's job, working
    # from these real points. When no classification exists at all, every
    # point in the tile crop is returned as roof evidence (the same
    # "can't discriminate without classification" honesty
    # _slab_floor_points already applies elsewhere in this module) --
    # logged, not silently assumed.
    #
    # Deliberately NOT clipped to the footprint polygon, unlike the floor
    # path above. `_slab_floor_points` clips to the polygon because a
    # slab return outside the walls is (for an aerial/indoor scan) almost
    # always a neighbouring surface, not this unit's floor. A roof is the
    # opposite case on purpose: a cantilever, jetty, eave, or balcony
    # slab genuinely, measurably extends past the wall line below it, and
    # clipping to the drafted polygon would delete exactly that evidence
    # -- the same acquisition-time clipping bug already fixed for
    # main.py's slab acquisition. The only spatial filter is the LiDAR
    # XY acquisition window applied upstream: seeded by the footprint but
    # grown from the data until measured evidence stops (see ACQUISITION
    # BROAD-PHASE), a performance device rather than a boundary. The
    # blueprint does not define, trim or discard these returns, and roof_points_xyz is the only candidate roof/exposed-surface evidence
    # this function produces (there is no XY-derived outline).
    #
    # EXPLICIT ROOF SEMANTICS. When ROOF_CLASSIFICATION_CODE is explicitly
    # configured, distinct from the ground/slab/wall codes, and roof-classified
    # measured points exist, roof_points_xyz is EXACTLY those classified
    # points (raw XYZ, nothing else) and metadata["roof_points_semantics"] is
    # "roof". Otherwise the generic candidate cloud below is kept and the
    # semantics stay None. Nothing is inferred from Z, normals, height,
    # position or geometry.
    # ==================================================================
    roof_semantic_mask = None
    if (pts_cls is not None and ROOF_CLASSIFICATION_CODE is not None
            and ROOF_CLASSIFICATION_CODE not in (GROUND_CLASSIFICATION_CODE,
                                                 SLAB_CLASSIFICATION_CODE,
                                                 WALL_CLASSIFICATION_CODE)):
        _roof_hit = (pts_cls == ROOF_CLASSIFICATION_CODE)
        if np.any(_roof_hit):
            roof_semantic_mask = _roof_hit
    if pts_cls is not None:
        not_roof = (pts_cls == GROUND_CLASSIFICATION_CODE)
        if SLAB_CLASSIFICATION_CODE is not None:
            not_roof = not_roof | (pts_cls == SLAB_CLASSIFICATION_CODE)
        roof_mask = ~not_roof
        excluded = int(np.count_nonzero(not_roof))
        if excluded:
            print(f"   ℹ️ Excluded {excluded} classified ground/slab return(s) "
                  f"from roof evidence (measured, but known not to be roof).")
    else:
        roof_mask = np.ones(point_count, dtype=bool)
        print("   ℹ️ No classification available for this footprint's cloud -- "
              "roof evidence is every measured return in the tile crop "
              "(ground/slab returns can't be excluded without classification).")
    if roof_semantic_mask is not None:
        roof_mask = roof_semantic_mask  # ONLY explicitly roof-classified points

    if np.any(roof_mask):
        roof_points_xyz = np.column_stack(
            (pts_x[roof_mask], pts_y[roof_mask], pts_z[roof_mask]))
        # A MEASURED diagnostic percentile of the roof-evidence subset --
        # metadata only (see docstring). Ignoring the excluded ground/slab
        # returns keeps this closer to an actual roof elevation than
        # blending in terrain would.
        z_roof = float(np.percentile(roof_points_xyz[:, 2], 95))
        z_roof_basis = "roof_evidence"
    else:
        # Every point in the tile crop was classified ground/slab -- no
        # surviving roof evidence. z_roof still gets a measured value (the
        # whole crop's own percentile) purely as a rough diagnostic
        # number for logging; it is not presented as roof-specific.
        roof_points_xyz = None
        z_roof = float(np.percentile(pts_z, 95))
        z_roof_basis = "whole_crop"
        print("   ⚠️ No roof evidence survived ground/slab exclusion for this "
              "footprint -- metadata z_roof is a diagnostic percentile of the "
              "WHOLE crop, not a roof-specific measurement.")

    # FULL measured structural cloud: everything not classified ground. The
    # roof mask above deliberately drops slab-classified returns (they are
    # not roof); without this array every intermediate/partial slab would
    # vanish here, before any 3D segmentation could see it. Raw XYZ only:
    # no floor attribution, no Z split, no footprint clip, no binning.
    if pts_cls is not None:
        structure_mask = pts_cls != GROUND_CLASSIFICATION_CODE
    else:
        structure_mask = np.ones(point_count, dtype=bool)
    structure_points_xyz = (
        np.column_stack((pts_x[structure_mask], pts_y[structure_mask],
                         pts_z[structure_mask]))
        if np.any(structure_mask) else None
    )
    structure_pt_count = 0 if structure_points_xyz is None else len(structure_points_xyz)
    slab_pt_count = (
        int(np.count_nonzero(pts_cls == SLAB_CLASSIFICATION_CODE))
        if (pts_cls is not None and SLAB_CLASSIFICATION_CODE is not None) else 0
    )

    # Authoritative per-point semantics, ONLY from explicit per-point source
    # classification codes: SLAB_CLASSIFICATION_CODE -> "slab",
    # ROOF_CLASSIFICATION_CODE -> "roof", WALL_CLASSIFICATION_CODE -> "wall",
    # each only when configured. Aligned 1:1 with structure_points_xyz; every
    # other point stays None (unlabelled). Nothing is inferred from Z, height,
    # normals, orientation, position, proximity, connectedness or leftover
    # points; a code configured for two labels is ambiguous and labels neither;
    # with no labelled point the value is None.
    structure_semantic_labels = None
    if structure_points_xyz is not None and pts_cls is not None:
        _codes = {name: code for name, code in (("slab", SLAB_CLASSIFICATION_CODE),
                                                ("roof", ROOF_CLASSIFICATION_CODE),
                                                ("wall", WALL_CLASSIFICATION_CODE))
                  if code is not None}
        _dup = {n for n, c in _codes.items() if sum(1 for o in _codes.values() if o == c) > 1}
        if _dup:
            print(f"   ⚠️ Semantic classification codes for {sorted(_dup)} collide -- "
                  f"ambiguous, so those labels are NOT produced.")
        struct_cls = pts_cls[structure_mask]
        for name, code in _codes.items():
            if name in _dup:
                continue
            hit = (struct_cls == code)
            if np.any(hit):
                if structure_semantic_labels is None:
                    structure_semantic_labels = np.full(len(structure_points_xyz), None, dtype=object)
                structure_semantic_labels[hit] = name
        if structure_semantic_labels is not None:
            structure_semantic_labels.setflags(write=False)

    # roof_points_semantics: "roof" ONLY when roof_points_xyz was built from
    # explicitly roof-classified points (see roof_semantic_mask above), i.e.
    # every point in it is roof-classified; otherwise None. Never inferred.
    roof_points_semantics = "roof" if roof_semantic_mask is not None else None

    floor_pt_count = 0 if floor_points_xyz is None else len(floor_points_xyz)
    roof_pt_count = 0 if roof_points_xyz is None else len(roof_points_xyz)
    print(f"   🧱 Structure cloud: {structure_pt_count} measured non-ground points "
          f"({slab_pt_count} slab-classified across all storeys, kept for 3D segmentation)")
    print(f"   📡 LiDAR: {point_count} points | roof evidence: {roof_pt_count} "
          f"measured points (metadata z_roof ≈ {z_roof:.2f}m, NEVER geometry) | "
          f"floor evidence: {floor_pt_count} points")

    # Tri-state acquisition status, read ONLY from what the collector recorded:
    # True = safety valve tripped, False = collector explicitly established a
    # non-truncated data-driven stop, None = status not recorded (never
    # inferred from a missing key).
    acq_truncated = acq_info.get("truncated")
    if acq_truncated is True:
        print("   ⚠️ Structure, roof and floor evidence from this acquisition are "
              "INCOMPLETE (acquisition truncated): the measured points are kept "
              "unchanged for diagnostics but are non-authoritative for final "
              "geometry, and the window edge is an acquisition limit, never a "
              "building boundary.")

    metadata = types.MappingProxyType({
        "z_roof": z_roof,                    # DIAGNOSTIC ONLY, never geometry
        "z_roof_basis": z_roof_basis,
        "point_count": point_count,
        "roof_point_count": roof_pt_count,
        "floor_point_count": floor_pt_count,
        "structure_point_count": structure_pt_count,
        "slab_point_count": slab_pt_count,
        "laz_paths": list(laz_paths),
        "acquisition_window": acq_info.get("window"),
        "acquisition_expansions": acq_info.get("expansions", 0),
        "acquisition_truncated": acq_truncated,
        "acquisition_extent_complete": (None if acq_truncated is None else not acq_truncated),
        "structure_cloud_incomplete": (None if acq_truncated is None else bool(acq_truncated)),
        # Extent-completeness of each geometry-bearing array (see docstring).
        "roof_evidence_incomplete": (
            bool(acq_truncated) if (roof_points_xyz is not None and acq_truncated is not None)
            else None),
        "floor_evidence_incomplete": (
            bool(acq_truncated)
            if (floor_points_xyz is not None and not floor_scan_path
                and acq_truncated is not None) else None),
        # Semantic contract with main.py (see docstring): authoritative or None.
        "structure_semantic_labels": structure_semantic_labels,
        "orientation_up": None,          # no authoritative UP metadata in the source
        "roof_points_semantics": roof_points_semantics,  # "roof" only from explicit ROOF code, else None
    })
    return ZEvidence(floor_points_xyz, roof_points_xyz, structure_points_xyz, metadata)


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

    # Ground floor (floor_index=0, default). NOTE: the ground floor is no
    # longer a special case -- it requires measured slab evidence exactly
    # like any upper floor. On an ordinary aerial tile with no
    # SLAB_CLASSIFICATION_CODE returns, None here is the CORRECT result.
    #
    # ZEvidence(floor_points_xyz, roof_points_xyz, structure_points_xyz,
    # metadata). No model, no poly_features, no 2D slot, no positional
    # z_roof scalar. "Success" is judged by whether any measured evidence
    # came back.
    (floor_points_xyz, roof_points_xyz,
     structure_points_xyz, meta) = calculate_z_bounds(dummy_room, floor_index=0)
    z_roof = meta["z_roof"]

    had_lidar_coverage = (roof_points_xyz is not None
                          or floor_points_xyz is not None
                          or meta["point_count"] > 0)

    if had_lidar_coverage:
        print(f"✅ LiDAR evidence retrieved!")
        if z_roof is not None:
            print(f"   z_roof (METADATA ONLY, never geometry): {z_roof:.2f}m")
        if roof_points_xyz is not None:
            print(f"   Roof evidence points: {len(roof_points_xyz)} "
                  f"(z range {roof_points_xyz[:, 2].min():.2f}m to "
                  f"{roof_points_xyz[:, 2].max():.2f}m -- overhangs, folds, "
                  f"vaults and any other multi-valued-over-(x,y) shape are "
                  f"exactly as measured here, never fit to a single surface)")
            assert roof_points_xyz.ndim == 2 and roof_points_xyz.shape[1] == 3, (
                "roof evidence must stay raw (M, 3) measured XYZ")
        else:
            print(f"   ℹ️ No roof evidence survived ground/slab classification "
                  f"exclusion for this footprint.")
        if structure_points_xyz is not None:
            print(f"   Structure cloud (roof + every storey's slab returns, "
                  f"ready for 3D segmentation): {len(structure_points_xyz)} points")
        if floor_points_xyz is not None:
            print(f"   Ground-floor slab evidence points: {len(floor_points_xyz)} "
                  f"(z range {floor_points_xyz[:, 2].min():.2f}m to "
                  f"{floor_points_xyz[:, 2].max():.2f}m -- non-planar floors "
                  f"show up as a spread here, not a single number, and never "
                  f"median-binned)")
        else:
            print(f"   ℹ️ No measured-slab evidence for the ground floor -- "
                  f"expected unless config.SLAB_CLASSIFICATION_CODE is set and "
                  f"the point cloud actually carries that classification. "
                  f"Terrain is NOT used as a substitute; for site context only, "
                  f"call get_terrain_context_points() separately.")

        # Upper floor example, floor 2. No expected_z / height hint exists
        # anymore -- this only returns points if the point cloud actually
        # carries SLAB_CLASSIFICATION_CODE-tagged returns AND records that
        # those returns belong to floor 2. On an ordinary aerial LiDAR tile
        # (like this synthetic test) neither exists, so None here is the
        # CORRECT, expected result, not a bug to work around. Note this is
        # now genuinely floor 2's slab, not "every slab in the building"
        # as it was before per-floor selection existed.
        upper_floor_pts = calculate_z_bounds(dummy_room, floor_index=2).floor_points_xyz
        if upper_floor_pts is not None:
            print(f"   Floor 2 slab evidence points: {len(upper_floor_pts)} "
                  f"(z range {upper_floor_pts[:, 2].min():.2f}m to "
                  f"{upper_floor_pts[:, 2].max():.2f}m)")
        else:
            print(f"   ℹ️ No measured-slab evidence recorded for floor 2 -- "
                  f"expected unless config.SLAB_CLASSIFICATION_CODE is set, the "
                  f"point cloud carries that classification, and floor identity "
                  f"is recorded (config.FLOOR_ID_DIMENSION, "
                  f"lidar_tile_index.{TILE_FLOOR_COLUMN}, or an explicit "
                  f"floor_scan_path).")

        # --- Illustrative SectionProfile wiring ---
        # SectionProfile isn't defined in this file, so this is an assumed
        # interface -- confirm/adjust the field names against your actual
        # SectionProfile class. Feed the measured XYZ points straight to a
        # 3D segmentation/topology pipeline; pass no scalar height.
        #
        #   from section_profile import SectionProfile
        #   profile = SectionProfile(
        #       floor_points_xyz=floor_points_xyz,   # (N,3) array or None
        #       roof_points_xyz=roof_points_xyz,     # (M,3) array or None
        #   )
        #   # z_roof (meta["z_roof"]) is deliberately NOT passed: geometry
        #   # never takes a scalar height.
    else:
        print(f"⚠️ Test failed. Did you run `python lidar_indexer.py` first to build the database index?")