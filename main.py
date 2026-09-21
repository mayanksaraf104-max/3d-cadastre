import os
import math
import numbers
from dataclasses import dataclass
from typing import Callable, Optional, Union

import cv2
import numpy as np
from pyproj import Transformer
from shapely.geometry import Polygon, MultiPoint

from ai_to_ogc import extract_ogc_boundaries
# SOI district-boundary jurisdiction lookup (aliased: this module defines its own
# resolve_jurisdiction()/JurisdictionUnavailable boundary below).
from Jurisdiction_resolver import (
    resolve_jurisdiction as _soi_resolve_jurisdiction,
    DistrictJurisdiction,
    JurisdictionUnavailable as _SOIJurisdictionUnavailable,
)
from z_engine import calculate_z_bounds
from blueprint_metrology import (
    resolve_building_envelope,
    BlueprintMetrologyError,
)
# 3D-FIX: build_uniform_storeys() is no longer imported/used. It existed to
# manufacture Storey objects (a floor count x a floor height) that were then
# handed straight to BRepPrimAPI_MakePrism -- i.e. floor count/height WAS
# the building-generation method. Floors are now metadata only: see
# _manual_override_storey_metadata() below, which labels floors but never
# touches geometry creation.
#
# 3D-FIX ROUND 4 (this pass): two remaining places where a 2D polygon was
# still allowed to overrule measured 3D evidence are gone.
#
#   (a) _measured_section_from_points() used to spatially clip its measured
#       (x, y, z) cloud to `poly` (the blueprint footprint, or a volume's
#       `region`) BEFORE tracing a boundary, and return None when fewer
#       than 3 points survived. That is a hard clip of measured evidence by
#       a drafted outline: a cantilever, jetty, bay, eave, or any slab that
#       genuinely oversails the drawing had the oversailing points DELETED,
#       and the resulting section was the intersection of reality with the
#       blueprint rather than reality. The clip is removed. `poly` is now
#       `validation_poly`: it is compared against the finished measured
#       boundary and logged, and it can never add, move, or remove a
#       vertex. The argument order changed (points first) so that no
#       caller can pass a polygon into a position where it would clip.
#
#   (b) _pair_floor_volumes() decided vertical topology in PLAN: it
#       rasterized each measured surface into a 2D `extent`, intersected
#       those extents, and compared the two clouds' MEDIAN Z inside the
#       overlap. Both halves of that are 2.5D. Overlapping XY projections
#       do not mean one surface is over the other -- a cantilever projects
#       over the slab below without roofing all of it, an undercut or a
#       folded eave projects onto plan twice, and a balcony's plan overlap
#       with a distant slab says nothing about what is above it. And a
#       single median Z collapses a tilted, bowed, or curved surface to one
#       number, so two surfaces that interleave (one above over part of the
#       overlap, below over the rest) got a single verdict for the whole
#       region. Pairing is now done against real B-Rep geometry built from
#       the measured points themselves: each surface becomes a triangulated
#       OpenCASCADE face patch, and the ceiling over a point is found by
#       casting a ray from that point along the slab's OWN outward normal
#       and taking the nearest face it actually pierces
#       (IntCurvesFace_ShapeIntersector). Per measured point, first-hit,
#       in true 3D. Overhangs, cantilevers, undercuts, curved/folded
#       surfaces and partial slabs therefore survive by construction, and
#       a volume's bottom/ceiling evidence is the measured subset that
#       genuinely bounds it -- which is what made (a)'s clip removable
#       without a partial slab silently inheriting a full slab's outline.
#
# 3D-FIX ROUND 3: per-floor volumes used to be produced by
# building ONE full-height solid for a unit and then cutting it with a
# horizontal BRepPrimAPI_MakeBox + BRepAlgoAPI_Common at z_dem + storey's
# base_offset / base_offset+height (_floor_cut_planes + _slice_solid_between,
# both removed). That reintroduced the exact z-field assumption the rest of
# this file's overhaul exists to remove: it asserted every storey has a flat
# top and bottom at a constant vertical spacing computed from floor_h,
# whether or not that was what the building's LiDAR/blueprint evidence
# actually showed. A jettied upper storey, a double-height lobby, or a
# storey whose slab tilts across the footprint would all be sliced as if
# they didn't.
#
# Each floor's volume is now its own independently lofted B-Rep solid
# (construct_unit_volume, unchanged), built from that SPECIFIC floor's own
# measured bottom section (its own slab -- one of the distinct slab
# surfaces _detect_slab_surfaces() segments out of z_engine's full measured
# point cloud for the unit) and its own measured top section (the floor
# above's own slab, i.e. this floor's real measured ceiling -- or, for the
# building's topmost storey, the roof point cloud). See
# _measured_section_from_points() below and the per-floor loop in
# run_unified_cadastre_pipeline. There is no base_offset/height/z_lo/z_hi
# arithmetic left anywhere in this construction path, and no horizontal
# cutting plane: a floor with no measured slab of its own is skipped and
# logged, never assumed from its neighbours' spacing.
#
# 3D-FIX ROUND 2: the previous "3D-FIX" still reconstructed
# every floor/roof surface as an explicit height field -- z_field(x, y),
# scalar or callable -- sampled on a grid over the footprint's WORLD (x, y)
# bounds and lofted between exactly two such rings. That is still a 2.5D
# representation wearing B-Rep clothing: a single Z value per (x, y) can
# never encode a cantilever, jetty, corbel, bay window, eave that curls
# back under itself, soffit, arch, or vault -- anything where the true
# surface is not a graph of z over world (x, y). It happened to *look*
# like real 3D because the two rings (floor, roof) could differ in shape
# and slope, but the space BETWEEN them was always a ruled surface with no
# way to bulge out and back in.
#
# This pass removes z_field entirely from the geometry core. Two
# structures replace it:
#   - SectionProfile: an explicit 3D cross-section curve (ring or open
#     profile) at a given level. construct_unit_volume() now lofts through
#     an ARBITRARY ORDERED LIST of these (BRepOffsetAPI_ThruSections over
#     N>=2 wires), so an intermediate profile can project further out (or
#     in) than the sections above and below it -- a genuine overhang, not
#     an assumption ruled out by construction.
#   - _measured_cap_shape(): floor/roof end caps are triangulated between the
#     section's actual measured XYZ vertices (the patch's OWN local (u, v)
#     frame is only a 2D neighbour chart, never a height field), so no
#     surface is interpolated or fitted, and a steep roof pitch, a near-
#     vertical dormer cheek or a tilted wall face keeps its measured shape.
# _measured_section_from_points() (this pass) is the only way a floor or
# roof SectionProfile gets built in this file: a section always comes from
# a real measured (x, y, z) point cloud (z_engine's floor_points_xyz /
# roof_points_xyz), never a flat scalar elevation and never a z=f(x,y)
# regression evaluation. Missing evidence means no section, logged, not a
# silent flat/regression substitute.
from db_engine import CadastreDatabaseEngine, CADASTRE_SRID
from export_ledger import export_postgis_to_glb
from subsurface_engine import True3DSubsurfaceEngine

# Advanced OpenCASCADE Imports
#
# 3D-FIX (geometry overhaul): the old import set only supported translation
# sweeps (BRepPrimAPI_MakePrism) and a boolean-Common/HalfSpace trick for
# "clipping" a roof surface. True B-Rep reconstruction needs: lofting between
# two independent boundary curves (BRepOffsetAPI_ThruSections) so walls can
# taper/slant/curve instead of only translate; sewing independent faces into
# a shell and turning that into a solid (BRepBuilderAPI_Sewing / _MakeSolid);
# healing + validating the result (ShapeFix_Solid / BRepCheck_Analyzer) so an
# invalid shell is caught instead of silently exported; and curve fitting
# (GeomAPI_PointsToBSpline, on top of the existing surface fitter) so a
# curved boundary ring can be built as a real curved edge instead of a
# faceted polygon.
#
# 3D-FIX ROUND 4 (this pass) additionally needs a way to ask a REAL
# QUESTION OF REAL 3D GEOMETRY: "is there measured surface material
# directly over this measured point, and which surface is hit first?"
# That is what replaces _pair_floor_volumes()'s old median-Z-over-
# XY-overlap test (see that function's docstring). It needs:
#   - TopoDS_Compound / BRep_Builder: to assemble each measured surface's
#     own triangulated B-Rep patch (built face-by-face from its actual
#     measured points -- see _measured_surface_brep) into one shape that
#     can be intersected. A measured surface is a SURFACE, not a solid, so
#     it is a compound of faces, not a sewn closed shell -- pretending
#     otherwise would be exactly the kind of invented geometry this file
#     exists to avoid.
#   - gp_Lin + IntCurvesFace_ShapeIntersector: true ray/B-Rep-face
#     intersection. `Perform(lin, tmin, tmax)` returns EVERY face the ray
#     actually pierces with its distance along the ray, so "the ceiling of
#     this point" is the nearest real forward hit -- vertical containment
#     is established by hitting measured geometry, never by two footprints
#     happening to overlap in plan.
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopAbs import TopAbs_FACE, TopAbs_REVERSED, TopAbs_SHELL
from OCC.Core.TopoDS import topods, TopoDS_Compound
from OCC.Core.BRep import BRep_Tool, BRep_Builder
from OCC.Core.TopLoc import TopLoc_Location
from OCC.Core.IntCurvesFace import IntCurvesFace_ShapeIntersector
from OCC.Core.BRepBuilderAPI import (
    BRepBuilderAPI_MakePolygon, BRepBuilderAPI_MakeFace, BRepBuilderAPI_MakeEdge,
    BRepBuilderAPI_MakeWire, BRepBuilderAPI_Sewing, BRepBuilderAPI_MakeSolid,
)
from OCC.Core.gp import gp_Pnt, gp_Vec, gp_GTrsf, gp_Mat, gp_XYZ, gp_Dir, gp_Lin
from OCC.Core.Geom import Geom_Plane
from OCC.Core.BRepOffsetAPI import BRepOffsetAPI_ThruSections
from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Cut
from OCC.Core.TColgp import TColgp_Array2OfPnt, TColgp_Array1OfPnt
from OCC.Core.GeomAPI import (
    GeomAPI_PointsToBSplineSurface, GeomAPI_PointsToBSpline, GeomAPI_ProjectPointOnSurf,
)
from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
from OCC.Core.ShapeFix import ShapeFix_Solid
from OCC.Core.BRepCheck import BRepCheck_Analyzer
# NOTE: BRepPrimAPI_MakeBox / Bnd_Box / BRepBndLib.brepbndlib_Add and
# BRepAlgoAPI_Common are gone from this import list. They existed ONLY to
# build a horizontal cutting slab and Boolean-intersect it against an
# already-built solid (_slice_solid_between, removed -- see the 3D-FIX
# ROUND 3 note above). Nothing in this file builds or trims cadastral
# geometry with a horizontal plane anymore.


LOCATION_SOURCES = ("device_gps", "manual", "survey")


@dataclass(frozen=True)
class PropertyLocation:
    """
    Validated, user-provided location of a property: an authoritative
    metadata record, never geometry.

    `latitude`/`longitude` (decimal degrees, EPSG:4326) are kept EXACTLY as
    supplied; nothing in the pipeline replaces them with a centroid, image
    centre, dummy coordinate or any computed fallback. The location only
    anchors/georeferences the blueprint in the cadastral CRS
    (`to_projected`); all X/Y/Z geometry still comes from measured 3D
    evidence, and no Z is ever derived from it.

    `accuracy_m`: optional horizontal accuracy (metres, > 0).
    `source`: one of LOCATION_SOURCES (provenance of the fix).
    `crs`: must be EPSG:4326 (the lat/lon are WGS84 degrees). Any other CRS is
    rejected rather than guessed at or silently reinterpreted.
    Raises ValueError on any invalid value; nothing is defaulted or repaired.
    """
    latitude: float
    longitude: float
    source: str
    accuracy_m: Optional[float] = None
    crs: str = "EPSG:4326"

    def __post_init__(self):
        def _num(name, v):
            if isinstance(v, bool) or not isinstance(v, numbers.Real) or not math.isfinite(v):
                raise ValueError(f"location {name} must be a finite number, got {v!r}")
            return float(v)

        lat, lon = _num("latitude", self.latitude), _num("longitude", self.longitude)
        if not -90.0 <= lat <= 90.0:
            raise ValueError(f"location latitude {lat} is outside [-90, 90]")
        if not -180.0 <= lon <= 180.0:
            raise ValueError(f"location longitude {lon} is outside [-180, 180]")
        if lat == 0.0 and lon == 0.0:
            raise ValueError("location (0, 0) is not a valid property location "
                             "(it is the usual signature of a missing GPS fix)")
        if self.source not in LOCATION_SOURCES:
            raise ValueError(f"location source must be one of {LOCATION_SOURCES}, got {self.source!r}")
        acc = self.accuracy_m
        if acc is not None:
            acc = _num("accuracy_m", acc)
            if acc <= 0:
                raise ValueError(f"location accuracy_m must be > 0, got {acc}")
        if str(self.crs).strip().upper() not in ("EPSG:4326", "4326"):
            raise ValueError(f"location crs must be EPSG:4326 (WGS84 lat/lon), got {self.crs!r}")
        object.__setattr__(self, "latitude", lat)
        object.__setattr__(self, "longitude", lon)
        object.__setattr__(self, "accuracy_m", acc)
        object.__setattr__(self, "crs", "EPSG:4326")

    def to_projected(self, srid=CADASTRE_SRID):
        """(x, y) in EPSG:`srid`, for spatial computation only. The original
        WGS84 latitude/longitude on this object are never modified."""
        return Transformer.from_crs("EPSG:4326", f"EPSG:{srid}", always_xy=True).transform(
            self.longitude, self.latitude)

    def as_dict(self):
        return {"latitude": self.latitude, "longitude": self.longitude,
                "accuracy_m": self.accuracy_m, "source": self.source, "crs": self.crs}


class JurisdictionError(ValueError):
    """The building's jurisdiction (state/district codes) is invalid or could
    not be established. Raised before any ULPIN registration; never guessed."""


@dataclass(frozen=True)
class Jurisdiction:
    """
    AUTHORITATIVE state/district codes recorded in a property's ULPIN, with
    the provenance of that authority (e.g. the government service or dataset
    that resolved them, or "manual_override" for a controlled/admin input).
    Codes are non-empty strings, kept exactly as given. Nothing is defaulted,
    looked up from a built-in table, or inferred here.
    """
    state_code: str
    district_code: str
    source: str

    def __post_init__(self):
        for name in ("state_code", "district_code", "source"):
            v = getattr(self, name)
            if not isinstance(v, str) or not v.strip():
                raise JurisdictionError(f"jurisdiction {name} must be a non-empty string, got {v!r}")

    def as_dict(self):
        return {"state_code": self.state_code, "district_code": self.district_code,
                "source": self.source}


@dataclass(frozen=True)
class JurisdictionUnavailable:
    """Explicit 'no authoritative jurisdiction' result -- never a guess."""
    reason: str


def _soi_resolver(latitude, longitude):
    """
    Default resolver: the SOI ABDB district layer (jurisdiction_resolver.py).
    The SOI state/district LGD codes are used EXACTLY as returned -- not
    transformed, padded or truncated. No SOI result (dataset not configured,
    point outside every district, ambiguous boundary) -> JurisdictionUnavailable.
    """
    try:
        d: DistrictJurisdiction = _soi_resolve_jurisdiction(latitude, longitude)
    except _SOIJurisdictionUnavailable as e:
        return JurisdictionUnavailable(str(e))
    return Jurisdiction(state_code=d.state_lgd, district_code=d.district_lgd,
                        source="SOI ABDB DISTRICT_BOUNDARY")


_jurisdiction_resolver: Optional[Callable] = _soi_resolver


def set_jurisdiction_resolver(resolver: Optional[Callable]):
    """
    Install the authoritative resolver (e.g. a government/state cadastral
    service or boundary dataset client): a callable
    `(latitude, longitude) -> Jurisdiction | JurisdictionUnavailable`.
    This is the ONLY place a jurisdiction source is plugged in; the pipeline
    never changes. The default is the SOI district resolver (set
    SOI_DISTRICT_BOUNDARY_SHP); pass None to remove it (nothing then resolves).
    """
    global _jurisdiction_resolver
    if resolver is not None and not callable(resolver):
        raise TypeError("resolver must be callable or None")
    _jurisdiction_resolver = resolver


def resolve_jurisdiction(latitude, longitude):
    """
    Resolve the authoritative jurisdiction of a WGS84 point through the
    installed resolver. Returns a `Jurisdiction`, or a `JurisdictionUnavailable`
    when no resolver is installed, the resolver fails or reports none, or it
    returns anything else. It never guesses and holds no coordinates or codes.
    """
    if _jurisdiction_resolver is None:
        return JurisdictionUnavailable(
            "no authoritative jurisdiction resolver is configured (see set_jurisdiction_resolver)")
    try:
        result = _jurisdiction_resolver(latitude, longitude)
    except Exception as e:
        return JurisdictionUnavailable(f"the jurisdiction resolver failed: {e}")
    if isinstance(result, (Jurisdiction, JurisdictionUnavailable)):
        return result
    return JurisdictionUnavailable("the jurisdiction resolver returned an invalid result")


class GNSSCoordinateAnchor:
    def __init__(self, gcp_pixels, gcp_real_world):
        A = np.c_[gcp_pixels, np.ones(gcp_pixels.shape[0])]
        coef_X, _, _, _ = np.linalg.lstsq(A, gcp_real_world[:, 0], rcond=None)
        coef_Y, _, _, _ = np.linalg.lstsq(A, gcp_real_world[:, 1], rcond=None)

        self.matrix = {
            'a': coef_X[0], 'b': coef_X[1], 'c': coef_X[2],
            'd': coef_Y[0], 'e': coef_Y[1], 'f': coef_Y[2]
        }

    def transform_xy(self, x, y):
        m = self.matrix
        gx = m['a'] * x + m['b'] * y + m['c']
        gy = m['d'] * x + m['e'] * y + m['f']
        return gx, gy

    def transform_polygon(self, poly: Polygon) -> Polygon:
        coords = [self.transform_xy(x, y) for x, y in poly.exterior.coords]
        holes = [
            [self.transform_xy(x, y) for x, y in interior.coords]
            for interior in poly.interiors
        ]
        return Polygon(coords, holes)


def robust_solid_to_wkt(shape, global_mirror=False):
    polygons = []
    explorer = TopExp_Explorer(shape, TopAbs_FACE)

    while explorer.More():
        face = topods.Face(explorer.Current())
        is_reversed = face.Orientation() == TopAbs_REVERSED
        needs_flip = is_reversed != global_mirror

        loc = TopLoc_Location()
        triangulation = BRep_Tool.Triangulation(face, loc)

        if triangulation is not None:
            for i in range(1, triangulation.NbTriangles() + 1):
                n1, n2, n3 = triangulation.Triangle(i).Get()
                if needs_flip:
                    n2, n3 = n3, n2

                p1 = triangulation.Node(n1).Transformed(loc.Transformation())
                p2 = triangulation.Node(n2).Transformed(loc.Transformation())
                p3 = triangulation.Node(n3).Transformed(loc.Transformation())

                poly = (f"(({p1.X():.4f} {p1.Y():.4f} {p1.Z():.4f}, "
                        f"{p2.X():.4f} {p2.Y():.4f} {p2.Z():.4f}, "
                        f"{p3.X():.4f} {p3.Y():.4f} {p3.Z():.4f}, "
                        f"{p1.X():.4f} {p1.Y():.4f} {p1.Z():.4f}))")
                polygons.append(poly)
        explorer.Next()

    if not polygons: return None
    return "POLYHEDRALSURFACE Z (" + ", ".join(polygons) + ")"


def extract_smart_boundaries(image_path):
    print("\n[STEP 1] Running Intelligent Vectorization...")
    ai_units = extract_ogc_boundaries(image_path)
    if ai_units and len(ai_units) > 0:
        print(f"✅ AI successfully extracted {len(ai_units)} spatial boundaries.")
        return ai_units

    print("⚠️ Blueprint AI model detected zero units. Engaging Drone/Aerial CV Fallback...")
    img = cv2.imread(image_path)
    if img is None: return []

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Foreground must be the DRAWN structure. On light-paper sheets (blueprints) that is the dark
    # linework; THRESH_BINARY would make the white page the foreground and RETR_EXTERNAL would
    # then return the page frame as the "unit". Dark-background aerial imagery is unchanged.
    _mode = cv2.THRESH_BINARY_INV if np.median(gray) > 127 else cv2.THRESH_BINARY
    _, thresh = cv2.threshold(gray, 150, 255, _mode)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    fallback_units = []
    for cnt in contours:
        epsilon = 0.02 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        if len(approx) >= 3 and cv2.contourArea(cnt) > 1000:
            coords = [(float(pt[0][0]), float(pt[0][1])) for pt in approx]
            coords.append(coords[0])
            raw_poly = Polygon(coords)
            clean_poly = raw_poly.buffer(-5.0, join_style=2).simplify(2.0, preserve_topology=True)
            if not clean_poly.is_empty and clean_poly.geom_type == 'Polygon':
                fallback_units.append({"polygon": clean_poly, "confidence": 0.95})
    return fallback_units


class SectionProfile:
    """
    One explicit 3D cross-section of a unit's envelope at a given level --
    the fundamental replacement for a height-field "floor" or "roof".

    `points_3d` is an ordered ring of real (x, y, z) vertices (already
    measured/extracted -- never re-derived from a z=f(x,y) evaluation).
    `holes_3d` is a list of such rings for interior openings (atriums,
    light-wells) at this same level. `level` is any orderable value used
    only to sort sections before lofting (a measured elevation, a storey
    index, or an arc-length param along a path) -- it is NOT used to
    compute geometry, so nothing here is a function of level.

    A wall reconstructed from >= 3 sections can bulge outward at an
    intermediate section and back in at the next one -- e.g. a jettied
    upper floor, a corbelled cornice, a bay window, a tapered plinth --
    because BRepOffsetAPI_ThruSections lofts a genuine ruled/interpolated
    skin THROUGH each independent profile in turn, not a graph over (x, y).
    Two sections (floor + roof) are NOT enough: with no measured intermediate
    wall section the lateral boundary is unmeasured, and construct_unit_volume
    rejects the volume rather than fabricate a wall between two rings.
    """
    __slots__ = ("level", "points_3d", "holes_3d", "smooth", "surface_points_3d",
                 "lateral_normals")

    def __init__(self, level, points_3d, holes_3d=None, smooth=False, surface_points_3d=None):
        self.level = level
        self.points_3d = [(float(x), float(y), float(z)) for x, y, z in points_3d]
        self.holes_3d = [
            [(float(x), float(y), float(z)) for x, y, z in ring]
            for ring in (holes_3d or [])
        ]
        self.smooth = smooth
        # Set only by _measured_wall_sections for a section supported by an
        # UNLABELLED measured patch: the patch's normals at the snapped
        # vertices, used by _wall_sections_for_volume to test that the patch
        # runs between the volume's bottom and ceiling. None = wall-labelled
        # support (or not a wall section). Never used to build geometry.
        self.lateral_normals = None
        # Optional DENSER interior point cloud (a true LiDAR/photogrammetry
        # "surface patch", not just this section's rim) used to fit the
        # end-cap surface's actual curvature. The boundary ring alone can
        # only ever support a flat or gently-sloped fit -- fitting a domed,
        # vaulted, or otherwise genuinely curved interior needs real points
        # FROM that interior, not an extrapolation from its edge. Falls
        # back to the boundary ring itself when no denser cloud exists,
        # which is a real, logged evidence gap, not a silent assumption.
        self.surface_points_3d = (
            [(float(x), float(y), float(z)) for x, y, z in surface_points_3d]
            if surface_points_3d is not None else None
        )


def _wire_from_points3d(points3d, smooth=False):
    """
    Build a wire from an ordered list of explicit (x, y, z) vertices.

    Every vertex carries its own elevation, so the wire can trace a
    sloped, tilted, or overhanging ring -- it is never re-flattened onto
    a plane or re-evaluated from a field. `smooth=True` (set when the
    extractor flags this ring as curve-derived -- e.g. a rounded facade
    traced as a dense polyline) fits ONE closed BSpline curve through the
    points instead of straight polygon edges, so a curved boundary lofts
    into a genuinely curved wall surface rather than a faceted
    approximation of one.
    """
    pts = points3d[:-1] if len(points3d) > 1 and points3d[0] == points3d[-1] else points3d
    if smooth and len(pts) >= 4:
        n = len(pts)
        arr = TColgp_Array1OfPnt(1, n + 1)
        for i, (x, y, z) in enumerate(pts, start=1):
            arr.SetValue(i, gp_Pnt(float(x), float(y), float(z)))
        arr.SetValue(n + 1, gp_Pnt(float(pts[0][0]), float(pts[0][1]), float(pts[0][2])))
        curve = GeomAPI_PointsToBSpline(arr).Curve()
        edge = BRepBuilderAPI_MakeEdge(curve).Edge()
        wire_builder = BRepBuilderAPI_MakeWire()
        wire_builder.Add(edge)
        return wire_builder.Wire()

    mk = BRepBuilderAPI_MakePolygon()
    for x, y, z in pts:
        mk.Add(gp_Pnt(float(x), float(y), float(z)))
    mk.Close()
    return mk.Wire()


def _fit_local_frame(points_3d):
    """
    Fit a local (centroid, u_dir, v_dir, normal) frame to a real 3D point
    cloud via PCA (SVD on the centered points). The least-variance axis is
    taken as the patch's own normal.

    This is the mechanism that lets a floor/roof end-cap surface be fit
    for what it actually is instead of what world-XY happens to see it
    as: a steep roof pitch, a near-vertical dormer cheek, or a tilted wall
    face is a well-behaved single-valued surface over ITS OWN plane even
    though it is degenerate or multi-valued as a function of world
    (x, y). Only a patch that folds back on ITSELF (relative to its own
    normal -- an arch, a vault, a true undercut) still needs the
    multi-section loft path in loft_wall_shell rather than a single fitted
    patch, and that path never funnels through this function at all.
    """
    pts = np.asarray(points_3d, dtype=float)
    centroid = pts.mean(axis=0)
    centered = pts - centroid
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    u_dir, v_dir, normal = vt[0], vt[1], vt[2]
    return centroid, u_dir, v_dir, normal


def _projection_is_ambiguous(points_xyz, uv, h, gap_spacings=3.0, cell_spacings=2.5):
    """
    True when a single local (u, v) projection of a measured cloud is
    multi-valued: distinct measured sheets (a fold, an undercut, stacked or
    overlapping slabs) share the same (u, v) region, so one surface or one
    boundary traced in that projection would connect distinct 3D regions.

    (u, v) is binned into cells `cell_spacings` median (u, v) spacings wide;
    inside a cell the points are ordered by height `h` along the frame's
    normal. Two height-adjacent points that are more than `gap_spacings`
    median 3D spacings apart in BOTH height and true 3D distance are two
    sheets, not one continuous surface (a steep but single, continuously
    sampled surface has no such gap: its height-adjacent points are
    ordinary neighbours). Uses measured points only.
    """
    from scipy.spatial import cKDTree
    pts = np.asarray(points_xyz, dtype=float)
    uv = np.asarray(uv, dtype=float)
    h = np.asarray(h, dtype=float)
    if pts.shape[0] < 4:
        return False
    sp3 = max(float(np.median(cKDTree(pts).query(pts, k=2)[0][:, 1])), 1e-6)
    spuv = max(float(np.median(cKDTree(uv).query(uv, k=2)[0][:, 1])), 1e-6)
    thr = gap_spacings * sp3
    ij = np.floor((uv - uv.min(axis=0)) / (cell_spacings * spuv)).astype(np.int64)
    key = ij[:, 0] * (int(ij[:, 1].max()) + 1) + ij[:, 1]
    order = np.lexsort((h, key))
    k_s, h_s, p_s = key[order], h[order], pts[order]
    cand = np.flatnonzero((k_s[1:] == k_s[:-1]) & ((h_s[1:] - h_s[:-1]) > thr))
    if cand.size == 0:
        return False
    return bool((np.linalg.norm(p_s[cand + 1] - p_s[cand], axis=1) > thr).any())


def _triangle_face(p0, p1, p2):
    """
    Planar triangular face through three MEASURED vertices (three points are
    exactly planar, so the face is the evidence, not a fit to it).
    """
    a, b, c = (np.asarray(p, dtype=float) for p in (p0, p1, p2))
    nrm = np.cross(b - a, c - a)
    ln = float(np.linalg.norm(nrm))
    if ln < 1e-12:
        raise ValueError("collinear measured vertices cannot form a cap facet")
    plane = Geom_Plane(gp_Pnt(*a.tolist()), gp_Dir(*(nrm / ln).tolist()))
    wire = BRepBuilderAPI_MakePolygon(
        gp_Pnt(*a.tolist()), gp_Pnt(*b.tolist()), gp_Pnt(*c.tolist()), True).Wire()
    return BRepBuilderAPI_MakeFace(plane, wire).Face()


def _ear_clip_ring(uv):
    """
    Triangulate a simple closed ring (ordered vertices, given in a 2D chart)
    by ear clipping. Returns index triples into the ring; uses ONLY the ring's
    own vertices (no new points). Raises ValueError when it cannot finish.
    """
    uv = np.asarray(uv, dtype=float)
    idx = list(range(len(uv)))
    x, y = uv[:, 0], uv[:, 1]
    if 0.5 * (np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) < 0:
        idx.reverse()  # counter-clockwise in the chart
    scale = max(float(np.ptp(uv[:, 0])), float(np.ptp(uv[:, 1])), 1e-12)
    eps = 1e-12 * scale * scale
    tris = []
    while len(idx) > 3:
        m = len(idx)
        for a in range(m):
            i, j, k = idx[a - 1], idx[a], idx[(a + 1) % m]
            p, q, r = uv[i], uv[j], uv[k]
            if (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0]) <= eps:
                continue  # reflex or collinear vertex: not an ear
            blocked = False
            for o in idx:
                if o in (i, j, k):
                    continue
                w = uv[o]
                d1 = (q[0] - p[0]) * (w[1] - p[1]) - (q[1] - p[1]) * (w[0] - p[0])
                d2 = (r[0] - q[0]) * (w[1] - q[1]) - (r[1] - q[1]) * (w[0] - q[0])
                d3 = (p[0] - r[0]) * (w[1] - r[1]) - (p[1] - r[1]) * (w[0] - r[0])
                if d1 >= -eps and d2 >= -eps and d3 >= -eps:
                    blocked = True
                    break
            if blocked:
                continue
            tris.append((i, j, k))
            idx.pop(a)
            break
        else:
            raise ValueError("ring cannot be triangulated from its own measured vertices")
    tris.append(tuple(idx))
    return tris


def _measured_cap_shape(points_3d, holes_3d=None, surface_points_3d=None):
    """
    End-cap geometry built ONLY from actual measured XYZ vertices: a compound
    of planar triangular faces whose corners are measured points. No
    interpolation, resampling, gridding, extrapolation or synthetic XYZ, and
    no height field: the local (u, v) frame is used solely as a 2D chart to
    decide which measured points are triangle neighbours; every coordinate of
    every face stays the measured 3D one, so curved, sloped or vertical caps
    keep their true shape.

    With a dense measured cloud (`surface_points_3d`): the triangles are the
    cloud's own alpha-complex (`_alpha_triangles`, the same primitive
    `_measured_section_from_points` traced the boundary from), so the cap
    covers exactly the measured support and never bridges a gap. Multi-sheet
    clouds (see `_projection_is_ambiguous`) are rejected (ValueError), and the
    complex's boundary edges must equal the section's measured rim/hole edges
    exactly, else ValueError -- nothing is trimmed, snapped or invented to
    make them agree.

    Without one (e.g. an interior void's cap): the ring's own vertices are
    ear-clipped; holes are then unsupported (ValueError).

    Returns (compound, measured_vertices (M, 3)).
    """
    def _open_ring(r):  # drop a repeated closing vertex
        r = np.asarray(r, dtype=float)
        return r[:-1] if len(r) > 1 and np.array_equal(r[0], r[-1]) else r

    ring = _open_ring(points_3d)
    holes = [_open_ring(h) for h in (holes_3d or [])]
    dense = surface_points_3d is not None and len(surface_points_3d) >= 3

    if dense:
        pts = np.asarray(surface_points_3d, dtype=float)
        centroid, u_dir, v_dir, normal = _oriented_local_frame(pts)
        local = pts - centroid
        uv = np.column_stack((local @ u_dir, local @ v_dir))
        if _projection_is_ambiguous(pts, uv, local @ normal):
            raise ValueError("measured support is multi-valued in its own local (u, v) frame "
                             "(distinct sheets share the same (u, v) region -- fold/undercut/"
                             "stacked); refusing to build one cap through them")
        kept, _alpha = _alpha_triangles(uv)
        if kept is None:
            raise ValueError("no measured support to build a cap from")
        canon = {}
        cid = np.array([canon.setdefault(tuple(p), n) for n, p in enumerate(pts.tolist())])
        edge_count = {}
        for a, b, c in kept:
            for e0, e1 in ((a, b), (b, c), (c, a)):
                e0, e1 = int(cid[e0]), int(cid[e1])
                key = (e0, e1) if e0 < e1 else (e1, e0)
                edge_count[key] = edge_count.get(key, 0) + 1
        boundary = {e for e, cnt in edge_count.items() if cnt == 1}
        want = set()
        for r in [ring] + holes:
            try:
                ids = [canon[tuple(p)] for p in r.tolist()]
            except KeyError:
                raise ValueError("a section boundary vertex is not one of the measured cap points")
            for e0, e1 in zip(ids, ids[1:] + ids[:1]):
                want.add((e0, e1) if e0 < e1 else (e1, e0))
        if boundary != want:
            raise ValueError("the measured-point triangulation's boundary does not match the "
                             "section's measured rim/holes; refusing to trim or invent a cap")
        tri_pts = [(pts[a], pts[b], pts[c]) for a, b, c in kept]
        verts = pts
    else:
        if holes:
            raise ValueError("a cap with holes needs the section's dense measured cloud")
        centroid, u_dir, v_dir, _n = _oriented_local_frame(ring)
        local = ring - centroid
        tri = _ear_clip_ring(np.column_stack((local @ u_dir, local @ v_dir)))
        tri_pts = [(ring[a], ring[b], ring[c]) for a, b, c in tri]
        verts = ring

    compound = TopoDS_Compound()
    builder = BRep_Builder()
    builder.MakeCompound(compound)
    for p0, p1, p2 in tri_pts:
        builder.Add(compound, _triangle_face(p0, p1, p2))
    return compound, verts


# Explicit survey tolerance (metres) within which a fitted supporting surface
# must CONTAIN every measured boundary vertex. Tiny on purpose: a surface that
# needs a larger move to reach the evidence is the wrong surface.
SURVEY_VERTEX_ON_SURFACE_TOL_M = 1e-3


def _verify_ring_on_surface(points_3d, surface, tol=SURVEY_VERTEX_ON_SURFACE_TOL_M):
    """
    Check that every measured vertex of a boundary ring is one of the
    measured vertices the end cap is built from (`surface` = that (M, 3)
    array of measured XYZ, from `_measured_cap_shape`), within `tol` metres,
    and return the ring's vertices EXACTLY as measured.

    Measured XYZ is never moved: no projection, no snapping. A vertex with no
    matching measured cap vertex is rejected (ValueError) rather than
    editing the evidence to fit.
    """
    from scipy.spatial import cKDTree
    tree = cKDTree(np.asarray(surface, dtype=float))
    exact = []
    for x, y, z in points_3d:
        if tree.query([float(x), float(y), float(z)])[0] > tol:
            raise ValueError("a measured boundary vertex is not one of the cap's measured "
                             "vertices; rejecting rather than moving evidence")
        exact.append((float(x), float(y), float(z)))
    return exact


def _make_trimmed_face(points_3d, holes_3d=None, smooth=False, surface_points_3d=None,
                       build_cap=True):
    """
    Build a section's end cap and boundary wires from MEASURED vertices only.

    The cap (`_measured_cap_shape`) is a compound of planar triangles whose
    corners are measured points -- no fitted, interpolated or resampled
    surface, no synthetic XYZ. `surface_points_3d`, when supplied, is the
    section's dense measured cloud whose triangulation forms the cap; its
    boundary must equal the section's measured rim/holes (ValueError
    otherwise). Boundary vertices are used EXACTLY as measured
    (`_verify_ring_on_surface`).

    `build_cap=False` (intermediate wall sections, which are never capped)
    skips the cap and returns only the wires.

    Wires are always polygonal through the measured vertices: a spline through
    them would interpolate between measurements and would not coincide with
    the triangulated cap's edges, so `smooth` no longer changes the wires.

    Returns (cap or None, outer_wire, hole_wires) -- the wires are exposed so
    loft_wall_shell() can loft the walls between the EXACT same wire
    geometry the cap's rim uses, so sewing produces a genuinely closed solid.
    """
    if build_cap:
        cap, verts = _measured_cap_shape(points_3d, holes_3d, surface_points_3d)
        outer_pts = _verify_ring_on_surface(points_3d, verts)
        hole_ring_pts = [_verify_ring_on_surface(h, verts) for h in (holes_3d or [])]
    else:
        cap = None
        outer_pts = [(float(x), float(y), float(z)) for x, y, z in points_3d]
        hole_ring_pts = [[(float(x), float(y), float(z)) for x, y, z in h]
                         for h in (holes_3d or [])]

    outer_wire = _wire_from_points3d(outer_pts, smooth=False)
    hole_wires = [_wire_from_points3d(h, smooth=False) for h in hole_ring_pts]
    return cap, outer_wire, hole_wires


def build_face_with_holes(points_3d, holes_3d=None, smooth=False):
    """
    Floor/ceiling/roof end cap as a compound of triangles between measured
    vertices only (no fitted or interpolated surface) -- see
    _make_trimmed_face / _measured_cap_shape. Kept as a named entry
    point for any external caller that only needs the face;
    construct_unit_volume calls _make_trimmed_face directly, once per
    section, so it can reuse each section's exact boundary wire for wall
    lofting (see loft_wall_shell below).
    """
    face, _outer_wire, _hole_wires = _make_trimmed_face(points_3d, holes_3d, smooth=smooth)
    return face


def loft_wall_shell(section_wires, section_hole_wires):
    """
    Build the lateral (wall) surface of one unit as a true lofted B-Rep
    skin THROUGH an arbitrary ordered list of >= 2 section wires -- never
    a flat footprint swept by a constant vector, and never limited to
    exactly a floor wire and a roof wire.

    `section_wires` must be the EXACT wires returned by
    `_make_trimmed_face()` for each section (see construct_unit_volume) --
    lofting between the same wire objects the end-cap faces were built
    from is what guarantees the wall shell's rails coincide with those
    faces' boundaries, so sewing them together produces a genuinely closed
    solid instead of a near-miss with a hairline gap.

    This is the concrete mechanism for overhangs, jetties, corbels, bay
    windows, and tapers: BRepOffsetAPI_ThruSections lofts THROUGH every
    section in the order given, including any that project further out
    (or in) than their neighbours. A 2-section (floor + roof) loft is REJECTED:
    it would invent an unmeasured lateral wall between two measured rings, so
    at least one measured intermediate wall section is required. With 3+ sections an
    intermediate profile can bulge and return, which a height field could
    never represent regardless of how it was fit.

    Interior rings (light-wells / atriums) are lofted and returned
    separately, so an atrium can be subtracted as its own true volume
    (which may itself taper or curve through its own sections) instead of
    a hole punched straight through a flat extrusion.
    """
    if len(section_wires) < 3:
        raise ValueError("loft_wall_shell needs >= 3 section wires (floor, at least one MEASURED "
                         "intermediate wall section, roof); a 2-ring loft would fabricate an "
                         "unmeasured wall")

    outer_loft = BRepOffsetAPI_ThruSections(False, False, 1e-6)
    for wire in section_wires:
        outer_loft.AddWire(wire)
    outer_loft.Build()
    outer_shell = outer_loft.Shape()

    inner_shells = []
    n_holes = min(len(h) for h in section_hole_wires) if section_hole_wires else 0
    for hole_idx in range(n_holes):
        inner_loft = BRepOffsetAPI_ThruSections(False, False, 1e-6)
        for section_holes in section_hole_wires:
            inner_loft.AddWire(section_holes[hole_idx])
        inner_loft.Build()
        inner_shells.append(inner_loft.Shape())

    return outer_shell, inner_shells


def _closed_void_solid(void_wall_shell, sections, hole_wires_per_section, hole_idx):
    """
    Close an interior void's lofted wall shell into a validated solid, capping
    it with triangles between the measured vertices of the first/last hole
    rings (ear-clipped; no new points) whose edges coincide with the loft's
    rails. Raises ValueError when a valid closed void can't be established.
    """
    cap_faces = []
    for k in (0, -1):
        cap_faces.append(_measured_cap_shape(sections[k].holes_3d[hole_idx])[0])

    sewing = BRepBuilderAPI_Sewing(1e-6)
    sewing.Add(void_wall_shell)
    for f in cap_faces:
        sewing.Add(f)
    sewing.Perform()
    sewn = sewing.SewedShape()

    shells = []
    it = TopExp_Explorer(sewn, TopAbs_SHELL)
    while it.More():
        shells.append(topods.Shell(it.Current()))
        it.Next()
    if len(shells) != 1:
        raise ValueError(f"interior void sewed into {len(shells)} shells; expected one closed shell")
    n_sewn = n_shell = 0
    it = TopExp_Explorer(sewn, TopAbs_FACE)
    while it.More():
        n_sewn += 1
        it.Next()
    it = TopExp_Explorer(shells[0], TopAbs_FACE)
    while it.More():
        n_shell += 1
        it.Next()
    if n_sewn != n_shell:
        raise ValueError("interior void has faces outside its shell; not closed")

    builder = BRepBuilderAPI_MakeSolid()
    builder.Add(shells[0])
    fixer = ShapeFix_Solid(builder.Solid())
    fixer.Perform()
    void = fixer.Solid()
    if not BRepCheck_Analyzer(void).IsValid():
        raise ValueError("interior void is not a valid closed solid; cut rejected")
    return void


def construct_unit_volume(sections, smooth=False):
    """
    Assemble ONE true B-Rep solid for an AI/LiDAR/blueprint-derived unit
    from an ordered list of >= 3 `SectionProfile`s -- real 3D cross-
    sections of the unit's envelope (floor, any intermediate evidence
    profile, roof/ceiling) -- with NO step anywhere inferring a wall from
    "floor height", a z_min/z_max pair standing in for real geometry, a
    vertical-extrusion vector, or a z=f(x, y) evaluation of any kind.

    Each end section's cap is a triangulation of that section's own measured
    points (actual measured XYZ vertices only, no fitted/interpolated
    surface -- see _measured_cap_shape).
    Only the FIRST and LAST sections' faces are kept as the solid's floor
    and roof/ceiling faces; any sections in between exist purely to steer
    the wall loft (see loft_wall_shell) and are not capped -- they are the
    real evidence for a taper, setback, jetty, corbel, or overhang between
    the floor and the roof.

    `sections` must already be sorted in traversal order (bottom to top,
    or along whatever path they were measured on) -- this function does
    not re-order them, since "level" is caller-defined metadata, not a
    geometric quantity this function is allowed to reinterpret.

    Two sections (floor + roof) are REJECTED (ValueError): the lateral boundary
    between them is unmeasured, and a ruled loft would fabricate a wall. At
    least one measured intermediate wall section is required; callers must
    skip the volume otherwise.

    Returns a validated (BRepCheck_Analyzer-passing) solid, or raises if
    the reconstructed shell can't be closed into a valid OGC 3D solid.
    """
    if len(sections) < 3:
        raise ValueError("construct_unit_volume needs >= 3 sections (floor, at least one MEASURED "
                         "intermediate wall section, roof); the lateral boundary is not measured")

    faces = []
    wires = []
    hole_wires_per_section = []
    for si, section in enumerate(sections):
        face, wire, hole_wires = _make_trimmed_face(
            section.points_3d, section.holes_3d, smooth=(smooth or section.smooth),
            surface_points_3d=section.surface_points_3d,
            build_cap=(si == 0 or si == len(sections) - 1),  # intermediates are never capped
        )
        faces.append(face)
        wires.append(wire)
        hole_wires_per_section.append(hole_wires)

    # Every section must carry the same number of hole rings: a hole missing
    # from one section leaves no measured-supported void to loft through, and
    # silently lofting only the common subset would fill measured openings.
    if len({len(h) for h in hole_wires_per_section}) > 1:
        raise ValueError("interior hole count differs between sections; cannot establish a "
                         "measured-supported closed void")

    outer_wall_shell, inner_wall_shells = loft_wall_shell(wires, hole_wires_per_section)

    sewing = BRepBuilderAPI_Sewing(1e-6)
    sewing.Add(faces[0])   # floor
    sewing.Add(faces[-1])  # roof/ceiling
    sewing.Add(outer_wall_shell)
    sewing.Perform()
    sewn = sewing.SewedShape()

    solid_builder = BRepBuilderAPI_MakeSolid()
    shells = []
    shell_explorer = TopExp_Explorer(sewn, TopAbs_SHELL)
    while shell_explorer.More():
        shells.append(topods.Shell(shell_explorer.Current()))
        shell_explorer.Next()
    if len(shells) > 1:
        # One solid is expected. Keeping only one shell would discard the
        # other measured geometry, so reject explicitly instead.
        raise ValueError(f"sewing produced {len(shells)} independent shells where one solid is "
                         f"expected; refusing to discard measured geometry")
    if shells:
        # Faces sewn but left outside the single shell would be dropped too.
        n_sewn, n_shell = 0, 0
        fe = TopExp_Explorer(sewn, TopAbs_FACE)
        while fe.More():
            n_sewn += 1
            fe.Next()
        fe = TopExp_Explorer(shells[0], TopAbs_FACE)
        while fe.More():
            n_shell += 1
            fe.Next()
        if n_sewn != n_shell:
            raise ValueError("sewing left measured faces outside the shell; refusing to discard them")
        solid_builder.Add(shells[0])
    else:
        # Sewing produced something without a distinct SHELL wrapper (can
        # happen for a very simple two-triangle skin) -- wrap it directly.
        solid_builder.Add(sewn)
    solid = solid_builder.Solid()

    fixer = ShapeFix_Solid(solid)
    fixer.Perform()
    solid = fixer.Solid()

    for hole_idx, inner_shell in enumerate(inner_wall_shells):
        # Subtract each atrium/light-well as its own true volume (which may
        # itself taper, curve, or overhang through its own sections)
        # instead of punching a straight hole through a flat extrusion. The
        # void must be a validated CLOSED solid built from the same section
        # hole wires as the loft; otherwise the cut is rejected.
        void = _closed_void_solid(inner_shell, sections, hole_wires_per_section, hole_idx)
        cut = BRepAlgoAPI_Cut(solid, void)
        if not cut.IsDone() or cut.HasErrors():
            raise ValueError("Boolean cut of the interior void failed")
        solid = cut.Shape()

    analyzer = BRepCheck_Analyzer(solid)
    if not analyzer.IsValid():
        raise ValueError(
            "Reconstructed unit solid failed OGC 3D validity check "
            "(non-closed or self-intersecting shell)"
        )

    return solid


def _ring_signed_area_2d(pts, ring_idx):
    """Shoelace area (always >= 0) of a closed ring of indices into `pts`."""
    r = pts[ring_idx]
    x, y = r[:, 0], r[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def _alpha_triangles(points_2d, alpha=None):
    """
    Delaunay triangulation of a real 2D point set, keeping ONLY triangles
    whose circumradius is <= `alpha` -- i.e. triangles tight enough that
    they do not bridge a real gap in the measured coverage.

    This is the single shared primitive behind two things that must agree
    with each other:
      - `_alpha_shape_rings`, which traces the boundary edges of the
        surviving complex into closed rings (a section's outline);
      - `_measured_surface_brep`, which turns the surviving triangles
        THEMSELVES into a B-Rep face patch (the geometry
        `_pair_floor_volumes` casts rays against).
    Because both come from the same alpha-complex, the surface a ray is
    tested against covers exactly the area the section boundary claims --
    no invented material beyond where points were actually measured, and
    no phantom hit over an unmeasured hole.

    `points_2d` is whatever 2D parametrization the caller has chosen. For a
    slab/roof patch that is the surface's OWN local (u, v) frame (see
    `_fit_local_frame`), never world (x, y): a steep pitch, a dormer cheek,
    or an eave that curls back under is single-valued in its own frame and
    hopelessly multi-valued in plan.

    `alpha` defaults to a small multiple of THIS point set's own median
    nearest-neighbour spacing, so it adapts to this surface's density
    rather than a fixed real-world distance.

    Returns (kept_simplices, alpha_used) -- (None, ...) when there are
    fewer than 3 points, the set is degenerate (collinear/duplicate), or
    nothing at all survives the alpha test.
    """
    pts = np.asarray(points_2d, dtype=float)
    if pts.shape[0] < 3:
        return None, None

    from scipy.spatial import Delaunay, cKDTree
    try:
        tri = Delaunay(pts)
    except Exception:
        return None, None  # degenerate cloud (e.g. all collinear/duplicate)

    if alpha is None:
        nn_d, _ = cKDTree(pts).query(pts, k=2)
        nn_spacing = float(np.median(nn_d[:, 1]))
        alpha = max(3.0 * nn_spacing, 1e-6)

    simplices = tri.simplices
    p0, p1, p2 = pts[simplices[:, 0]], pts[simplices[:, 1]], pts[simplices[:, 2]]
    a = np.linalg.norm(p1 - p2, axis=1)
    b = np.linalg.norm(p0 - p2, axis=1)
    c = np.linalg.norm(p0 - p1, axis=1)
    s = (a + b + c) / 2.0
    area = np.sqrt(np.clip(s * (s - a) * (s - b) * (s - c), 0.0, None))
    with np.errstate(divide="ignore", invalid="ignore"):
        circum_r = np.where(area > 1e-12, (a * b * c) / (4.0 * area + 1e-15), np.inf)
    kept = simplices[circum_r <= alpha]
    if kept.shape[0] == 0:
        return None, alpha  # alpha too tight for this cloud -- nothing survives
    return kept, alpha


def _alpha_shape_rings(points_xy, alpha=None):
    """
    Concave hull / alpha-shape boundary of a real, possibly irregular 2D
    point set, via Delaunay triangulation: keep only triangles whose
    circumradius is <= `alpha` (i.e. "tight" enough that they don't bridge
    a real gap in the measured coverage), then trace the boundary edges of
    what survives into closed rings.

    This -- not a bounding box, not a blueprint outline, not a convex hull
    -- is what actually preserves a setback, a partial slab's ragged edge,
    or a genuine overhang's silhouette: a triangle that would have to span
    an unmeasured gap to close the shape has too large a circumradius and
    is dropped, so the boundary hugs exactly where points were measured.

    `alpha` (max circumradius) defaults to a small multiple of THIS
    cloud's own median nearest-neighbour spacing, so it adapts to this
    particular surface's point density rather than a fixed real-world
    distance.

    Returns (exterior_ring, hole_rings): `exterior_ring` is a closed list
    of indices into `points_xy` (first index repeated at the end); the
    largest-area ring found. `hole_rings` is a list of similarly-closed
    smaller rings whose centroid lies inside the exterior ring -- a real
    interior gap in the measured coverage (an atrium, a light-well, or
    simply a hole punched through this specific slab), not a blueprint
    opening. Returns (None, []) when there are too few points, the point
    set is degenerate, or the alpha-complex boundary is non-manifold (a
    pinch point, or two lobes touching at a single point) and so cannot be
    traced unambiguously into simple rings -- callers must treat that as "no
    boundary" (no convex-hull fallback: it would bridge unmeasured gaps).
    """
    pts = np.asarray(points_xy, dtype=float)
    n = pts.shape[0]
    if n < 3:
        return None, []

    if n == 3:
        # A single triangle IS its own boundary -- no alpha threshold to
        # choose, and Delaunay of exactly 3 points is degenerate to set up.
        area2 = abs((pts[1, 0] - pts[0, 0]) * (pts[2, 1] - pts[0, 1])
                     - (pts[2, 0] - pts[0, 0]) * (pts[1, 1] - pts[0, 1]))
        if area2 < 1e-12:
            return None, []  # collinear -- no real 2D boundary to trace
        return [0, 1, 2, 0], []

    kept, _alpha_used = _alpha_triangles(pts, alpha=alpha)
    if kept is None:
        # Degenerate cloud, or alpha too tight for it -- nothing survives.
        return None, []

    # An edge shared by exactly one surviving triangle is on the boundary
    # of the alpha-shape (outer silhouette OR the rim of an interior gap);
    # an edge shared by two survives only in the shape's interior.
    edge_count = {}
    for u, v, w in kept:
        for e0, e1 in ((u, v), (v, w), (w, u)):
            key = (e0, e1) if e0 < e1 else (e1, e0)
            edge_count[key] = edge_count.get(key, 0) + 1
    boundary_edges = [e for e, cnt in edge_count.items() if cnt == 1]
    if not boundary_edges:
        return None, []

    adj = {}
    for u, v in boundary_edges:
        adj.setdefault(u, []).append(v)
        adj.setdefault(v, []).append(u)
    if any(len(neigh) != 2 for neigh in adj.values()):
        # Non-manifold boundary (a pinch point, or alpha carving the patch
        # into touching-but-distinct lobes) -- ambiguous to trace into
        # simple rings without guessing; let the caller fall back instead.
        return None, []

    visited = set()
    rings = []
    for start in adj:
        if start in visited:
            continue
        ring = [start]
        visited.add(start)
        prev, cur = None, start
        closed = False
        for _ in range(len(adj) + 1):
            neighbors = adj[cur]
            nxt = neighbors[0] if neighbors[0] != prev else neighbors[1]
            if nxt == start:
                closed = True
                break
            ring.append(nxt)
            visited.add(nxt)
            prev, cur = cur, nxt
        if not closed or len(ring) < 3:
            return None, []  # malformed cycle -- fall back rather than
                              # emit a boundary that isn't a clean loop
        rings.append(ring + [ring[0]])

    rings.sort(key=lambda r: _ring_signed_area_2d(pts, r), reverse=True)
    exterior = rings[0]
    holes = []
    if len(rings) > 1:
        try:
            from shapely.geometry import Polygon as _Poly, Point as _Point
            ext_poly = _Poly(pts[exterior]).buffer(0)
            for r in rings[1:]:
                centroid_pt = _Point(pts[r[:-1]].mean(axis=0))
                if ext_poly.contains(centroid_pt):
                    holes.append(r)
        except Exception:
            holes = []  # hole containment is best-effort; exterior stands either way

    return exterior, holes


def _oriented_local_frame(points_3d):
    """
    `_fit_local_frame` with the two remaining degrees of freedom pinned
    down, so a boundary traced in this frame is deterministic.

    SVD returns an arbitrary sign/handedness for (u, v, normal), which
    means the same point cloud could trace its ring clockwise on one run
    and counter-clockwise on another. Here the normal is flipped to point
    upward (non-negative Z -- a gravity convention about which side of its
    own surface is "up", not a plan-view assumption), and v is rebuilt as
    normal x u so (u, v, normal) is right-handed. A ring traced
    counter-clockwise in (u, v) then has a consistent, repeatable
    orientation with respect to the patch's own outward side, whatever the
    patch's tilt.
    """
    centroid, u_dir, v_dir, normal = _fit_local_frame(points_3d)
    if normal[2] < 0:
        normal = -normal
    v_dir = np.cross(normal, u_dir)
    nv = np.linalg.norm(v_dir)
    if nv < 1e-12:
        # u happened to be parallel to the normal (a degenerate cloud) --
        # keep the raw SVD axes rather than manufacture a frame.
        return _fit_local_frame(points_3d)
    v_dir = v_dir / nv
    return centroid, u_dir, v_dir, normal


def _signed_ring_area_2d(pts_2d, ring_idx):
    """Signed shoelace area of a closed ring of indices (>0 = CCW)."""
    r = np.asarray(pts_2d, dtype=float)[ring_idx]
    x, y = r[:, 0], r[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def _measured_section_from_points(points_xyz, smooth=False, validation_poly=None):
    """
    Build a SectionProfile's boundary from a REAL measured (x, y, z) point
    cloud -- the ONLY way a floor or roof section gets built in this file.

    The perimeter comes ENTIRELY from `points_xyz`'s own measured extent
    in its own local surface frame, via alpha-shape / concave-hull extraction
    (`_alpha_shape_rings`; if that tracing fails or is non-manifold there is
    NO fallback -- the function returns None): the boundary tightens around exactly the points
    that were actually measured, so an irregular edge, a partial slab's
    ragged extent, a setback, or a genuine overhang's silhouette all come
    through as the real concave shape they are -- never smoothed, squared
    off, or stretched to match a drafted outline. Where the surviving
    alpha-complex has more than one ring, an interior ring is kept as a
    real measured hole (an atrium, a light-well, or simply a gap in this
    specific slab's own coverage) -- not a blueprint-declared opening.

    NO POLYGON EVER CLIPS `points_xyz` (3D-FIX ROUND 4). `validation_poly`
    -- the unit's blueprint footprint, or a volume's own annotated
    `region` -- is consulted only AFTER the boundary is final, to log how
    the two compare. It cannot add, move, or delete a vertex, and it
    cannot cause this function to return None. Every measured point is
    eligible to become a ring vertex and every measured point is carried
    into `surface_points_3d`, so a cantilever, jetty, bay, balcony, eave,
    or any slab edge that genuinely oversails the drawing keeps the
    material that proves it.

    This is a deliberate reversal of the previous behaviour, which first
    spatially clipped `points_xyz` to `poly` and returned None when fewer
    than 3 points survived. That clip existed to stop a partial or
    mezzanine volume from inheriting a full slab's outline, because
    `_pair_floor_volumes` used to hand every volume the WHOLE cloud of the
    surface it was built from. It solved that at the cost of intersecting
    measured reality with a drafted outline -- points outside the drawing
    were deleted, so the "measured" section was really
    reality AND blueprint, and an overhang was silently amputated. The
    cause is fixed instead of the symptom: `_pair_floor_volumes` now
    resolves floor/ceiling topology by casting rays against real B-Rep
    surface patches and hands each volume the measured SUBSET of points
    that genuinely bounds it (see its docstring). A partial slab therefore
    arrives here already carrying only its own evidence, with no polygon
    involved on either side.

    The boundary is traced in the cloud's OWN local (u, v) frame
    (`_oriented_local_frame`, PCA on the real 3D points), not in world
    (x, y). A steeply pitched deck, a dormer cheek, a vaulted soffit, or
    an eave that curls back under itself is single-valued and
    well-conditioned in its own frame while being compressed or outright
    multi-valued in plan -- tracing in plan would collapse or self-
    intersect exactly the curved and folded surfaces this pipeline exists
    to preserve. Ring vertices are still the measured 3D points
    themselves; only the parametrization used to order them changed.

    This also still replaces two removed functions, for the same reasons
    as before:
      - `_flat_section_from_scalar(poly, z_scalar)`: stamped the ENTIRE
        boundary ring at one constant elevation.
      - `_fitted_section_from_model(poly, level, model, features)`: lifted
        the boundary onto a z = model(x, y) REGRESSION surface -- still a
        single Z per (x, y), and prone to silent extrapolation beyond the
        cloud that trained it.
    Neither a flat scalar nor a regression height field appears anywhere
    in this path: every ring vertex is an actual measured point, and the
    cap it later gets (in `_make_trimmed_face`, via `_measured_cap_shape`)
    is a triangulation of real measured points too.

    Returns None -- never a flat guess, never a blueprint-shaped
    substitute, never a convex hull -- only when there isn't enough real
    evidence to derive a boundary from: `points_xyz` is None, has fewer than
    3 finite points, is degenerate in its own local frame (e.g. collinear),
    its alpha-shape boundary cannot be traced (non-manifold), or the cloud is
    multi-sheet -- its single local (u, v) projection is ambiguous (folded /
    undercut / stacked sheets) or it splits into disconnected 3D sheets --
    since one boundary traced there would connect distinct 3D regions. A
    disagreement with `validation_poly`, however large, is NEVER one of
    those cases. Callers must treat None as "no section for this floor"
    and skip it, logged, exactly like any other missing-evidence path in
    this pipeline.
    """
    if points_xyz is None:
        return None
    points_xyz = np.asarray(points_xyz, dtype=float)
    if points_xyz.ndim != 2 or points_xyz.shape[1] != 3:
        return None
    points_xyz = points_xyz[np.isfinite(points_xyz).all(axis=1)]
    if points_xyz.shape[0] < 3:
        return None

    # The cloud's own PCA frame -- see the docstring. Nothing is projected
    # onto world (x, y), and nothing is discarded.
    centroid, u_dir, v_dir, _normal = _oriented_local_frame(points_xyz)
    local = points_xyz - centroid
    uv = np.column_stack((local @ u_dir, local @ v_dir))

    # A boundary traced in ONE local (u, v) projection is only meaningful for
    # a single connected, single-valued sheet. Multi-sheet evidence (a fold or
    # undercut projecting onto itself, or physically disconnected sheets)
    # yields no section rather than a boundary joining distinct 3D regions.
    if _projection_is_ambiguous(points_xyz, uv, local @ _normal):
        return None
    from scipy.spatial import cKDTree
    _sp3 = max(float(np.median(cKDTree(points_xyz).query(points_xyz, k=2)[0][:, 1])), 1e-6)
    _min_sheet = max(10, int(0.02 * points_xyz.shape[0]))
    if sum(1 for c in _connected_components_3d(points_xyz, 4.0 * _sp3) if len(c) >= _min_sheet) > 1:
        return None

    exterior_idx, hole_idxs = _alpha_shape_rings(uv)
    if exterior_idx is None:
        # Alpha-shape tracing failed (too few points for a meaningful concave
        # hull, or a non-manifold boundary). A convex hull would invent edges
        # across unmeasured gaps, so no boundary is fabricated: no section.
        return None

    # Deterministic orientation: exterior counter-clockwise in the patch's
    # own right-handed (u, v) frame, interior rings clockwise. This is a
    # re-ordering of measured vertices only -- no vertex is added, moved,
    # or dropped.
    if _signed_ring_area_2d(uv, exterior_idx) < 0:
        exterior_idx = list(exterior_idx)[::-1]
    hole_idxs = [list(h)[::-1] if _signed_ring_area_2d(uv, h) > 0 else list(h)
                 for h in hole_idxs]

    ring = [tuple(points_xyz[i]) for i in exterior_idx]
    holes = [[tuple(points_xyz[i]) for i in hidx] for hidx in hole_idxs]
    z_seed = float(np.median(points_xyz[:, 2]))

    # ------------------------------------------------------------------
    # VALIDATION / ANNOTATION ONLY -- runs after `ring` and `holes` are
    # already final, reads them, and writes nothing back. `validation_poly`
    # cannot clip, reshape, or veto the measured boundary; the worst it can
    # do is print a line for a human to look at. Measured coverage BEYOND
    # the polygon is reported as a preserved overhang, not as an error.
    # ------------------------------------------------------------------
    if validation_poly is not None and not validation_poly.is_empty:
        try:
            measured_poly = _polygonal_only(
                Polygon(points_xyz[exterior_idx][:, :2]).buffer(0))
            if not measured_poly.is_empty:
                inter = measured_poly.intersection(validation_poly).area
                union = measured_poly.union(validation_poly).area
                iou = (inter / union) if union > 0 else 0.0
                beyond = _polygonal_only(measured_poly.difference(validation_poly)).area
                if iou < 0.5:
                    print(f"   ℹ️ measured slab/roof boundary (from its own point "
                          f"cloud, {points_xyz.shape[0]} pts) overlaps only "
                          f"{iou:.0%} with the blueprint/plan-extent outline at "
                          f"this level -- the measured shape is authoritative and "
                          f"has been kept in full; this is a flag for review, not "
                          f"a correction.")
                if beyond > 0:
                    print(f"   ℹ️ {beyond:.1f} m2 of this section lies OUTSIDE the "
                          f"blueprint/plan-extent outline (overhang, cantilever, "
                          f"balcony, or eave) -- retained as measured, not clipped.")
        except Exception:
            pass  # validation is best-effort; a geometry error never blocks it

    return SectionProfile(
        level=z_seed, points_3d=ring, holes_3d=holes, smooth=smooth,
        surface_points_3d=[(float(x), float(y), float(z)) for x, y, z in points_xyz],
    )


def _transform_wall_section_to_global(gnss, section):
    """
    Turn one AI/blueprint-extracted intermediate wall-section dict --
    {"level": <float>, "ring_xy": [(x, y), ...], "z": <scalar or a
    per-vertex list matching ring_xy>, "holes": [...], "smooth": <bool>},
    all in the unit's own LOCAL blueprint pixel frame -- into a georeferenced
    SectionProfile that is REFERENCE ONLY.

    This is the actual overhang/jetty/corbel/bay-window/taper evidence:
    an AI extractor that emits one or more of these per unit is what lets
    construct_unit_volume() loft a wall that bulges out and back in,
    instead of the straight two-section (floor-to-roof) loft that is all
    that's possible without it.

    The AI/blueprint `z` is NOT carried into the SectionProfile: it is neither
    measured nor evidence, so storing it in `points_3d` could let it be
    mistaken for measured XYZ. Only the transformed XY (the correspondence
    position) is kept; the third coordinate is NaN, meaning "unmeasured".
    _measured_wall_sections matches by XY alone and replaces every vertex with
    an actual measured XYZ point; a section that was not replaced fails loudly
    (NaN) instead of silently acting as geometry.
    """
    def _ring_to_global(ring_xy):
        return [(*gnss.transform_xy(x, y), float("nan")) for x, y in ring_xy]

    ring_xyz = _ring_to_global(section["ring_xy"])
    holes_xyz = [_ring_to_global(hole["ring_xy"]) for hole in section.get("holes", [])]
    return SectionProfile(
        level=section["level"], points_3d=ring_xyz, holes_3d=holes_xyz,
        smooth=bool(section.get("smooth", False)),
    )

# Wall sections come from the AI/blueprint extractor (traced rings with
# asserted Z), not from measurement. They may only become B-Rep geometry when
# measured XYZ backs them: at least this fraction of a section's vertices must
# have a measured point, on ONE measured wall surface patch, within this
# distance IN XY (the section's transformed reference position; AI/blueprint Z
# never selects measured points). Both are tolerances for verifying evidence,
# not geometry parameters.
WALL_SECTION_SUPPORT_TOL_M = 0.5
WALL_SECTION_MIN_SUPPORTED_FRACTION = 0.8
# An unlabelled patch counts as lateral wall evidence for a volume only when
# its normal is transverse to the measured bottom->ceiling segment there:
# |cos(normal, segment)| <= this (i.e. the surface runs along the segment).
WALL_SECTION_MAX_CLEARANCE_COS = 0.5
# With an authoritative UP direction, a probe whose normal is within ~10 deg
# of perpendicular to it (|cos| below this) has no reliable side and stays open.
ORIENTATION_MIN_ABS_DOT = 0.17


def _section_vertices(section):
    verts = list(section.points_3d)
    for ring in section.holes_3d:
        verts.extend(ring)
    return np.asarray(verts, dtype=float)


def _measured_wall_sections(sections, measured_xyz, semantic_labels=None):
    """
    Keep only the wall sections whose every-vertex evidence is MEASURED, on
    the wall's OWN measured surface. Proximity to a measured point is not
    evidence of that: the nearest point may belong to a slab, roof, balcony,
    eave or another sheet. So the measured cloud is first segmented, with the
    same primitives and parameters as _detect_slab_surfaces (local
    dimensionality -> smooth patches by 3D adjacency + normal continuity),
    and a section may be supported by a patch that `semantic_labels`
    (optional per-point AI/external labels aligned with `measured_xyz`) marks
    wall/facade, or by an UNLABELLED measured patch of any orientation.
    Patches labelled slab/roof/floor never can. Proximity alone is not wall
    evidence: a section carried by an unlabelled patch is tagged with that
    patch's normals (`lateral_normals`) and is only usable by a volume whose
    measured bottom->ceiling segments run along the patch (see
    _wall_sections_for_volume); otherwise it is rejected. Support also
    requires the measured vertex/edge evidence below.
    A section is kept only when >= WALL_SECTION_MIN_SUPPORTED_FRACTION of its
    vertices (rim and holes) lie within WALL_SECTION_SUPPORT_TOL_M, measured in
    XY only, of points of ONE such patch -- the patch most of its vertices are
    nearest to. AI/blueprint Z NEVER influences which measured point is
    chosen: the section is 2D/reference correspondence, matched by its
    transformed XY position, and the vertex that results is the actual
    measured XYZ point (its own measured Z). On a near-vertical patch several
    measured heights share one XY, so the XY-nearest measured point is
    taken; edge/connectivity checks below then validate it in full 3D. Otherwise wall-surface evidence is not established and the section is
    rejected; it is never snapped to arbitrary measured points.
    The AI/blueprint section is a reference CORRESPONDENCE only: every kept
    vertex is REPLACED by its XY-nearest actual measured XYZ point ON THAT
    PATCH, so no AI-supplied X, Y or Z survives as geometry. Vertex order
    is preserved; a vertex with no such point in tolerance is dropped
    (never kept at its AI position), as is a consecutive repeat when two
    vertices land on the same measured point. A ring left with < 3 vertices is
    dropped (a hole ring alone, or the whole section if it is the rim).
    Nearest-point support does not prove the ring's EDGES: after snapping,
    every consecutive vertex pair of every ring (closing edge included, and
    an edge bridging a dropped vertex) must be joined by measured points of
    that same wall patch. (1) The edge line itself must lie on measured
    points: samples along the straight edge, at half the patch's link radius,
    each need a point of that patch within that distance, so a gap on the
    edge is never spanned. (2) A chain of hops that are spatially adjacent
    with continuous normals (the patch's own segmentation graph) must join the
    two ends within a link-radius corridor around the edge, so the edge cannot
    connect distinct sheets. An edge failing either test is not invented: the
    whole section is rejected.
    Unsupported sections are dropped whole. Returns (kept, dropped_count).
    """
    if not sections:
        return [], 0
    if measured_xyz is None or len(measured_xyz) == 0:
        return [], len(sections)
    from scipy.spatial import cKDTree
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components
    pts = np.asarray(measured_xyz, dtype=float)
    labels = None
    if semantic_labels is not None:
        labels = np.asarray(semantic_labels, dtype=object)
        if labels.shape != (pts.shape[0],):
            raise ValueError("semantic_labels must have one entry per measured point")
    finite = np.isfinite(pts).all(axis=1)
    pts = pts[finite]
    if labels is not None:
        labels = labels[finite]

    # Wall-capable measured surface patches (parameters mirror
    # _detect_slab_surfaces: k=12, 18 deg normal continuity, radius ~4x the
    # cloud's own spacing, 25-point floor; role never from orientation).
    normals, a1d, a2d, a3d, nn_spacing = _neighbourhood_geometry(pts, 12)
    surface_like = (a2d >= a1d) & (a2d >= a3d)
    cand, cand_n = pts[surface_like], normals[surface_like]
    cand_labels = labels[surface_like] if labels is not None else None
    if cand.shape[0] < 25:
        return [], len(sections)
    link_r = max(4.0 * nn_spacing, 1e-3)
    wall_patches, wall_normals, wall_labelled = [], [], []
    for comp in _smooth_patch_components(cand, cand_n, link_r, 18.0):
        if comp.shape[0] < 25:
            continue
        role = _patch_semantic_role(cand_labels[comp]) if cand_labels is not None else None
        if role in ("slab", "roof"):
            continue  # authoritatively slab/roof/floor: cannot evidence a wall
        wall_patches.append(cand[comp])
        wall_normals.append(cand_n[comp])
        wall_labelled.append(role == "wall")
    if not wall_patches:
        return [], len(sections)
    wall_pts = np.vstack(wall_patches)
    wall_patch_id = np.concatenate(
        [np.full(len(p), k) for k, p in enumerate(wall_patches)])
    wall_tree_xy = cKDTree(wall_pts[:, :2])  # XY correspondence; AI Z never queried

    graphs = {}

    def _patch_graph(k):
        # The patch's own segmentation graph: spatially adjacent points with
        # continuous normals.
        if k not in graphs:
            pp, nn = wall_patches[k], wall_normals[k]
            pairs = cKDTree(pp).query_pairs(r=link_r, output_type="ndarray")
            if pairs.size:
                cosb = np.abs(np.einsum("ij,ij->i", nn[pairs[:, 0]], nn[pairs[:, 1]]))
                pairs = pairs[cosb >= np.cos(np.radians(18.0)) - 1e-12]
            if pairs.size:
                graphs[k] = csr_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])),
                                       shape=(len(pp), len(pp)))
            else:
                graphs[k] = csr_matrix((len(pp), len(pp)))
        return graphs[k]

    def _ring_connected(ring_idx, k, tree):
        # Every consecutive pair (closing edge included) must be measured
        # along the edge itself AND joined by connected measured points of
        # this patch, not by a chord across a gap.
        pp, g = wall_patches[k], _patch_graph(k)
        for a, b in zip(ring_idx, ring_idx[1:] + ring_idx[:1]):
            seg = pp[b] - pp[a]
            n_s = max(int(np.ceil(float(np.linalg.norm(seg)) / (0.5 * link_r))), 1)
            on_edge = pp[a] + np.linspace(0.0, 1.0, n_s + 1)[:, None] * seg
            if (tree.query(on_edge, k=1)[0] > 0.5 * link_r).any():
                return False
            t = np.clip((pp - pp[a]) @ seg / max(float(seg @ seg), 1e-24), 0.0, 1.0)
            near_edge = np.linalg.norm(pp - (pp[a] + t[:, None] * seg), axis=1) <= link_r
            _, lab = connected_components(g[near_edge][:, near_edge], directed=False)
            pos = np.cumsum(near_edge) - 1
            if lab[pos[a]] != lab[pos[b]]:
                return False
        return True

    def _snap(ring, tree_xy):
        if len(ring) == 0:
            return None
        # XY correspondence only: the ring's AI/blueprint Z is not used.
        d, idx = tree_xy.query(np.asarray(ring, dtype=float)[:, :2], k=1)
        out = []
        for dist, j in zip(d, idx):
            if dist <= WALL_SECTION_SUPPORT_TOL_M and (not out or int(j) != out[-1]):
                out.append(int(j))
        if len(out) > 1 and out[0] == out[-1]:
            out.pop()
        return out if len(out) >= 3 else None

    kept = []
    for sec in sections:
        verts = _section_vertices(sec)
        d, j = wall_tree_xy.query(verts[:, :2], k=1)
        near = d <= WALL_SECTION_SUPPORT_TOL_M
        if not near.any():
            continue
        # The ONE wall patch this section must be supported by.
        pid = int(np.bincount(wall_patch_id[j[near]]).argmax())
        patch = wall_patches[pid]
        tree = cKDTree(patch)  # measured 3D, for the edge tests only
        tree_xy = cKDTree(patch[:, :2])
        d = tree_xy.query(verts[:, :2], k=1)[0]
        if float(np.mean(d <= WALL_SECTION_SUPPORT_TOL_M)) < WALL_SECTION_MIN_SUPPORTED_FRACTION:
            continue
        rim = _snap(sec.points_3d, tree_xy)
        if rim is None:
            continue
        holes = [h for h in (_snap(r, tree_xy) for r in sec.holes_3d) if h is not None]
        if not all(_ring_connected(r, pid, tree) for r in [rim] + holes):
            continue  # an edge would cross a measured gap: not invented
        xyz = lambda ring: [tuple(float(c) for c in patch[j]) for j in ring]
        new_sec = SectionProfile(level=sec.level, points_3d=xyz(rim),
                                 holes_3d=[xyz(h) for h in holes], smooth=sec.smooth)
        if not wall_labelled[pid]:
            pn = wall_normals[pid]
            new_sec.lateral_normals = np.array([pn[j] for j in rim]
                                               + [pn[j] for h in holes for j in h])
        kept.append(new_sec)
    return kept, len(sections) - len(kept)


def _wall_sections_for_volume(sections, bottom_points, ceiling_points):
    """
    Pick the (already measured-supported) wall sections that sit BETWEEN this
    volume's own measured bottom and ceiling evidence -- in 3D, not by a
    scalar Z range. For each section vertex, the nearest measured bottom
    point b and nearest measured ceiling point c (3D nearest neighbours)
    define a local segment; the vertex is bracketed when its projection onto
    b->c falls strictly inside that segment. A folded, tilted, vaulted or
    undercut bottom/ceiling therefore brackets locally, against the surface
    that is actually there, instead of against one median level. A section
    is selected when >= WALL_SECTION_MIN_SUPPORTED_FRACTION of its vertices
    are bracketed, and the selected sections are ordered for the loft by
    their mean position t along those measured b->c segments. A section
    supported only by an unlabelled patch must in addition have that patch's
    normals transverse to the b->c segments (see WALL_SECTION_MAX_CLEARANCE_COS);
    no lateral evidence, no wall: the volume is rejected by the caller.
    """
    if not sections or bottom_points is None or ceiling_points is None:
        return []
    bot = np.asarray(bottom_points, dtype=float)
    top = np.asarray(ceiling_points, dtype=float)
    if len(bot) == 0 or len(top) == 0:
        return []
    from scipy.spatial import cKDTree
    bot_tree, top_tree = cKDTree(bot), cKDTree(top)
    picked = []
    for sec in sections:
        verts = _section_vertices(sec)
        b = bot[bot_tree.query(verts, k=1)[1]]
        c = top[top_tree.query(verts, k=1)[1]]
        seg = c - b
        len2 = np.einsum("ij,ij->i", seg, seg)
        valid = len2 > 1e-12
        t = np.einsum("ij,ij->i", verts - b, seg) / np.where(valid, len2, 1.0)
        inside = valid & (t > 0.0) & (t < 1.0)
        ln = sec.lateral_normals
        if ln is not None:
            # Unlabelled support: the measured patch must run ALONG the
            # bottom->ceiling segment (transverse normal), else it is a
            # slab/roof/balcony sheet and is not lateral wall evidence.
            unit_seg = seg / np.sqrt(np.where(valid, len2, 1.0))[:, None]
            along = np.abs(np.einsum("ij,ij->i", ln, unit_seg)) <= WALL_SECTION_MAX_CLEARANCE_COS
            inside = inside & along
        if float(np.mean(inside)) >= WALL_SECTION_MIN_SUPPORTED_FRACTION:
            picked.append((float(np.mean(t[inside])), sec))
    picked.sort(key=lambda p: p[0])
    return [sec for _, sec in picked]


def _manual_override_storey_metadata(floors, basements, floor_h):
    """
    Pure METADATA replacement for blueprint_metrology.build_uniform_storeys().

    A surveyor override (explicit floors=/floor_h=) still needs to produce
    (floor_level, tier, base_offset, height) tuples so downstream floor
    labeling/registration works the same as the derived-evidence path --
    but the result of this function is NEVER passed to a prism, a cutting
    plane, or any other shape-generating/shape-trimming call anywhere in
    this file.

    `base_offset` and `height` are carried for logging/audit purposes only
    (they describe what the surveyor asserted). Per-floor VOLUMES are now
    built entirely from each floor's own measured evidence -- see
    _detect_slab_surfaces(), _measured_section_from_points() and the
    per-unit loop in run_unified_cadastre_pipeline, which segments each
    unit's own full measured point cloud into slab surfaces directly, with
    no reference to floor_level, floor count or floor_h at all. A floor_h
    override changes the metadata label a volume is registered under (via
    tier/floor-count reporting only); it can never change the volume's
    shape, and there is no code path left in this file that would let it.
    """
    class _Storey:
        __slots__ = ("floor_level", "tier", "base_offset", "height")
        def __init__(self, floor_level, tier, base_offset, height):
            self.floor_level = floor_level
            self.tier = tier
            self.base_offset = base_offset
            self.height = height

    storeys = []
    for b in range(basements, 0, -1):
        storeys.append(_Storey(-b, "BASEMENT", -b * floor_h, floor_h))
    for f in range(floors):
        storeys.append(_Storey(f, "ABOVE_GROUND", f * floor_h, floor_h))
    return storeys


def _neighbourhood_geometry(points_xyz, k_neighbors=12):
    """
    Per-point local 3D neighbourhood geometry: a normal direction and a
    scale-free DIMENSIONALITY description of what the neighbourhood is
    shaped like. Computed once, from the measured points alone, and reused
    by every stage of `_detect_slab_surfaces`.

    From the eigenvalues (L0 <= L1 <= L2) of each point's own k-NN
    covariance, the standard Demantke dimensionality features are formed
    from their square roots (s_i = sqrt(L_i)):
        a1D = (s2 - s1) / s2   -- neighbourhood spread along ONE direction
                                  (a wire, a railing, a branch, an edge)
        a2D = (s1 - s0) / s2   -- spread across TWO directions and thin in
                                  the third: a SURFACE, of any shape
        a3D =  s0 / s2         -- spread in all three: volumetric scatter
                                  (vegetation canopy, scan noise, clutter)
    These are ratios, so they carry no length scale and no orientation.
    Nothing here measures flatness: a2D is the dominant feature for a
    strongly curved, vaulted, or folded surface just as much as for a flat
    one, because curvature at the scale of a dozen neighbouring points
    perturbs s0 by far less than the surface's own extent perturbs s1 and
    s2. A test on a2D therefore asks "are these points lying ON something
    two-dimensional?", which is a manifold question, NOT "do these points
    lie near a plane?", which is the planar assumption this pass removes.

    `normal` is the eigenvector of L0 -- the least-spread direction, i.e.
    the local surface perpendicular. Its SIGN is meaningless (PCA cannot
    orient a surface), so every comparison downstream uses the unsigned
    angle between normals and never their dot-product sign. That is
    deliberate: a surface that folds far enough to face the other way
    (an eave curling back under, the far side of a vault) would be torn in
    two by any orientation-propagating scheme.

    Returns (normals (N, 3), a1D, a2D, a3D, nn_spacing) -- the last being
    the cloud's own median nearest-neighbour distance, which every
    length-like tolerance in this file is expressed as a multiple of.
    """
    from scipy.spatial import cKDTree

    pts = np.asarray(points_xyz, dtype=float)
    n = pts.shape[0]
    if n < 4:
        return (np.tile(np.array([0.0, 0.0, 1.0]), (n, 1)),
                np.zeros(n), np.zeros(n), np.ones(n), 1e-6)

    k = min(k_neighbors, n - 1)
    nn_dists, nn_idx = cKDTree(pts).query(pts, k=k + 1)  # column 0 is the point
    nn_spacing = max(float(np.median(nn_dists[:, 1])), 1e-6)

    nb = pts[nn_idx]                                     # (n, k+1, 3)
    centered = nb - nb.mean(axis=1, keepdims=True)
    cov = np.einsum("nki,nkj->nij", centered, centered) / max(k, 1)
    evals, evecs = np.linalg.eigh(cov)                   # ascending
    evals = np.clip(evals, 0.0, None)
    normals = evecs[:, :, 0]

    s = np.sqrt(evals)                                   # s[:,0] <= s[:,1] <= s[:,2]
    s2 = np.where(s[:, 2] > 1e-12, s[:, 2], 1e-12)
    a1d = (s[:, 2] - s[:, 1]) / s2
    a2d = (s[:, 1] - s[:, 0]) / s2
    a3d = s[:, 0] / s2

    # A neighbourhood with no spread at all (coincident returns) describes
    # nothing; call it scatter so it is dropped rather than trusted.
    dead = s[:, 2] <= 1e-12
    a1d[dead], a2d[dead], a3d[dead] = 0.0, 0.0, 1.0
    normals[dead] = np.array([0.0, 0.0, 1.0])

    lens = np.linalg.norm(normals, axis=1)
    normals = normals / np.where(lens[:, None] > 1e-12, lens[:, None], 1.0)
    return normals, a1d, a2d, a3d, nn_spacing


def _smooth_patch_components(points_xyz, normals, radius, max_normal_deg):
    """
    Segment a point set into SMOOTH SURFACE PATCHES: connected components of
    a graph that links two points when they are both spatially adjacent
    (within `radius`) and their local surface normals agree to within
    `max_normal_deg`.

    This is the segmentation primitive that replaces plane fitting. Its
    criterion is *differential*, not global -- each edge asks only whether
    the surface bends sharply BETWEEN two neighbouring measurements. A
    vault, a dome, a barrel roof, a warped or twisted slab, a folded eave,
    or a ramp that curves as it climbs all turn gradually: every individual
    step between adjacent points is small even though the patch's total
    turning is large, so the whole thing stays one component no matter how
    far it departs from any plane. A plane-consensus test asks the opposite,
    global question ("does ONE plane explain all of it?") and necessarily
    shatters exactly those shapes into shards.

    Normal agreement uses the UNSIGNED angle, arccos(|n_i . n_j|), because
    PCA normals have arbitrary sign. A consequence worth stating: a surface
    that folds through 90 degrees and keeps going stays connected across the
    fold, which is the intended behaviour for an eave or a soffit return.

    What still separates patches is a genuine CREASE: where a slab meets a
    wall, or one roof pitch meets another at a ridge, the normal turns by
    tens of degrees within a point spacing or two, the edges there fail the
    angle test, and the two surfaces come out as the distinct surfaces they
    are. `radius` alone would have fused them, since they touch in space.

    Returns a list of index arrays (into `points_xyz`), largest first.
    """
    from scipy.spatial import cKDTree
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    pts = np.asarray(points_xyz, dtype=float)
    m = pts.shape[0]
    if m == 0:
        return []
    if m == 1:
        return [np.array([0])]

    pairs = cKDTree(pts).query_pairs(r=radius, output_type="ndarray")
    if pairs.size:
        # Unsigned normal agreement: |cos| >= cos(threshold).
        cos_between = np.abs(np.einsum("ij,ij->i",
                                       normals[pairs[:, 0]], normals[pairs[:, 1]]))
        smooth = cos_between >= np.cos(np.radians(max_normal_deg)) - 1e-12
        pairs = pairs[smooth]

    if pairs.size:
        rows = np.concatenate([pairs[:, 0], pairs[:, 1]])
        cols = np.concatenate([pairs[:, 1], pairs[:, 0]])
        graph = coo_matrix((np.ones(rows.shape[0]), (rows, cols)), shape=(m, m))
    else:
        graph = coo_matrix((m, m))
    n_comp, labels = connected_components(graph, directed=False)
    comps = [np.flatnonzero(labels == c) for c in range(n_comp)]
    comps.sort(key=len, reverse=True)
    return comps


_SLAB_SEMANTIC_LABELS = ("slab", "floor", "ceiling", "deck")
_ROOF_SEMANTIC_LABELS = ("roof",)
_WALL_SEMANTIC_LABELS = ("wall", "facade", "façade")


def _patch_semantic_role(patch_labels):
    """
    "slab" / "roof" / "wall" from the AUTHORITATIVE per-point semantic labels
    of one patch (explicit source classification), or None when the patch is
    not labelled clearly enough (fewer than half its points carry a
    slab/roof/wall label, or the top two tie). "roof" is its own role, never
    folded into "slab". None means "generic measured surface candidate":
    orientation is never used to assign a role.
    """
    if patch_labels is None or len(patch_labels) == 0:
        return None
    lab = np.array([str(l).strip().lower() if l is not None else "" for l in patch_labels])
    counts = {"slab": int(np.isin(lab, _SLAB_SEMANTIC_LABELS).sum()),
              "roof": int(np.isin(lab, _ROOF_SEMANTIC_LABELS).sum()),
              "wall": int(np.isin(lab, _WALL_SEMANTIC_LABELS).sum())}
    ranked = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
    if sum(counts.values()) < 0.5 * len(lab) or ranked[0][1] == ranked[1][1]:
        return None
    return ranked[0][0]


def _detect_slab_surfaces(points_xyz, k_neighbors=12,
                          scatter_margin=0.0, min_points=25,
                          max_normal_deg=18.0, connectivity_radius_m=None,
                          semantic_labels=None, roles_out=None):
    """
    True 3D surface segmentation for ONE unit's full measured (x, y, z)
    point cloud: find the actual connected structural surfaces it contains,
    whatever shape they are. (Legacy name: it returns ALL measured candidate
    surfaces, including role=None unlabelled ones; only `roles_out[label] ==
    "slab"` marks an authoritatively slab-labelled surface.) No declared storeys, no floor count, no
    floor_h, no Z banding, no XY projection -- and, as of this pass, NO
    PLANE FITTING ANYWHERE.

    WHAT THIS REPLACES. The previous version ended in a RANSAC loop: within
    each spatially connected blob it repeatedly fit a plane, peeled off the
    points lying within a residual of it as "one detected surface", and
    re-fit on the remainder. Two assumptions were buried in that:
      - that a structural surface IS a plane, so anything that is not one
        gets carved into the planes that best approximate it. A barrel
        vault came out as a staircase of narrow planar strips; a dome as a
        rosette of facets; a warped or twisted slab as several overlapping
        shards; a folded eave as two unrelated pieces. Each shard then
        entered the topology stage as a separate "surface", so a single
        room could be roofed by a dozen fictitious ceilings.
      - that a surface no plane explains is not a surface at all. The loop
        broke out when no fit succeeded and those points were discarded --
        so the MORE curved the evidence, the more of it was thrown away.
    A per-point planarity threshold and a per-point tilt filter compounded
    it: points were dropped BEFORE grouping for being locally non-flat or
    too steep, which truncated a vault at its springing and clipped the
    curved lip off an eave or a bullnose edge.

    THE RULE NOW -- neighbourhood geometry, then normal continuity, then
    connectivity. Nothing is fit to anything:

      1. NEIGHBOURHOOD DIMENSIONALITY (`_neighbourhood_geometry`). Each
         point's own k-NN covariance gives a local normal and the scale-free
         (a1D, a2D, a3D) dimensionality features. A point is kept when a2D
         is its largest feature -- when its neighbourhood is shaped like a
         SURFACE rather than like a line (a railing, a cable, a branch) or a
         volume (vegetation, scan noise, clutter). This is a manifold test,
         not a flatness test: a2D dominates for a vault, a fold, and a flat
         floor alike, so curvature costs a point nothing. `scatter_margin`
         tightens it (a2D must beat a3D by that margin) for noisy surveys;
         at 0 it is a plain argmax. Nothing about elevation, tilt, or
         orientation enters here.

      2. SMOOTH-PATCH SEGMENTATION (`_smooth_patch_components`). The
         surviving points are linked when they are spatially adjacent AND
         their normals agree within `max_normal_deg`, and the connected
         components of that graph are the surface patches. Because the test
         is between NEIGHBOURS, total curvature is unbounded: a patch may
         turn through any angle as long as it turns gradually, so vaulted,
         domed, warped, folded, and irregular surfaces survive whole. A
         sharp crease -- slab-to-wall, or a roof ridge -- does break the
         link, which is correct: those are two surfaces.

      3. PATCH ROLE FROM SEMANTICS, NEVER FROM ORIENTATION. Every measured
         surface patch is kept as a generic candidate whatever its
         orientation (steep, near-vertical, curved, folded and eccentric
         roofs included). The only role filter is `semantic_labels`, an
         optional per-point array (aligned with `points_xyz`) of
         authoritative AI/external labels: a patch whose points are
         predominantly labelled wall/facade or roof is not a slab surface
         (roof arrives separately via roof_points_xyz). Slab/floor/ceiling/
         deck-labelled and unlabelled patches are kept whole as
         measured surfaces. Keeping a patch is NOT a claim that it is a
         floor/ceiling: if `roles_out` (a dict) is given it receives
         {label: "slab" | None}, "slab" only for authoritative slab/floor/
         ceiling/deck semantics, None for unlabelled/ambiguous. Only
         "slab" surfaces may act as floor/ceiling in _pair_floor_volumes.

      4. SIZE FLOOR. A patch with fewer than `min_points` points is dropped
         as too small to be a real structural surface.

    An accepted patch is returned as its ORIGINAL measured (x, y, z) points
    -- not resampled, not projected, not replaced by a fitted surface's
    idealised points, and not expressed as any height field h = f(u, v).
    The patch is a set of real 3D measurements and nothing else, which is
    what `_measured_surface_brep` needs in order to tessellate it with
    local charts and what the ray-casting topology in `_pair_floor_volumes`
    needs in order to find first hits on its real shape.

    `connectivity_radius_m` defaults to ~4x the cloud's own median
    nearest-neighbour spacing -- comfortably above the percolation
    threshold, so a real surface's points form one component, while still
    scaling with THIS survey's density rather than a fixed distance.

    There is no RNG here any more. RANSAC sampled planes at random, so the
    same cloud could segment differently between runs and a cadastral
    record was not reproducible from its own evidence; every step above is
    deterministic, and `random_seed` is gone rather than left as a dead
    parameter implying otherwise.

    Returns {label: (N, 3) points}, one entry per detected surface, keyed by
    that surface's own median Z (metres, rounded to mm). The label is a
    data-derived handle used only for logging, sorting, and tiering -- never
    a floor index, never required to be an integer or evenly spaced, and
    never read by anything that builds geometry.
    """
    if points_xyz is None:
        return {}
    pts = np.asarray(points_xyz, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 3:
        return {}
    labels = None
    if semantic_labels is not None:
        labels = np.asarray(semantic_labels, dtype=object)
        if labels.shape != (pts.shape[0],):
            raise ValueError("semantic_labels must have one entry per input point")
    finite = np.isfinite(pts).all(axis=1)
    pts = pts[finite]
    n = pts.shape[0]
    if n < max(min_points, k_neighbors + 1, 3):
        return {}
    if labels is not None:
        labels = labels[finite]

    # ---- Step 1: what is each point's neighbourhood SHAPED like? ----
    normals, a1d, a2d, a3d, nn_spacing = _neighbourhood_geometry(pts, k_neighbors)
    surface_like = (a2d >= a1d) & (a2d >= a3d + scatter_margin)
    candidates = pts[surface_like]
    cand_normals = normals[surface_like]
    cand_labels = labels[surface_like] if labels is not None else None
    if candidates.shape[0] < min_points:
        return {}  # nothing in this cloud is shaped like a surface at all

    # ---- Step 2: smooth patches = spatial adjacency + normal continuity ----
    if connectivity_radius_m is None:
        connectivity_radius_m = max(4.0 * nn_spacing, 1e-3)
    components = _smooth_patch_components(
        candidates, cand_normals, connectivity_radius_m, max_normal_deg)

    # ---- Steps 3+4: accept or reject each WHOLE patch; never truncate one.
    surfaces = {}
    for comp in components:
        patch = candidates[comp]
        if patch.shape[0] < min_points:
            continue  # too small to be a real structural surface
        role = _patch_semantic_role(cand_labels[comp]) if cand_labels is not None else None
        if role in ("wall", "roof"):
            continue  # authoritatively wall/roof: not a slab surface; orientation is never consulted
        label = round(float(np.median(patch[:, 2])), 3)
        # Two distinct surfaces rounding to the same mm-precision label
        # still both survive -- nudge so neither is dropped.
        while label in surfaces:
            label = round(label + 1e-3, 6)
        surfaces[label] = patch
        if roles_out is not None:
            roles_out[label] = role

    return surfaces


def _points_inside(geom, points_xyz):
    """
    Rows of `points_xyz` (N, 3) whose XY lies inside `geom`.

    3D-FIX ROUND 4: this is a QUESTION, not a knife. It survives in exactly
    two roles -- asking whether a detected surface has any presence over a
    unit's footprint (attribution: the answer decides whether a whole
    surface belongs to this unit, never where that surface ends), and
    measuring coverage for a log line. It is no longer used anywhere to
    replace a measured cloud with a smaller one; `_measured_section_from_points`
    in particular no longer calls it at all. Anything that reintroduces
    `points = _points_inside(polygon, points)` into a geometry path is
    reintroducing the blueprint-clipping bug that pass removed.
    """
    try:
        from shapely import contains_xy
    except ImportError:  # shapely < 2.0
        from shapely.vectorized import contains as contains_xy
    if geom is None or geom.is_empty or len(points_xyz) == 0:
        return points_xyz[:0]
    mask = contains_xy(geom, points_xyz[:, 0], points_xyz[:, 1])
    return points_xyz[np.asarray(mask, dtype=bool)]


def _polygonal_only(geom):
    """Drop the lines/points a boolean op can leave behind; keep polygons."""
    from shapely.ops import unary_union
    parts = getattr(geom, "geoms", [geom])
    return unary_union([g for g in parts
                        if g.geom_type in ("Polygon", "MultiPolygon") and not g.is_empty])


def _measured_plan_extent(points_xyz, footprint):
    """
    The plan-view (x, y) coverage a measured point cloud actually has -- so
    a partial floor / mezzanine gets ITS OWN outline instead of being
    stamped across the whole footprint and extrapolated, AND a slab that
    genuinely oversails the blueprint (an overhang, a cantilever, a bay
    that reads bigger in LiDAR/photogrammetry than on the drawing) keeps
    that real extra extent instead of having it silently cut away.

    Coverage = grid cells (2x the cloud's own median nearest-neighbour
    spacing) that contain at least one measured point. `footprint` is NOT
    intersected into this -- it is never a hard mask on the measured
    evidence. The measured cloud alone decides `extent`, full stop: it can
    come out smaller than `footprint` (a partial slab), the same size, or
    LARGER (a real measured overhang/cantilever beyond the blueprint
    outline) -- all three are legitimate results, not error cases, and
    none of them get corrected back toward `footprint`.

    `footprint` is used only AFTER `extent` is already final, purely to
    annotate how the two compare:
      - `overhang`: the part of `extent` outside `footprint` -- measured
        coverage beyond the blueprint, kept as real geometry, not
        discarded or flagged as bad evidence.
      - `full`: whether `extent` covers `footprint` with no real gap left
        (a gap narrower than one grid cell is a sampling edge effect, not
        a measured absence, so it doesn't count against `full`).
    Neither annotation ever feeds back into `extent`, `points`, or any
    other returned geometry -- `footprint` validates; it does not clip.

    3D-FIX ROUND 4: this function no longer takes part in deciding floor
    topology. `_pair_floor_volumes` used to pair surfaces by intersecting
    these plan extents, which made an XY shadow stand in for vertical
    containment; pairing is now done by ray-casting against real B-Rep
    surface patches, and `extent` survives purely as the `region` / `full`
    annotation attached to a volume that has already been decided in 3D.
    Nothing that builds shape reads it.

    Returns None when there is no usable evidence (fewer than 3 finite
    points, or no plan extent), else a dict:
        {"points": (N, 3) cleaned cloud, "extent": polygonal geometry
         (the measured cloud's OWN coverage, never clipped to footprint),
         "full": bool, "res": cell size in metres,
         "overhang": polygonal geometry of extent beyond footprint, or
         None when there isn't any}
    """
    from scipy.spatial import cKDTree
    from shapely.geometry import box
    from shapely.ops import unary_union

    if points_xyz is None:
        return None
    pts = np.asarray(points_xyz, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 3:
        return None
    pts = pts[np.isfinite(pts).all(axis=1)]
    if pts.shape[0] < 3:
        return None

    xy = pts[:, :2]
    nn = cKDTree(xy).query(xy, k=2)[0][:, 1]
    nn = nn[nn > 0]
    if nn.size == 0:
        return None
    res = 2.0 * float(np.median(nn))

    # `extent` is derived ENTIRELY from the measured cloud's own occupied
    # cells -- `footprint` plays no part in computing it, so a real
    # overhang/cantilever beyond the blueprint survives intact.
    cells = np.unique(np.floor(xy / res).astype(np.int64), axis=0)
    extent = _polygonal_only(unary_union([box(cx * res, cy * res, (cx + 1) * res, (cy + 1) * res)
                                          for cx, cy in cells]))
    if extent.is_empty:
        return None

    # Validation/annotation ONLY, computed AFTER `extent` is final --
    # neither line below alters `extent` itself.
    overhang = _polygonal_only(extent.difference(footprint))
    gap = _polygonal_only(footprint.difference(extent))
    full = gap.is_empty or gap.buffer(-res / 2.0).is_empty
    return {"points": pts, "extent": extent, "full": full, "res": res,
            "overhang": overhang if not overhang.is_empty else None}


# ======================================================================
# MEASURED-SURFACE B-REP TOPOLOGY
#
# Everything below exists so that "is this surface above that one?" can be
# answered by intersecting a ray with real geometry instead of comparing
# two shadows on the ground. The chain is:
#     measured points
#       -> triangulated B-Rep face patch in the surface's own frame
#          (_measured_surface_brep, via _alpha_triangles)
#       -> ray cast along each measured point's own outward normal
#          (_local_surface_normals + _first_forward_hit)
#       -> nearest real face actually pierced = that point's ceiling.
# No step in that chain projects anything onto world (x, y), and no step
# consults a blueprint polygon.
# ======================================================================


def _median_nn_spacing(points_xyz):
    """
    Median nearest-neighbour distance of a real point cloud -- the cloud's
    own natural length scale. Every tolerance in this section is expressed
    as a multiple of it, so nothing here carries a hard-coded real-world
    distance that would be wrong for a denser or sparser survey. Returns
    None when the cloud is too small or fully degenerate.
    """
    from scipy.spatial import cKDTree
    pts = np.asarray(points_xyz, dtype=float)
    if pts.shape[0] < 2:
        return None
    d = cKDTree(pts).query(pts, k=2)[0][:, 1]
    d = d[d > 0]
    if d.size == 0:
        return None
    return float(np.median(d))


def _decimate_indices(points_xyz, max_points):
    """
    Indices of a spatially even subsample of a cloud, capped at
    `max_points`, by keeping the first point in each cell of a 3D voxel
    grid whose size is bisected until the occupied-cell count fits.

    This is a COST limiter for the two things whose runtime scales with
    point count -- how finely the B-Rep probe patch is tessellated, and how
    many rays get cast -- and nothing else. It never decides what evidence
    a volume is built from: `_pair_floor_volumes` casts rays from a
    decimated probe set and carries each probe's verdict back only onto
    measured points in the same connected, normal-continuous surface
    component; points with no probe there stay unresolved, not borrowed.

    Even (voxel) subsampling rather than random subsampling matters here:
    a random draw under-samples sparse regions, which are exactly where a
    thin cantilever edge or a narrow mezzanine lip lives.
    """
    pts = np.asarray(points_xyz, dtype=float)
    n = pts.shape[0]
    if max_points is None or n <= max_points:
        return np.arange(n)

    origin = pts.min(axis=0)
    diag = float(np.linalg.norm(pts.max(axis=0) - origin))
    if not np.isfinite(diag) or diag <= 0:
        return np.arange(min(n, max_points))

    lo, hi = diag / float(n), diag  # cell size bracket: too fine .. too coarse
    keep = None
    for _ in range(24):
        cell = 0.5 * (lo + hi)
        keys = np.floor((pts - origin) / cell).astype(np.int64)
        _uniq, first = np.unique(keys, axis=0, return_index=True)
        if first.size > max_points:
            lo = cell
        else:
            keep = first
            hi = cell
            if first.size >= 0.8 * max_points:
                break
    if keep is None:  # bisection never landed under the cap -- plain stride
        keep = np.arange(0, n, max(1, n // max_points))[:max_points]
    return np.sort(keep)


def _local_surface_normals(points_xyz, k_neighbors=12):
    """
    Per-point outward normal of a real measured surface, from each point's
    own k nearest 3D neighbours -- computed by `_neighbourhood_geometry`,
    the same single primitive `_detect_slab_surfaces` segments with, so
    the direction a volume is probed along and the direction that decided
    which patch a point belongs to can never drift apart.

    This is the direction `_pair_floor_volumes` probes in, and using the
    surface's OWN perpendicular rather than the world Z axis is what makes
    the pairing three-dimensional rather than gravity-fed: a pitched deck,
    a ramp, a tilted mezzanine, or a bowed slab is probed across its real
    clear space, and a curved slab's probe direction turns with the
    curvature instead of staying fixed. For a flat slab the normal IS
    vertical, so the familiar case is unchanged.

    Normals are returned UNSIGNED: PCA cannot say which side of a surface
    faces the clear space, and no world-Z convention is applied. Which side
    a probe leaves from is established by `_pair_floor_volumes` from
    measured evidence (which side actually reaches another measured
    surface), and left unresolved when that is ambiguous. A point whose
    local covariance is fully degenerate (all neighbours coincident) has no
    measurable normal and gets NaN, so it can never probe or be probed.
    """
    pts = np.asarray(points_xyz, dtype=float)
    if pts.shape[0] == 0:
        return np.zeros((0, 3))
    normals, a1d, a2d, a3d, _spacing = _neighbourhood_geometry(pts, k_neighbors)
    normals = normals.copy()
    normals[(a1d == 0.0) & (a2d == 0.0) & (a3d == 1.0)] = np.nan  # degenerate: no normal
    return normals


def _connected_components_3d(points_xyz, radius):
    """
    Connected components of a point cloud in TRUE 3D, linking points within
    `radius` of each other -- the same neighbour-graph technique
    `_detect_slab_surfaces` uses to tell one physical slab from another.

    `_pair_floor_volumes` uses it to split the part of a slab that shares a
    ceiling into physically separate pieces. Doing this in 3D rather than
    on a plan polygon matters for exactly the shapes this pipeline is meant
    to keep: two lobes of a slab that are disjoint in space but whose plan
    outlines touch at a corner stay separate volumes, and a split-level
    deck whose halves are joined by a real ramp stays ONE volume even
    though a plan view shows two elevations.

    Returns a list of index arrays, largest component first.
    """
    from scipy.spatial import cKDTree
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    pts = np.asarray(points_xyz, dtype=float)
    m = pts.shape[0]
    if m == 0:
        return []
    if m == 1:
        return [np.array([0])]

    pairs = cKDTree(pts).query_pairs(r=radius, output_type="ndarray")
    if pairs.size:
        rows = np.concatenate([pairs[:, 0], pairs[:, 1]])
        cols = np.concatenate([pairs[:, 1], pairs[:, 0]])
        graph = coo_matrix((np.ones(rows.shape[0]), (rows, cols)), shape=(m, m))
    else:
        graph = coo_matrix((m, m))
    n_comp, labels = connected_components(graph, directed=False)
    comps = [np.flatnonzero(labels == c) for c in range(n_comp)]
    comps.sort(key=len, reverse=True)
    return comps


def _local_chart_triangles(points_xyz, k_neighbors=16, alpha=None,
                           max_normal_deg=18.0, max_edge_factor=4.0):
    """
    Tessellate a measured surface patch of ANY shape into 3D triangles,
    using one small local chart per point instead of a single global
    parametrization.

    For each point, its k nearest neighbours are projected into THAT
    POINT'S own tangent frame, Delaunay-triangulated there, and only the
    triangles incident to the point itself are kept; the union over all
    points, deduplicated, is the mesh. Each chart covers a neighbourhood a
    few point-spacings across, where any smooth surface is locally a
    graph -- so the method is indifferent to how the patch behaves
    globally.

    This replaces a single Delaunay in the patch's overall PCA (u, v)
    frame. That earlier approach was a global h = f(u, v) height field in
    disguise: it is correct only while the whole patch projects one-to-one
    onto one plane. It was tolerable while `_detect_slab_surfaces` split
    everything into near-planar pieces by RANSAC, and stopped being
    tolerable the moment that stage started preserving vaults, domes, and
    folds whole -- a barrel vault turning past 90 degrees, or an eave
    curling back under itself, projects onto its own mean plane TWICE, so
    the global triangulation would weld the two sheets together into a
    self-intersecting mesh and the ray topology built on it would report
    hits on surface that does not exist.

    Alpha filtering is applied in 3D: a triangle is kept only when its
    circumradius is within `alpha` (default a small multiple of the
    patch's own median nearest-neighbour spacing), so the mesh still stops
    where the measurements stop -- an unmeasured gap, a light-well, or the
    ragged edge of a partial slab is left open rather than bridged.

    Surface-consistency gate: alpha alone cannot tell two physically
    separate sheets apart when they lie within a few spacings of each other
    (a fold, an undercut, an eave over a soffit), because a triangle
    reaching across the gap can still be small. A triangle is therefore
    also rejected unless its three MEASURED vertices are locally
    surface-consistent: (a) each edge is at most `max_edge_factor` times the
    larger of its endpoints' own local point spacing; (b) each edge lies
    within `max_normal_deg` of the tangent plane at BOTH its endpoints
    (tangent continuity -- an edge running along a vertex normal is a bridge
    between sheets, not a step along one); (c) the unsigned angle between
    the two endpoint normals of every edge is within `max_normal_deg`
    (normal continuity); and (d) the triangle's own facet normal agrees with
    all three vertex normals to the same angle. Every test uses measured
    points and their neighbourhood-derived normals only. A rejected
    triangle is simply omitted -- no connecting face is ever added.

    Returns (triangles (T, 3) index array, alpha_used), or (None, ...) when
    the patch is too small or degenerate to mesh.
    """
    from scipy.spatial import Delaunay, cKDTree

    pts = np.asarray(points_xyz, dtype=float)
    n = pts.shape[0]
    if n < 3:
        return None, None
    if n == 3:
        if np.linalg.norm(np.cross(pts[1] - pts[0], pts[2] - pts[0])) < 1e-12:
            return None, None  # collinear -- no surface
        return np.array([[0, 1, 2]]), None

    tree = cKDTree(pts)
    k = min(k_neighbors, n - 1)
    nn_dists, nn_idx = tree.query(pts, k=k + 1)
    spacing = float(np.median(nn_dists[:, 1]))
    if alpha is None:
        alpha = max(3.0 * spacing, 1e-6)

    normals, _a1, _a2, _a3, _sp = _neighbourhood_geometry(pts, k_neighbors=k)

    tri_set = set()
    for i in range(n):
        idx = nn_idx[i]
        local = pts[idx] - pts[i]
        nrm = normals[i]
        # Any two directions spanning this point's own tangent plane.
        helper = np.array([1.0, 0.0, 0.0])
        if abs(np.dot(helper, nrm)) > 0.9:
            helper = np.array([0.0, 1.0, 0.0])
        u_dir = np.cross(nrm, helper)
        u_dir /= (np.linalg.norm(u_dir) or 1.0)
        v_dir = np.cross(nrm, u_dir)
        chart = np.column_stack((local @ u_dir, local @ v_dir))
        try:
            dt = Delaunay(chart)
        except Exception:
            continue  # this chart is degenerate; other charts still cover it
        for simplex in dt.simplices:
            if 0 not in simplex:
                continue  # keep only triangles incident to the chart's centre
            tri = tuple(sorted(int(idx[s]) for s in simplex))
            if len(set(tri)) == 3:
                tri_set.add(tri)

    if not tri_set:
        return None, alpha

    tris = np.array(sorted(tri_set), dtype=np.int64)
    p0, p1, p2 = pts[tris[:, 0]], pts[tris[:, 1]], pts[tris[:, 2]]
    a = np.linalg.norm(p1 - p2, axis=1)
    b = np.linalg.norm(p0 - p2, axis=1)
    c = np.linalg.norm(p0 - p1, axis=1)
    s = (a + b + c) / 2.0
    area = np.sqrt(np.clip(s * (s - a) * (s - b) * (s - c), 0.0, None))
    with np.errstate(divide="ignore", invalid="ignore"):
        circum_r = np.where(area > 1e-12, (a * b * c) / (4.0 * area + 1e-15), np.inf)
    tris = tris[circum_r <= alpha]

    if tris.shape[0]:
        # Surface-consistency gate -- see the docstring. Vectorised over all
        # surviving triangles; only ever removes triangles.
        local_sp = np.maximum(nn_dists[:, 1:min(4, k + 1)].mean(axis=1), 1e-9)
        cos_tol = np.cos(np.radians(max_normal_deg))
        sin_tol = np.sin(np.radians(max_normal_deg))
        q0, q1, q2 = pts[tris[:, 0]], pts[tris[:, 1]], pts[tris[:, 2]]
        face_n = np.cross(q1 - q0, q2 - q0)
        face_len = np.linalg.norm(face_n, axis=1)
        keep = face_len > 1e-12
        face_n = face_n / np.where(face_len > 1e-12, face_len, 1.0)[:, None]
        for ca, cb in ((0, 1), (1, 2), (0, 2)):
            ia, ib = tris[:, ca], tris[:, cb]
            edge = pts[ib] - pts[ia]
            edge_len = np.linalg.norm(edge, axis=1)
            edge_dir = edge / np.where(edge_len > 1e-12, edge_len, 1.0)[:, None]
            keep &= edge_len <= max_edge_factor * np.maximum(local_sp[ia], local_sp[ib])
            keep &= np.abs(np.einsum("ij,ij->i", edge_dir, normals[ia])) <= sin_tol
            keep &= np.abs(np.einsum("ij,ij->i", edge_dir, normals[ib])) <= sin_tol
            keep &= np.abs(np.einsum("ij,ij->i", normals[ia], normals[ib])) >= cos_tol
        for cc in range(3):
            keep &= np.abs(np.einsum("ij,ij->i", face_n, normals[tris[:, cc]])) >= cos_tol
        tris = tris[keep]

    if tris.shape[0] == 0:
        return None, alpha
    return tris, alpha


def _measured_surface_brep(points_xyz, max_facet_points=2500, alpha=None):
    """
    Turn ONE measured surface's (x, y, z) points into a real OpenCASCADE
    B-Rep face patch: a local-chart tessellation of the measured points
    (`_local_chart_triangles`), with every triangle built as an explicit
    `Geom_Plane` through its OWN three measured vertices, trimmed by the
    triangle wire. This is the geometry `_pair_floor_volumes` fires rays
    at, and it is the whole reason that function can stop reasoning about
    plan-view shadows.

    Why this shape, specifically:
      - It is a COMPOUND OF FACES, not a sewn shell and not a solid. A
        measured slab is a surface: it has an extent and two sides, and it
        is not the boundary of anything closed. Sewing it into a
        pseudo-solid so it could be point-classified would be inventing
        the very containment the caller is trying to measure.
      - The vertices are the measured points themselves -- no smoothing, no
        resampling onto a grid, no fitted surface standing in for the
        evidence. (End-cap faces are likewise triangulated between the same measured
        points -- see `_measured_cap_shape`.)
      - Alpha filtering means the patch stops where the measurements stop.
        A ray passing over an unmeasured gap, a light-well, or the open
        side of a partial slab finds nothing there, which is the correct
        answer rather than a phantom ceiling spanning the void.
      - Tessellation is per-point LOCAL CHART, never one global frame (see
        `_local_chart_triangles`). A pitched roof, a dormer cheek, a vaulted
        soffit, an eave that curls back under itself, or a patch that folds
        right past its own mean plane is meshed correctly, because no step
        ever asks the patch as a whole to project one-to-one onto anything.
        This matters more since `_detect_slab_surfaces` stopped splitting
        curved evidence into near-planar shards: a vault now arrives here in
        one piece, and a single global (u, v) Delaunay would have welded its
        two sheets together where they overlap in projection. A ray fired at
        the resulting mesh pierces each real sheet in turn and takes the
        nearest, which is what recovers a true undercut ordering.

    `max_facet_points` caps tessellation cost only (see
    `_decimate_indices`); the caller keeps and pairs the full cloud.

    Returns None -- never an approximate stand-in -- when the cloud is too
    small, degenerate, or produces no surviving triangle. Otherwise a dict:
        {"shape": TopoDS_Compound of faces, "n_faces": int,
         "spacing": this cloud's median nearest-neighbour distance,
         "facet_points": the points actually tessellated, "alpha": float}
    """
    pts = np.asarray(points_xyz, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 3:
        return None
    pts = pts[np.isfinite(pts).all(axis=1)]
    if pts.shape[0] < 3:
        return None

    facet_pts = pts[_decimate_indices(pts, max_facet_points)]
    if facet_pts.shape[0] < 3:
        return None

    tris, alpha_used = _local_chart_triangles(facet_pts, alpha=alpha)
    if tris is None:
        return None

    builder = BRep_Builder()
    compound = TopoDS_Compound()
    builder.MakeCompound(compound)

    n_faces = 0
    for i0, i1, i2 in tris:
        a, b, c = facet_pts[i0], facet_pts[i1], facet_pts[i2]
        nrm = np.cross(b - a, c - a)
        ln = float(np.linalg.norm(nrm))
        if ln <= 1e-12:
            continue  # collinear in 3D -- no face to build
        nrm = nrm / ln
        try:
            # An explicit supporting surface through the triangle's own
            # three measured vertices, trimmed by the triangle wire -- the
            # same "surface + wire" discipline _make_trimmed_face uses, so
            # no face in this file is ever built from a wire alone. Three
            # points are exactly planar by construction, so this plane is
            # the evidence, not a best fit to it.
            plane = Geom_Plane(gp_Pnt(*a.tolist()), gp_Dir(*nrm.tolist()))
            wire = BRepBuilderAPI_MakePolygon(
                gp_Pnt(*a.tolist()), gp_Pnt(*b.tolist()), gp_Pnt(*c.tolist()), True).Wire()
            face = BRepBuilderAPI_MakeFace(plane, wire).Face()
        except Exception:
            continue  # one unbuildable facet never invalidates the patch
        builder.Add(compound, face)
        n_faces += 1

    if n_faces == 0:
        return None
    return {
        "shape": compound,
        "n_faces": n_faces,
        "spacing": _median_nn_spacing(facet_pts) or 1e-3,
        "facet_points": facet_pts,
        "alpha": alpha_used,
    }


def _first_forward_hit(intersector, origin_xyz, direction_xyz, t_min, t_max):
    """
    Nearest point at which a ray actually pierces a loaded B-Rep face
    patch, strictly ahead of its origin.

    `IntCurvesFace_ShapeIntersector.Perform` reports EVERY face the ray
    crosses together with its distance along the ray, so taking the
    smallest such distance gives genuine first-hit occlusion ordering: if a
    balcony slab and the floor plate two storeys up both sit over a point,
    the balcony is what that point is under. A ray that crosses a folded
    patch several times likewise resolves to the first crossing, which is
    what makes an undercut come out as an undercut instead of an average.

    Returns (distance, hit_point) or None when the ray misses entirely --
    and a miss is reported as a miss, never rounded up to "probably the
    nearest thing in plan".
    """
    try:
        lin = gp_Lin(
            gp_Pnt(float(origin_xyz[0]), float(origin_xyz[1]), float(origin_xyz[2])),
            gp_Dir(float(direction_xyz[0]), float(direction_xyz[1]), float(direction_xyz[2])),
        )
    except Exception:
        return None  # null/degenerate direction -- no probe is possible
    try:
        intersector.Perform(lin, t_min, t_max)
        if not intersector.IsDone() or intersector.NbPnt() == 0:
            return None
        best = None
        for i in range(1, intersector.NbPnt() + 1):
            w = float(intersector.WParameter(i))
            if w < t_min:
                continue
            if best is None or w < best[0]:
                p = intersector.Pnt(i)
                best = (w, np.array([p.X(), p.Y(), p.Z()], dtype=float))
        return best
    except Exception:
        return None


def _pair_floor_volumes(footprint, slab_points_by_label, roof_points_xyz=None,
                        max_probe_points=2500, max_facet_points=2500, orientation_up=None,
                        surface_roles=None):
    """
    Decide, for ONE unit, which measured surface is the bottom and which is
    the ceiling of each floor volume -- from the 3D evidence itself.

    Inputs: the unit's global-frame `footprint` polygon (ANNOTATION ONLY --
    see below); `slab_points_by_label` {floor_label: (N, 3) measured slab
    points, or None when nothing was measured for that label}; and the roof
    point cloud (or None).

    WHAT THIS REPLACES (3D-FIX ROUND 4). The previous rule rasterized each
    measured cloud into a 2D plan `extent`, intersected those extents, and
    declared one surface to be above another when its MEDIAN Z inside the
    overlap was greater. Both halves of that are 2.5D reasoning wearing 3D
    vocabulary:
      - An XY overlap is a shadow, not a containment. A cantilever or
        balcony projects over the slab below without roofing it; an eave
        that curls back under itself, an arch, or any undercut projects
        onto plan twice, so its shadow claims ground it is not over; and
        two structures whose shadows coincide may have nothing between
        them at all. Overlap alone therefore implied vertical containment
        that the geometry did not support, which is precisely the failure
        this pass exists to remove.
      - A median Z collapses a whole surface to one number. A tilted deck,
        a bowed slab, a vault, or a ramp has no single elevation, and two
        surfaces that interleave -- one over the other across part of the
        overlap and under it across the rest -- got one verdict for the
        entire region, so the wrong half was silently mispaired.

    ELIGIBILITY. A surface can be a floor/ceiling only with authoritative
    semantic evidence: `surface_roles` {label: "slab"} (from
    _detect_slab_surfaces, i.e. slab/floor/ceiling/deck labels). A label
    whose role is missing, None or anything else (an unlabelled surface that
    may be a facade/wall) is never built, probed or hit: it is reported in
    the notes and no volume comes from it. A ray hit never promotes a
    surface to a floor/ceiling. The roof cloud `roof_points_xyz` is NOT
    assumed to be roof-semantic (nothing here can verify that the upstream
    cloud is more than generic exposed surface): it is eligible only when
    the caller passes `surface_roles["roof"] == "roof"`, which the caller
    sets only from explicit upstream roof semantics. Otherwise it is left
    unresolved and noted. No Z, height, count or plan test is used, and no
    role is ever inferred or fabricated.

    THE RULE NOW. Every eligible measured cloud (each slab, and the roof) is turned
    into a real OpenCASCADE B-Rep face patch built from its own points
    (`_measured_surface_brep`). For each slab taken as a bottom, a ray is
    cast FROM each measured point ALONG that point's own surface normal
    (`_local_surface_normals`; the side is set by which side reaches measured
    surface, never by world Z) and intersected against every other
    patch (`_first_forward_hit`). The nearest face the ray actually pierces
    is that point's ceiling. Per point, first hit, in true 3D:
      - an XY overlap with no material over the point produces no hit and
        no volume, so a shadow can never imply containment again;
      - a cantilever or balcony roofs exactly the points its patch is
        actually over, and oversails the rest with no volume beneath;
      - an undercut or folded surface is pierced repeatedly and resolves to
        its FIRST crossing, which is what an undercut means;
      - a curved, bowed, or pitched surface is probed perpendicular to
        itself at every point, so its clear space is measured along the
        curvature rather than averaged away;
      - a partial slab or mezzanine produces volume over exactly the points
        it covers, and the slab above keeps the open remainder.
    Nothing consults labels, label order, floor count or storey height, and
    nothing compares medians.

    Points sharing a ceiling are then split into physically separate pieces
    by 3D connectivity (`_connected_components_3d`), not by cutting a plan
    polygon -- so one slab still yields several volumes for a mezzanine,
    a split level, or a disjoint wing, but two lobes that merely touch at a
    corner in plan are not fused.

    EVIDENCE HANDED BACK. `bottom_points` / `ceiling_points` are the
    measured SUBSETS that genuinely bound each volume -- the bottom points
    whose rays reached this ceiling, and the ceiling points those rays
    actually landed among. That is what let `_measured_section_from_points`
    drop its polygon clip entirely: a partial volume now arrives already
    carrying only its own evidence, established by ray casting against
    measured geometry, so no outline has to be intersected in afterwards to
    stop it inheriting a full slab's footprint. Every measured point of a
    surface is carried to some volume or reported as open; nothing is
    thrown away, and a shared slab still hands the volume below and the
    volume above the same points where both genuinely bound it.

    `footprint` NEVER touches geometry here. It is passed to
    `_measured_plan_extent` purely to compute the `region` / `full`
    annotations after a volume is already final, and those annotations are
    not read back by anything that builds shape. A volume whose measured
    evidence oversails the blueprint keeps it, and `region` comes out
    larger than `footprint` rather than being trimmed to it.

    `max_probe_points` / `max_facet_points` bound runtime (rays cast, and
    triangles per patch). Probe verdicts are carried back onto the full cloud
    only within the same connected, normal-continuous 3D surface component
    (never by nearest Euclidean probe alone); points with no such probe stay
    unresolved and build no volume.

    Returns (volumes, notes). Each volume dict: "bottom" (label), "ceiling"
    (label or "roof"), "bottom_points"/"ceiling_points" (the bounding
    measured subsets described above), "bottom_surface_points"/
    "ceiling_surface_points" (each surface's whole cloud, for provenance),
    "clearance" (median measured distance from bottom to ceiling along the
    probe normals -- a real statistic, never a storey height),
    "region"/"full" (plan-view annotation only, possibly None), and
    "spans_unmeasured" (labels with no measured slab sitting between bottom
    and ceiling by label -- a log annotation only: it flags the
    double-height-vs-dropout ambiguity).
    """
    from scipy.spatial import cKDTree

    # `orientation_up`: optional authoritative UP direction -- the direction
    # pointing from a floor toward its ceiling (sensor/IMU metadata already
    # expressed as UP). It is NOT a gravity / down / acceleration vector: a
    # probe leaves along the normal side with normal . orientation_up > 0, so
    # a gravity (down) vector would flip every probe into the floor. The
    # caller must convert gravity to UP explicitly (negate it) BEFORE supplying
    # it; it is never inverted, guessed or inferred here, and nothing else
    # (e.g. a "gravity" key) is ever read as UP. It only picks which side of
    # an unsigned surface normal a probe leaves from; it never builds,
    # projects or trims geometry. Absent or unusable (wrong shape, non-finite,
    # zero), orientation comes from measured two-sided hits alone and
    # ambiguous probes stay unresolved.
    up = None
    if orientation_up is not None:
        up = np.asarray(orientation_up, dtype=float).reshape(-1)
        if up.shape != (3,) or not np.isfinite(up).all() or np.linalg.norm(up) < 1e-12:
            up = None
        else:
            up = up / np.linalg.norm(up)

    notes = []
    surfaces = []
    unmeasured = []

    def _build_surface(label, pts):
        arr = np.asarray(pts, dtype=float) if pts is not None else None
        if arr is None or arr.ndim != 2 or arr.shape[1] != 3:
            return None
        arr = arr[np.isfinite(arr).all(axis=1)]
        if arr.shape[0] < 3:
            return None
        brep = _measured_surface_brep(arr, max_facet_points=max_facet_points)
        if brep is None:
            return None
        intersector = IntCurvesFace_ShapeIntersector()
        intersector.Load(brep["shape"], 1e-6)
        return {
            "key": label,
            "label": label,
            "points": arr,                              # the WHOLE measured cloud
            "normals": _local_surface_normals(arr),
            "spacing": brep["spacing"],
            "shape": brep["shape"],
            "inter": intersector,
            "tree": cKDTree(arr),
        }

    for label, pts in slab_points_by_label.items():
        if (surface_roles or {}).get(label) != "slab":
            notes.append(f"floor label {label}: authoritative surface-role evidence is missing "
                         f"(no slab/floor/ceiling/deck semantic label) -- surface left "
                         f"unresolved, not eligible as floor/ceiling, no volume built from it")
            continue
        surf = _build_surface(label, pts)
        if surf is None:
            unmeasured.append(label)
            notes.append(f"floor label {label}: no measured slab evidence that can be built "
                         f"into a 3D surface patch")
            continue
        surfaces.append(surf)

    roof = None
    if (surface_roles or {}).get("roof") != "roof":
        notes.append("roof point cloud has no explicit roof semantics (authoritative surface-role "
                     "evidence is missing) -- left unresolved, not eligible as a ceiling")
    else:
        roof = _build_surface("roof", roof_points_xyz)
        if roof is None:
            notes.append("no roof point cloud usable as a 3D surface for this unit")
    if roof is not None:
        surfaces.append(roof)

    if len(surfaces) < 2:
        return [], notes

    # Deterministic processing/reporting order only; the pairing below never
    # reads it -- ordering by elevation is not evidence of what is above what.
    surfaces.sort(key=lambda s: float(np.median(s["points"][:, 2])))
    by_key = {s["key"]: s for s in surfaces}

    # Ray length bound, taken from the measured data's own extent rather
    # than a guessed maximum storey height.
    all_pts = np.vstack([s["points"] for s in surfaces])
    t_max = 2.0 * float(np.linalg.norm(all_pts.max(axis=0) - all_pts.min(axis=0))) + 1.0

    def _spans(bottom_label, ceiling_label):
        if ceiling_label == "roof":
            return sorted(l for l in unmeasured if l > bottom_label)
        lo, hi = sorted((bottom_label, ceiling_label))
        return sorted(l for l in unmeasured if lo < l < hi)

    volumes = []
    for bottom in surfaces:
        if bottom["key"] == "roof":
            continue  # nothing is above the roof
        others = [s for s in surfaces if s is not bottom]
        if not others:
            continue

        # ---- Probe: one ray per measured point (decimated only for cost),
        # along that point's OWN outward normal, against real B-Rep faces.
        probe_idx = _decimate_indices(bottom["points"], max_probe_points)
        probes = bottom["points"][probe_idx]
        probe_dirs = bottom["normals"][probe_idx]
        t_min = max(1e-6, 1e-2 * bottom["spacing"])

        n_probes = probes.shape[0]
        hit_key = np.empty(n_probes, dtype=object)
        hit_key[:] = None
        hit_pos = np.full((n_probes, 3), np.nan)
        hit_dist = np.full(n_probes, np.nan)

        probe_ok = np.ones(n_probes, dtype=bool)
        for j in range(n_probes):
            if not np.isfinite(probe_dirs[j]).all():
                probe_ok[j] = False  # no measurable normal: unresolved
                continue
            # The normal is unsigned and no world-Z convention picks a side.
            # With an authoritative UP direction the side is the one facing it
            # (unresolved when the normal is nearly perpendicular to it).
            # Otherwise both sides are cast and the side is established by
            # measured evidence only when exactly ONE side reaches another
            # measured surface; hits on both sides leave the orientation
            # ambiguous, so the probe stays unresolved.
            if up is not None:
                dot = float(probe_dirs[j] @ up)
                if abs(dot) < ORIENTATION_MIN_ABS_DOT:
                    probe_ok[j] = False
                    continue
                signs = (1.0 if dot > 0 else -1.0,)
            else:
                signs = (1.0, -1.0)
            per_side = []
            for sign in signs:
                best_side = None
                for cand in others:
                    h = _first_forward_hit(cand["inter"], probes[j], sign * probe_dirs[j],
                                           t_min, t_max)
                    if h is None:
                        continue
                    # A valid forward B-Rep ray hit (distance >= t_min, within
                    # t_max, on a real face) is accepted or rejected by its
                    # true 3D ray distance and topology only -- never by
                    # world-Z ordering, and never promoted to a ceiling on
                    # the strength of plan position.
                    if best_side is None or h[0] < best_side[0]:
                        best_side = (h[0], cand["key"], h[1])
                per_side.append(best_side)
            if len(per_side) == 2 and per_side[0] is not None and per_side[1] is not None:
                probe_ok[j] = False
                continue
            best = next((b for b in per_side if b is not None), None)
            if best is not None:
                hit_dist[j], hit_key[j], hit_pos[j] = best[0], best[1], best[2]

        # ---- Carry each probe's verdict back onto measured points ONLY within
        # the same local 3D surface component (spatial adjacency + normal
        # continuity, `_smooth_patch_components`) and only from a probe whose
        # normal continues the point's own. A nearby but separate sheet (fold,
        # undercut, stacked slab) is never borrowed from by Euclidean nearness.
        # A point with no such probe stays UNRESOLVED (no ceiling assigned).
        comp_radius = max(4.0 * bottom["spacing"], 1e-3)
        max_normal_deg = 18.0  # same continuity threshold as _detect_slab_surfaces
        n_pts = bottom["points"].shape[0]
        patch_label = np.full(n_pts, -1, dtype=int)
        for ci, members in enumerate(_smooth_patch_components(
                bottom["points"], bottom["normals"], comp_radius, max_normal_deg)):
            patch_label[members] = ci
        point_ceiling = np.empty(n_pts, dtype=object)
        point_ceiling[:] = None
        point_hit = np.full((n_pts, 3), np.nan)
        point_dist = np.full(n_pts, np.nan)
        resolved = np.zeros(n_pts, dtype=bool)
        cos_limit = np.cos(np.radians(max_normal_deg)) - 1e-12
        probe_patch = patch_label[probe_idx]
        for ci in np.unique(patch_label):
            members = np.flatnonzero(patch_label == ci)
            probe_ids = np.flatnonzero(probe_patch == ci)
            if probe_ids.size == 0:
                continue
            src = probe_ids[cKDTree(probes[probe_ids]).query(bottom["points"][members], k=1)[1]]
            cont = np.abs(np.einsum("ij,ij->i", bottom["normals"][members],
                                    probe_dirs[src])) >= cos_limit
            cont &= probe_ok[src]  # ambiguous/undefined orientation: unresolved
            ok, src = members[cont], src[cont]
            point_ceiling[ok] = hit_key[src]
            point_hit[ok] = hit_pos[src]
            point_dist[ok] = hit_dist[src]
            resolved[ok] = True

        n_unresolved = int((~resolved).sum())
        if n_unresolved:
            notes.append(f"floor label {bottom['label']}: {n_unresolved} of {n_pts} measured slab "
                         f"points have no probe with an established orientation in their own "
                         f"connected, normal-continuous surface component (no probe, undefined "
                         f"normal, or measured surfaces on both sides) -- ceiling left "
                         f"unresolved, no volume built there")
        n_open = int(sum(1 for k, r in zip(point_ceiling, resolved) if r and k is None))
        if n_open:
            notes.append(f"floor label {bottom['label']}: {n_open} of {len(point_ceiling)} "
                         f"measured slab points have no measured surface above them along "
                         f"the slab's own outward normal -- no volume built there")

        ceiling_keys = sorted({k for k in point_ceiling if k is not None}, key=str)
        for ckey in ceiling_keys:
            cand = by_key[ckey]
            sel = np.array([k == ckey for k in point_ceiling], dtype=bool)
            sub_pts = bottom["points"][sel]
            sub_hits = point_hit[sel]
            sub_dist = point_dist[sel]
            # How near a measured ceiling point has to be to a ray's landing
            # point to count as evidence for THIS volume. Scaled to the
            # CEILING's own spacing, because that is the cloud being sampled:
            # one spacing reaches the facet vertices the ray actually landed
            # between, and the 1.5x allows for an uneven local density. A
            # looser radius would sweep in ceiling material that overhangs
            # past the bottom slab's edge -- real, measured, and correctly
            # kept in "ceiling_surface_points", but NOT part of what roofs
            # this volume. That distinction is the whole point of the
            # cantilever case: the oversail is preserved as geometry without
            # manufacturing floor beneath it.
            assoc_r = max(1.5 * cand["spacing"], 1e-3)

            for comp in _connected_components_3d(sub_pts, comp_radius):
                comp_pts = sub_pts[comp]
                if comp_pts.shape[0] < 3:
                    notes.append(f"floor label {bottom['label']}: a {comp_pts.shape[0]}-point "
                                 f"patch under surface '{cand['label']}' is too small to derive "
                                 f"a section from -- no volume built there")
                    continue

                # The ceiling's OWN measured points that these rays actually
                # landed among -- a measured-to-measured association, with no
                # polygon involved on either side.
                comp_hits = sub_hits[comp]
                comp_hits = comp_hits[np.isfinite(comp_hits).all(axis=1)]
                hit_idx = set()
                if comp_hits.shape[0]:
                    for neigh in cand["tree"].query_ball_point(comp_hits, r=assoc_r):
                        hit_idx.update(neigh)
                if len(hit_idx) < 3:
                    notes.append(f"floor label {bottom['label']}: rays reach surface "
                                 f"'{cand['label']}' but land near fewer than 3 of its measured "
                                 f"points -- not enough ceiling evidence, no volume built there")
                    continue
                ceiling_pts = cand["points"][np.array(sorted(hit_idx))]

                # ---- Annotation only, computed after the volume is final.
                ann = _measured_plan_extent(comp_pts, footprint)
                region = ann["extent"] if ann is not None else None
                full = bool(ann["full"]) if ann is not None else False

                comp_dist = sub_dist[comp]
                comp_dist = comp_dist[np.isfinite(comp_dist)]
                clearance = float(np.median(comp_dist)) if comp_dist.size else None

                volumes.append({
                    "bottom": bottom["label"],
                    "ceiling": cand["label"],
                    "region": region,
                    "full": full,
                    "bottom_points": comp_pts,
                    "ceiling_points": ceiling_pts,
                    "bottom_surface_points": bottom["points"],
                    "ceiling_surface_points": cand["points"],
                    "clearance": clearance,
                    "spans_unmeasured": _spans(bottom["label"], cand["label"]),
                    "_sort": (float(np.median(ceiling_pts[:, 2])),
                              float(np.mean(comp_pts[:, 0])),
                              float(np.mean(comp_pts[:, 1]))),
                })

    volumes.sort(key=lambda v: (v["bottom"], v["_sort"]))
    for v in volumes:
        del v["_sort"]
    return volumes, notes


def run_unified_cadastre_pipeline(image_path, site_lat=None, site_lon=None,
                                  state_code=None, district_code=None,
                                  pixel_scale_m=None, gcp_pixels=None, gcp_real_world=None,
                                  floor_h=None, floors=None, is_demolition=False, 
                                  lidar_path=None, include_demo_subsurface=False,
                                  export_model=True, db=None, building_label=None,
                                  require_blueprint_evidence=True, location=None,
                                  jurisdiction=None):
    """
    `jurisdiction`: optional authoritative `Jurisdiction` (or dict of its
    fields: state_code, district_code, source) for this building. If given it
    is validated and used. If omitted and a GPS `location` is supplied, the
    codes come from `resolve_jurisdiction(lat, lon)` -- the replaceable
    resolver boundary -- and if that yields no authoritative result the run
    fails with JurisdictionError before anything is registered. The legacy
    `state_code`/`district_code` pair is an explicit manual override (source
    "manual_override"); it cannot be combined with `jurisdiction`. Codes are
    never defaulted, looked up from a built-in table, or guessed.

    `location`: the property's user-provided location, a `PropertyLocation`
    (or a dict of its fields: latitude, longitude, source, optional
    accuracy_m, crs). It is validated, kept unmodified as authoritative
    location metadata (returned as `result["location"]`), and used only to
    anchor the blueprint (with `pixel_scale_m`) when no GCPs are supplied;
    GPS never produces geometry or Z. The legacy `site_lat`/`site_lon` pair
    is still accepted and is wrapped as a `source="manual"` location; giving
    both `location` and `site_lat`/`site_lon` is an error.

    `site_lat`/`site_lon`/`pixel_scale_m` (or `gcp_pixels`/`gcp_real_world`):
    FIX -- there used to be a hardcoded fallback anchor (a fixed dummy UTM
    coordinate) used whenever site_lat/site_lon were omitted, plus a
    hardcoded assumption that every blueprint image is exactly 1000x1000px
    representing a 50x50m footprint. That meant (a) any building registered
    without an explicit site coordinate silently landed on top of every
    other such building at the same phantom location, and (b) even WITH a
    real site_lat/site_lon, the footprint was scaled as if every building
    were the same physical size regardless of what the blueprint actually
    depicted. Neither is acceptable for a real multi-building cadastre.

    There is now NO default location or default scale. A caller must
    georeference each building one of two ways:
      1. `site_lat` + `site_lon` + `pixel_scale_m` (real-world meters
         represented by one blueprint pixel) -- the blueprint's pixel
         origin (0,0) is anchored at that lat/lon, and the other two GCPs
         are derived from the blueprint's ACTUAL pixel dimensions (read
         from the file) times that scale, not an assumed footprint size.
      2. `gcp_pixels` + `gcp_real_world` -- explicit matched ground-control
         points (>=3 each) measured directly off this blueprint, for exact
         manual georeferencing (survey-grade, no scale assumption at all).
    Omitting all of these raises ValueError rather than silently
    registering the building at a fixed, wrong location.

    `state_code`/`district_code`: FIX -- ULPIN allocation used to hardcode
    state_code="10", district_code="05" for every building regardless of
    where it actually was (a leftover from single-site testing). Since the
    whole point of this pipeline is that buildings can be anywhere, the
    jurisdiction codes baked into the legal ID must come from an authoritative
    source (the `jurisdiction` argument, the resolver, or an explicit manual
    override), not a constant. No default.

    `floors` / `floor_h`: FIX -- these used to default to 1 and 3.0 and were
    filled in by a human on a web form. That made the Z axis decorative: the
    same footprint was replayed N times at a constant pitch, which is a 2.5D
    extrusion, not a 3D cadastre. Both are now OPTIONAL OVERRIDES. Left as
    None (the normal path) the vertical structure is DERIVED by
    blueprint_metrology.resolve_building_envelope() from:
        - level marks on the drawing ("FFL +3.000") -> measured pitch, and
          one storey per mark, so a double-height podium stays double-height
        - a massing code ("G+3", "2B+G+12") -> explicit storey count
        - floor-panel titles ("SECOND FLOOR PLAN")
        - the LiDAR envelope already indexed by lidar_indexer.py -> measured
          total height, cross-checked against the annotations
    Passing an explicit number still works (surveyor override, regression
    tests), and is logged as an override so it is visible in the audit trail.

    `require_blueprint_evidence`: when True (default) and NO vertical evidence
    is recoverable, the pipeline ABORTS instead of silently inventing a
    single-storey building. This is the mechanism that compels the operator to
    upload a real blueprint sheet: a guessed storey count is not a legal
    record. Set False only for smoke tests.

    `export_model`: FIX -- previously always True. export_postgis_to_glb()
    re-queries and re-exports EVERY property in the whole national ledger
    on every call, so running this pipeline N times back-to-back (e.g.
    from run_batch_cadastre_pipeline below) re-did that full-ledger work
    N times for no benefit. Batch callers pass export_model=False for
    every building except the last.

    `db`: FIX -- previously always opened its own `CadastreDatabaseEngine()`
    (a fresh DB connection) per call. Batch registration of several
    buildings now shares ONE engine/connection across the whole batch
    instead of opening and tearing one down per building.

    `building_label`: optional short label (e.g. "Building 2/5") purely
    for clearer log output when running as part of a batch.
    """
    
    print("\n==================================================")
    print("🚀 RUNNING TRUE 3D VERTICAL CADASTRE PIPELINE")
    print("==================================================")

    print("\n[STEP 0] Calibrating GNSS/CORS Global Coordinate Anchor...")

    if isinstance(location, dict):
        location = PropertyLocation(**location)
    if location is not None and not isinstance(location, PropertyLocation):
        raise ValueError("location must be a PropertyLocation or a dict of its fields")
    if site_lat is not None or site_lon is not None:
        if location is not None:
            raise ValueError("pass either `location` or site_lat/site_lon, not both")
        if site_lat is None or site_lon is None:
            raise ValueError("site_lat and site_lon must be supplied together")
        location = PropertyLocation(latitude=site_lat, longitude=site_lon, source="manual")
    location_meta = location.as_dict() if location is not None else None
    if location is not None:
        print(f"   📍 User location (kept as supplied): lat {location.latitude}, "
              f"lon {location.longitude}, accuracy_m={location.accuracy_m}, "
              f"source={location.source}, crs={location.crs}")

    if gcp_pixels is not None and gcp_real_world is not None:
        # Most accurate path: caller supplies real ground-control points
        # (pixel <-> real-world pairs) measured directly off THIS blueprint.
        # No assumption is made about scale or footprint size at all.
        pixels = np.asarray(gcp_pixels, dtype=float)
        utm_coords = np.asarray(gcp_real_world, dtype=float)
        if pixels.shape[0] < 3 or utm_coords.shape[0] < 3 or pixels.shape[0] != utm_coords.shape[0]:
            raise ValueError("gcp_pixels and gcp_real_world must each supply the same, >=3, matched control points.")
        print(f"   🛰️ Using {pixels.shape[0]} explicit ground-control points supplied for this building.")
    elif location is not None:
        if pixel_scale_m is None:
            raise ValueError(
                "pixel_scale_m (real-world meters represented by one blueprint pixel) "
                "is required alongside the location so this building's footprint "
                "is sized from real data, not an assumed constant. Pass pixel_scale_m, "
                "or pass gcp_pixels/gcp_real_world for exact control-point georeferencing."
            )
        img_probe = cv2.imread(image_path)
        if img_probe is None:
            raise ValueError(f"Could not read '{image_path}' to determine its pixel dimensions for georeferencing.")
        img_h, img_w = img_probe.shape[:2]

        # Project the location into the Cadastre's grid for this computation
        # only; the original WGS84 lat/lon stay on `location`, unmodified.
        base_x, base_y = location.to_projected(CADASTRE_SRID)

        # GCPs derived from the blueprint's ACTUAL pixel dimensions and the
        # caller-supplied real scale -- not a hardcoded 1000px/50m constant.
        pixels = np.array([[0, 0], [img_w, 0], [0, img_h]])
        utm_coords = np.array([
            [base_x, base_y],
            [base_x + img_w * pixel_scale_m, base_y],
            [base_x, base_y - img_h * pixel_scale_m],
        ])
        print(f"   🛰️ Target Locked: Lat {location.latitude}, Lon {location.longitude} -> X:{base_x:.1f}, Y:{base_y:.1f} "
              f"(scale {pixel_scale_m}m/px over a {img_w}x{img_h}px blueprint)")
    else:
        raise ValueError(
            "This building has no georeferencing information. Every building must be "
            "located by real coordinates -- pass `location` (or site_lat & site_lon) "
            "with pixel_scale_m, or gcp_pixels & gcp_real_world. There is no default "
            "site location."
        )

    # Jurisdiction (recorded in every ULPIN): explicit authoritative input,
    # else resolved from the GPS location through the replaceable resolver.
    # Never guessed; fails here, before any LiDAR/AI work or registration.
    if isinstance(jurisdiction, dict):
        jurisdiction = Jurisdiction(**jurisdiction)
    if jurisdiction is not None and not isinstance(jurisdiction, Jurisdiction):
        raise JurisdictionError("jurisdiction must be a Jurisdiction or a dict of its fields")
    if state_code is not None or district_code is not None:
        if jurisdiction is not None:
            raise JurisdictionError("pass either `jurisdiction` or state_code/district_code, not both")
        if state_code is None or district_code is None:
            raise JurisdictionError("state_code and district_code must be supplied together")
        jurisdiction = Jurisdiction(state_code=state_code, district_code=district_code,
                                    source="manual_override")
    if jurisdiction is None:
        if location is None:
            raise JurisdictionError(
                "no jurisdiction: supply `jurisdiction` (or a manual state/district override), "
                "or a GPS `location` so it can be resolved. Nothing is assumed.")
        _resolved = resolve_jurisdiction(location.latitude, location.longitude)
        if isinstance(_resolved, JurisdictionUnavailable):
            raise JurisdictionError(
                f"jurisdiction unavailable for ({location.latitude}, {location.longitude}): "
                f"{_resolved.reason}. No state/district code is guessed; supply an authoritative "
                f"`jurisdiction` or connect a resolver.")
        jurisdiction = _resolved
    state_code, district_code = jurisdiction.state_code, jurisdiction.district_code
    jurisdiction_meta = jurisdiction.as_dict()
    print(f"   🗺️ Jurisdiction: state {state_code}, district {district_code} "
          f"(source: {jurisdiction.source})")

    gnss = GNSSCoordinateAnchor(pixels, utm_coords)
    mat_vals = gnss.matrix

    transform_is_mirror = (mat_vals['a'] * mat_vals['e'] - mat_vals['b'] * mat_vals['d']) < 0
    if transform_is_mirror:
        print("   ⚠️ GCP fit has negative determinant (reflection) — will compensate on WKT export.")

    raw_units = extract_smart_boundaries(image_path)
    # No AI/blueprint boundary: measured LiDAR/XYZ may still establish the evidence. The single unit
    # polygon is then ONLY the georeferenced sheet extent -- the LiDAR acquisition seed and attribution
    # window (see calculate_z_bounds). It is not a building outline and clips nothing; volumes still come
    # solely from measured points. No measured evidence over it -> explicit failure just after STEP 1.5.
    measured_seed_only = False
    if not raw_units:
        _probe = cv2.imread(image_path)
        if _probe is not None:
            _h, _w = _probe.shape[:2]
            raw_units = [{"polygon": Polygon([(0, 0), (_w, 0), (_w, _h), (0, _h)])}]
            measured_seed_only = True
            print("   ℹ️ No AI/blueprint boundaries: seeding LiDAR/XYZ acquisition from the "
                  "georeferenced sheet extent (acquisition window only, not a footprint).")
    _unit_prefix = "LIDAR_Unit" if measured_seed_only else "AI_Unit"
    if not raw_units:
        print("❌ Pipeline Aborted: No valid geometries found.")
        return {"registered_count": 0, "registered_units": [], "failed_units": [], "error": "No valid geometries found in blueprint.", "location": location_meta, "jurisdiction": jurisdiction_meta}

    solids = []
    unit_metadata = []

    # ------------------------------------------------------------------
    # [STEP 1.5] Derive the vertical structure from the drawing itself.
    #
    # calculate_z_bounds() is an expensive LiDAR read, so we run it ONCE per
    # unit here and cache the result -- the metrology pass needs the roof
    # model of the largest footprint (to measure total building height), and
    # STEP 2 below needs the same per-unit values. Note this call is at
    # floor_index=0 (ground): roof_points_xyz and the z_roof metadata come from
    # the WHOLE collected point cloud regardless of floor_index, so one call
    # suffices for those; but its floor_points_xyz element is SPECIFICALLY
    # the ground floor's own slab evidence. Every OTHER measured slab
    # surface -- however many there are, at whatever elevations -- is
    # segmented by _detect_slab_surfaces() in STEP 2 out of the measured XYZ
    # structural cloud (structure_points_xyz) this same call returns; there
    # is no floor_index probing left anywhere in this file.
    # ------------------------------------------------------------------
    print("\n[STEP 1.5] Deriving vertical structure from blueprint evidence...")

    z_cache = {}
    semantic_cache = {}
    orientation_cache = {}
    roof_semantics_cache = {}
    for i, unit in enumerate(raw_units, start=1):
        poly_global = gnss.transform_polygon(unit["polygon"])
        # z_engine.calculate_z_bounds's contract:
        #   (floor_points_xyz, roof_points_xyz, structure_points_xyz, metadata)
        # floor_points_xyz is the ground floor's own measured slab evidence
        # ((N, 3) array or None); roof_points_xyz is the measured roof cloud
        # ((M, 3) array or None) and the only roof geometry evidence;
        # structure_points_xyz is the AUTHORITATIVE measured structural cloud
        # ((K, 3) array or None) that _detect_slab_surfaces segments;
        # metadata["z_roof"] is a diagnostic scalar (or None) and is METADATA
        # ONLY -- it is never used to build geometry here.
        (floor_points_xyz, roof_points_xyz,
         structure_points_xyz, metadata) = calculate_z_bounds(
            poly_global, lidar_path=lidar_path, floor_index=0)
        ground_floor_points_xyz = floor_points_xyz
        z_roof = metadata["z_roof"]
        # Optional authoritative per-point surface labels aligned with
        # structure_points_xyz (None when the source supplies none).
        semantic_cache[i] = metadata.get("structure_semantic_labels")
        # Optional authoritative UP direction (NOT gravity/down; the source
        # must already express it as UP). Orients normals only.
        orientation_cache[i] = metadata.get("orientation_up")
        # Explicit upstream statement that roof_points_xyz is ROOF-semantic
        # ("roof"); anything else/absent = generic surface evidence, not a roof.
        roof_semantics_cache[i] = metadata.get("roof_points_semantics")

        # LiDAR completeness gate (read immediately, from metadata only). The
        # measured arrays are kept untouched for diagnostics, but a cloud whose
        # acquisition/evidence is incomplete (False/True flags) -- or whose
        # completeness is unavailable (None or a missing key, which is NOT
        # proof of completeness) -- must never build cadastral geometry.
        _incomplete = [name for name in ("structure_cloud_incomplete",
                                         "roof_evidence_incomplete",
                                         "floor_evidence_incomplete")
                       if metadata.get(name) is True]
        _extent = metadata.get("acquisition_extent_complete")
        if _extent is False or _incomplete:
            evidence_block = (f"LiDAR acquisition is incomplete "
                              f"(acquisition_extent_complete={_extent}"
                              + (f", {', '.join(_incomplete)}" if _incomplete else "") + ")")
        elif _extent is not True:
            evidence_block = ("LiDAR acquisition completeness is unavailable "
                              f"(acquisition_extent_complete={_extent!r}); not treated as complete")
        else:
            evidence_block = None

        wall_sections_local = unit.get("wall_sections") or []

        # 3D-FIX (slab discovery): the unit's measured point cloud for STEP 2
        # to segment (see _detect_slab_surfaces) is z_engine's own
        # structure_points_xyz, used AS RETURNED. It is the authoritative
        # measured structural cloud: it is never rebuilt here from the
        # ground-floor and roof subsets (which omit every upper slab), never
        # clipped to the blueprint, and never padded. z_engine grows its own
        # acquisition window from the data, so no separate query or footprint
        # buffer is made here. Neighbouring structures that come along are not
        # trimmed -- STEP 2 attributes whole detected surfaces to this unit
        # and drops whole surfaces with no presence over its footprint.
        full_slab_points_xyz = (
            structure_points_xyz
            if structure_points_xyz is not None and len(structure_points_xyz) > 0
            else None
        )

        # ------------------------------------------------------------------
        # STRICT TRUE-3D RULE: 2D blueprint/footprint geometry never
        # creates, trims, positions or defines any 3D roof geometry.
        # `roof_points_xyz` is the sole geometric evidence for the roof; the
        # roof section is derived from it alone by
        # _measured_section_from_points (STEP 2).
        #
        # `poly_top_global` is therefore ONLY an optional, independently
        # traced semantic/reference outline (the AI extractor's own
        # `roof_boundary`), handed downstream as a logged cross-check
        # (`validation_poly`) that can never clip, shape or place a
        # measured point. When the extractor supplied none it stays None:
        # the ground footprint (unit["polygon"]) is NOT substituted, no XY
        # outline is fabricated, and no hull/projection of the roof points
        # is made here. The roof section then comes straight from
        # roof_points_xyz with no 2D cross-check at all.
        # ------------------------------------------------------------------
        poly_top_local = unit.get("roof_boundary")
        if poly_top_local is not None:
            top_boundary_source = "AI-extracted roof_boundary (semantic/reference only)"
            poly_top_global = gnss.transform_polygon(poly_top_local)
        else:
            top_boundary_source = "no independent roof_boundary -- roof derived from roof_points_xyz alone, no 2D cross-check"
            poly_top_global = None

        # A single real, MEASURED scalar for envelope resolution's OCR/
        # LiDAR cross-check (blueprint_metrology needs a ground datum to
        # interpret level marks like "FFL +3.000" against). This is a
        # statistic OF real ground-floor slab evidence, used only for that
        # audit cross-check -- it never feeds a floor/roof SectionProfile
        # (see _measured_section_from_points, which always re-derives Z
        # from the point cloud itself, never from this scalar). None when
        # there's no ground-floor slab evidence at all, rather than
        # inventing a datum.
        z_dem_audit = (
            float(np.median(np.asarray(ground_floor_points_xyz)[:, 2]))
            if ground_floor_points_xyz is not None and len(ground_floor_points_xyz) > 0
            else None
        )

        z_cache[i] = (
            poly_global, poly_top_global, z_dem_audit, z_roof,
            top_boundary_source,
            roof_points_xyz, ground_floor_points_xyz, wall_sections_local,
            full_slab_points_xyz, evidence_block,
        )

    if measured_seed_only and all(v[8] is None for v in z_cache.values()):  # v[8] = measured structure cloud
        print("❌ Pipeline Aborted: no AI/blueprint boundaries and no measured LiDAR/XYZ evidence.")
        return {"registered_count": 0, "registered_units": [], "failed_units": [],
                "error": ("No footprint could be established: AI/blueprint boundary detection found no units "
                          "and no LiDAR/XYZ evidence was found over the sheet's georeferenced extent. "
                          "Nothing is assumed."),
                "location": location_meta, "jurisdiction": jurisdiction_meta}

    # The largest footprint is the building mass; small units are rooms
    # inside it, whose roof fit is noisier and whose extent understates the
    # building height.
    primary_idx = max(z_cache, key=lambda k: z_cache[k][0].area)
    (primary_poly, primary_poly_top, primary_z_dem, primary_z_roof,
     _primary_top_src, _primary_roof_xyz, _primary_floor_xyz, _primary_wall_sections,
     _primary_full_slab_xyz, _primary_evidence_block) = z_cache[primary_idx]

    # FIX: `resolve_building_envelope` used to be called with
    # require_evidence=require_blueprint_evidence unconditionally, which meant
    # it could raise and abort the pipeline BEFORE the code below ever got a
    # chance to apply an explicit floors=/floor_h= override. That made the
    # documented "surveyor override" escape hatch (see the two paragraphs
    # above and the override block just below) unreachable in practice: a
    # caller supplying floors=1, floor_h=3.0 -- e.g. the web UI, whose
    # floor_height/floor_count fields are `required` with defaults, so EVERY
    # submission already carries an override -- would still get aborted with
    # BlueprintMetrologyError on an image with no OCR/LiDAR evidence, because
    # the abort happened before the override was ever consulted.
    #
    # An explicit override IS the vertical evidence for this run (recorded
    # below as storey_count_source="manual_override", confidence 0.5, and
    # logged distinctly from a derived value) -- so evidence should only be
    # required from the drawing/LiDAR when no override was supplied.
    has_manual_override = floors is not None or floor_h is not None
    if primary_z_dem is None:
        print("   ℹ️ No ground-floor slab evidence for the primary footprint -- "
              "passing z_dem=None to envelope resolution rather than a guessed "
              "datum (see z_dem_audit in STEP 1.5).")
    try:
        envelope = resolve_building_envelope(
            image_path=image_path,
            footprint_global=primary_poly,
            z_dem=primary_z_dem,
            z_roof=primary_z_roof,
            roof_model=None,      # z_engine no longer provides a roof model
            roof_features=None,   # ...or polynomial features
            require_evidence=require_blueprint_evidence and not has_manual_override,
        )
    except BlueprintMetrologyError as e:
        print(f"❌ Pipeline Aborted: {e}")
        return {
            "registered_count": 0, "registered_units": [], "failed_units": [],
            "error": str(e),
            "error_code": "INSUFFICIENT_VERTICAL_EVIDENCE",
            "location": location_meta,
            "jurisdiction": jurisdiction_meta,
        }

    print(f"   🏢 {envelope.describe()}")
    for line in envelope.evidence:
        print(f"      - {line}")

    # Explicit caller values still win, but are recorded as an override so the
    # ledger can distinguish a surveyed Z from a typed-in one. 3D-FIX (slab
    # discovery): this override is metadata ONLY (see
    # _manual_override_storey_metadata) and now affects nothing but the
    # audited storey-count/height reported in the ledger below
    # (envelope.storeys / storey_count_source / floor_height_source) --
    # slab discovery in STEP 2 no longer reads envelope.storeys, floors, or
    # floor_h at all; it segments each unit's own measured point cloud
    # directly (see _detect_slab_surfaces).
    override_note = None
    if floors is not None or floor_h is not None:
        eff_floors = floors if floors is not None else envelope.floors_above_ground
        eff_height = floor_h if floor_h is not None else envelope.median_floor_height
        envelope.storeys = _manual_override_storey_metadata(eff_floors, envelope.basements, eff_height)
        override_note = f"manual override: floors={floors}, floor_h={floor_h}"
        envelope.storey_count_source = "manual_override"
        envelope.floor_height_source = "manual_override"
        envelope.confidence = 0.5
        print(f"   ⚠️ {override_note} -- derived values discarded for this run.")

    print(f"\n[STEP 2] Reconstructing true 3D B-Rep volume(s) for "
          f"{len(raw_units)} AI/blueprint-derived unit(s) from their own "
          f"surface evidence...")

    for i, unit in enumerate(raw_units, start=1):
        (poly_global, poly_top_global, z_dem_audit, _z_roof,
         top_boundary_source, roof_points_xyz, ground_floor_points_xyz, wall_sections_local,
         full_slab_points_xyz, evidence_block) = z_cache[i]
        smooth_boundary = bool(unit.get("curved", False))

        # Gate BEFORE any measured geometry reaches wall-section support, slab
        # detection, pairing or volume construction: an incomplete (or
        # completeness-unknown) cloud is not used to reconstruct B-Reps, and
        # nothing is cropped, padded or completed in its place.
        if evidence_block is not None:
            print(f"   ⏭️ AI_Unit_{i}: skipped -- {evidence_block}. The measured points are "
                  f"not used to build geometry; no volume is reconstructed or fabricated.")
            continue

        # Intermediate wall-section evidence (overhang/jetty/corbel/taper)
        # transformed once per unit; it is only applied to a volume that spans
        # the whole footprint and whose real measured Z range contains it.
        mid_sections_global = sorted(
            (_transform_wall_section_to_global(gnss, s) for s in wall_sections_local),
            key=lambda s: s.level,
        )
        # Wall sections are AI/blueprint-traced, not measured: only those
        # backed by measured XYZ may reach the loft.
        mid_sections_global, _unsupported = _measured_wall_sections(
            mid_sections_global, full_slab_points_xyz,
            semantic_labels=(semantic_cache.get(i) if full_slab_points_xyz is not None else None))
        if _unsupported:
            print(f"   ℹ️ AI_Unit_{i}: {_unsupported} AI/blueprint wall section(s) have no "
                  f"measured wall-surface support (>= {WALL_SECTION_MIN_SUPPORTED_FRACTION:.0%} of vertices "
                  f"within {WALL_SECTION_SUPPORT_TOL_M} m of ONE measured wall surface patch) -- not used "
                  f"as geometry.")

        # ------------------------------------------------------------------
        # 3D-FIX (slab discovery): floor-volume TOPOLOGY comes from measured
        # surfaces, not from storey order, floor count, or floor height, and
        # not from a probing grid of candidate labels either.
        # _detect_slab_surfaces() segments THIS unit's own full measured
        # point cloud into distinct surfaces with no plane fitting anywhere
        # (3D-FIX ROUND 5): a per-point neighbourhood DIMENSIONALITY test
        # keeps whatever is shaped like a surface and drops linear clutter
        # and volumetric scatter (vegetation, noise); smooth patches are
        # then grown by spatial adjacency plus NORMAL CONTINUITY between
        # neighbours, so a patch may turn through any total angle as long
        # as it turns gradually; and only then does a WHOLE patch receive a
        # role -- from authoritative per-point semantic labels ONLY, never
        # from Z, normals, orientation, height, XY overlap or proximity.
        # A patch without such a label keeps role None: it stays a raw
        # measured candidate surface (its XYZ can still evidence a blueprint
        # wall section by measured correspondence in
        # _measured_wall_sections, which never promotes it to "wall"), but
        # it is NOT a slab and is never paired as floor/ceiling. Full-plan floors, tilted decks,
        # vaults, domes, warped and folded roofs, mezzanines, split-levels
        # and partial slabs all come out the same way (detected surfaces are kept
        # whole, but only authoritatively slab-labelled ones may later be paired as
        # floor/ceiling -- see _pair_floor_volumes), each keyed
        # by its own measured median elevation and returned as its own
        # original measured points. _pair_floor_volumes()
        # then establishes which surface is actually over which by casting
        # rays against real B-Rep patches built from those points (3D-FIX
        # ROUND 4). A surface with no measured evidence simply never appears
        # in this dict; nothing is filled in from a neighbour, a floor count,
        # or an assumed height.
        # ------------------------------------------------------------------
        # surface_roles[label] is "slab" (authoritative label) or None
        # (unlabelled): the dict below holds ALL measured candidate surfaces,
        # not only slabs.
        surface_roles = {}
        measured_surface_points = _detect_slab_surfaces(
            full_slab_points_xyz, roles_out=surface_roles,
            semantic_labels=(semantic_cache.get(i) if full_slab_points_xyz is not None else None))

        # ATTRIBUTION, NOT CLIPPING. The cloud was queried with an overhang
        # margin (see STEP 1.5), so it can contain surfaces belonging to a
        # neighbouring structure. A detected surface is kept when any of its
        # measured points sits over this unit's footprint, and when it is
        # kept it is kept WHOLE -- including every point that oversails the
        # blueprint, which is exactly the cantilever/balcony/eave evidence
        # this pass exists to protect. A surface with no presence at all over
        # the footprint is not this unit's and is dropped entire; no surface
        # is ever cut along the footprint's edge.
        attributed = {}
        for label, pts in measured_surface_points.items():
            if len(_points_inside(poly_global, np.asarray(pts, dtype=float))) > 0:
                attributed[label] = pts
        dropped = len(measured_surface_points) - len(attributed)
        if dropped:
            print(f"   ℹ️ AI_Unit_{i}: {dropped} detected surface(s) lie entirely outside "
                  f"this unit's footprint (neighbouring structure or ground) -- not "
                  f"attributed to it. Surfaces that ARE attributed keep all their "
                  f"points, including those beyond the blueprint outline.")
        measured_surface_points = attributed

        if not measured_surface_points:
            print(f"⚠️ Skipping AI_Unit_{i}: no measured surfaces detected "
                  f"in this unit's point cloud -- nothing assumed from floor "
                  f"count/height.")
            continue
        _n_slab_role = sum(1 for l in measured_surface_points if surface_roles.get(l) == "slab")
        print(f"   ℹ️ AI_Unit_{i}: {len(measured_surface_points)} measured surface(s) "
              f"detected at elevation(s) {sorted(measured_surface_points.keys())} "
              f"({_n_slab_role} authoritatively slab-labelled, "
              f"{len(measured_surface_points) - _n_slab_role} unlabelled/role=None -- "
              f"unlabelled ones are never used as floor/ceiling).")

        # tier is reporting/audit metadata only -- it never feeds geometry.
        # Derived purely from each detected surface's own measured elevation
        # against this unit's own measured ground-slab datum (z_dem_audit,
        # from STEP 1.5) -- never from a declared storey, floor count, or
        # floor_h. With no ground evidence at all, tier is honestly unknown
        # rather than guessed.
        tier_by_label = {}
        for label in measured_surface_points:
            if z_dem_audit is None:
                tier_by_label[label] = "UNKNOWN_TIER"
            elif label < z_dem_audit - 1e-6:
                tier_by_label[label] = "BASEMENT"
            else:
                tier_by_label[label] = "ABOVE_GROUND"

        if roof_semantics_cache.get(i) == "roof":
            surface_roles["roof"] = "roof"  # explicit roof semantics only; never inferred
        if semantic_cache.get(i) is None:
            print(f"   ℹ️ AI_Unit_{i}: authoritative surface-role evidence is missing (no semantic "
                  f"labels for the measured cloud) -- measured surfaces are left unresolved; no "
                  f"floor/ceiling role is inferred from Z, orientation, overlap or proximity.")
        volumes, topology_notes = _pair_floor_volumes(
            poly_global, measured_surface_points, roof_points_xyz, orientation_up=orientation_cache.get(i),
            surface_roles=surface_roles)
        for note in topology_notes:
            print(f"   ℹ️ AI_Unit_{i}: {note}")
        if not volumes:
            print(f"⚠️ Skipping AI_Unit_{i}: no floor volume has both a measured bottom "
                  f"and a measured ceiling -- nothing assumed from floor count/height.")
            continue

        # Naming ordinals (Floor_1, Floor_2, ... / Basement_1, Basement_2,
        # ...) derived purely from the actual bottoms this unit's own
        # evidence produced, ranked by their own measured elevation --
        # never from floor count/height. Ground-most basement is Basement_1;
        # lowest above-ground bottom is Floor_1, matching the historical
        # convention export_ledger.py colour-codes on.
        bottom_labels = sorted({v["bottom"] for v in volumes})
        above_ground_labels = [l for l in bottom_labels if tier_by_label[l] != "BASEMENT"]
        basement_labels = sorted((l for l in bottom_labels if tier_by_label[l] == "BASEMENT"), reverse=True)
        floor_ordinal = {l: idx + 1 for idx, l in enumerate(above_ground_labels)}
        basement_ordinal = {l: idx + 1 for idx, l in enumerate(basement_labels)}

        parts_per_bottom = {}
        for v in volumes:
            parts_per_bottom[v["bottom"]] = parts_per_bottom.get(v["bottom"], 0) + 1
        part_index = {}

        for v in volumes:
            floor_level = v["bottom"]
            tier = tier_by_label[floor_level]
            region = v["region"]
            ceiling = v["ceiling"]

            bottom_source = f"floor {floor_level}'s own measured slab"
            if ceiling == "roof":
                top_source = f"roof point cloud ({top_boundary_source} for cross-check only)"
            else:
                top_source = f"floor {ceiling}'s own measured slab"

            # 3D-FIX ROUND 4: these polygons are handed over as
            # `validation_poly` -- a cross-check that gets logged -- and can no
            # longer clip a single measured point. The bottom/ceiling evidence
            # is already the measured subset `_pair_floor_volumes` established
            # by ray-casting against real B-Rep surface patches, so a partial
            # slab, a mezzanine lip, or a cantilever arrives carrying exactly
            # its own points and needs no outline intersected in afterwards.
            # `poly_top_global` in particular is an independently traced
            # semantic/reference outline (or None when the extractor gave
            # none), not measurement, so it is at most a cross-check for the
            # roof section and never its shape. It is never backfilled from
            # the ground footprint.
            top_validation = region
            if ceiling == "roof" and v["full"]:
                top_validation = poly_top_global  # may be None: no cross-check
            bottom_section = _measured_section_from_points(
                v["bottom_points"], smooth=smooth_boundary, validation_poly=region)
            top_section = _measured_section_from_points(
                v["ceiling_points"], smooth=smooth_boundary, validation_poly=top_validation)
            if bottom_section is None or top_section is None:
                print(f"⚠️ Skipping AI_Unit_{i} floor {floor_level}: bottom/ceiling section "
                      f"could not be fit from its measured points.")
                continue

            if v["full"]:
                floor_mid_sections = _wall_sections_for_volume(
                    mid_sections_global, v["bottom_points"], v["ceiling_points"])
            else:
                # Wall-section rings are traced for the whole footprint; they
                # can't be assumed to apply to a partial region.
                floor_mid_sections = []
            if not floor_mid_sections:
                print(f"⚠️ Skipping AI_Unit_{i} floor {floor_level}: no measured intermediate wall "
                      f"section bracketed by this volume -- lateral boundary is not measured, "
                      f"no wall is fabricated between its bottom and ceiling.")
                continue
            sections = [bottom_section] + floor_mid_sections + [top_section]
            wall_evidence_note = (f"{len(floor_mid_sections)} measured-supported intermediate wall section(s) "
                                  f"bracketed by this volume's own measured bottom/ceiling")

            try:
                floor_solid = construct_unit_volume(sections, smooth=smooth_boundary)
            except Exception as e:
                print(f"⚠️ Failed to reconstruct AI_Unit_{i} floor {floor_level}: {e}")
                continue

            if tier == "BASEMENT":
                unit_id = f"{_unit_prefix}_{i}_Basement_{basement_ordinal[floor_level]}"
            else:
                # Keep the historical "_Floor_N" naming (ground = Floor_1);
                # export_ledger.py colour-codes on that substring.
                unit_id = f"{_unit_prefix}_{i}_Floor_{floor_ordinal[floor_level]}"
            if parts_per_bottom[floor_level] > 1:
                # One slab paired with several ceilings (mezzanine, split-level,
                # disjoint region): each piece needs its own ledger id.
                part_index[floor_level] = part_index.get(floor_level, 0) + 1
                unit_id = f"{unit_id}_P{part_index[floor_level]}"

            BRepMesh_IncrementalMesh(floor_solid, 2.0)

            solids.append(floor_solid)
            unit_metadata.append({
                "id": unit_id,
                # A real measured statistic of THIS volume's own bottom
                # evidence -- not a z_lo derived from base_offset/height.
                "z_base": bottom_section.level,
                "floor_level": floor_level,
                "tier": tier,
                "mirror": transform_is_mirror,
                "ceiling": ceiling,
                "full_footprint": v["full"],
                "spans_unmeasured_labels": v["spans_unmeasured"],
            })
            clearance_note = (f", measured clearance ~{v['clearance']:.2f}m along the slab's "
                              f"own normal" if v.get("clearance") is not None else "")
            span_note = (f", spans floor label(s) {v['spans_unmeasured']} with NO measured slab "
                         f"(read as open space -- a LiDAR dropout there would look identical)"
                         if v["spans_unmeasured"] else "")
            print(f"   🏢 Reconstructed: {unit_id} (bottom: {bottom_source} "
                  f"@ ~{bottom_section.level:.2f}m, top: {top_source} "
                  f"@ ~{top_section.level:.2f}m{clearance_note}, "
                  f"walls: {wall_evidence_note}{span_note})")

    if include_demo_subsurface:
        # DISABLED (strict 3D). The former demo built a tunnel from hardcoded
        # synthetic waypoints and a fixed radius -- fabricated XYZ, not
        # measured evidence -- and appended it to `solids`. No geometry is
        # generated here; the flag is accepted only for caller compatibility.
        print("\n[STEP 3] include_demo_subsurface ignored: synthetic subsurface "
              "geometry (hardcoded waypoints/radius) is disabled; only geometry "
              "from measured 3D evidence is registered.")

    label_prefix = f"[{building_label}] " if building_label else ""
    print(f"\n[STEP 4] {label_prefix}Registering into National PostGIS Cadastre Ledger...")
    registered_count = 0
    registered_units = []  # FIX: previously nothing was returned from this function,
    # so api.py had no way to tell the caller which ULPIN(s) their upload actually
    # produced -- the frontend needs this list to show the user their new ULPIN(s).
    failed_units = []

    # FIX: a caller running several buildings in one batch (see
    # run_batch_cadastre_pipeline) passes its own already-open `db` so
    # every building in the batch shares one connection/transaction
    # sequence instead of opening a new one each time. A single-building
    # caller (the "if __name__" block below, or api.py's single-upload
    # endpoint) leaves `db=None` and gets the original self-contained
    # connection lifecycle.
    owns_db = db is None
    if owns_db:
        db = CadastreDatabaseEngine()
    try:
        for idx, solid in enumerate(solids):
            if idx >= len(unit_metadata):
                break
            unit_info = unit_metadata[idx]

            wkt_string = robust_solid_to_wkt(solid, global_mirror=unit_info.get("mirror", False))

            if wkt_string and "()" not in wkt_string:
                ulpin = db.register_property(
                    unit_id=unit_info["id"],
                    ogc_3d_wkt=wkt_string,
                    state_code=state_code,
                    district_code=district_code,
                    tier_type=unit_info["tier"],
                    floor_level=unit_info["floor_level"],
                    is_demolition=is_demolition,
                    # User-provided location: metadata/reference only, stored
                    # verbatim; never used for or derived into geometry.
                    location_latitude=location_meta["latitude"] if location_meta else None,
                    location_longitude=location_meta["longitude"] if location_meta else None,
                    location_accuracy_m=location_meta["accuracy_m"] if location_meta else None,
                    location_source=location_meta["source"] if location_meta else None,
                    location_crs=location_meta["crs"] if location_meta else None,
                )
                if ulpin:
                    registered_count += 1
                    registered_units.append({
                        "ulpin": ulpin,
                        "unit_id": unit_info["id"],
                        "tier": unit_info["tier"],
                        "floor_level": unit_info["floor_level"],
                    })
                else:
                    # A None return means register_property() rejected the
                    # unit -- most commonly a volumetric clash against an
                    # already-registered building (this one or another one
                    # entirely; the clash check is global, not per-building).
                    failed_units.append(unit_info["id"])
            else:
                failed_units.append(unit_info["id"])
    finally:
        if owns_db:
            db.close()

    print(f"\n🎉 {label_prefix}Successfully registered {registered_count} Multi-Tier Georeferenced 3D assets.")

    if export_model:
        print("\n[STEP 5] Exporting WebGL 3D Model (.glb)...")
        export_postgis_to_glb(output_filename="approved_cadastre.glb")
    print(f"✨ {label_prefix}Pipeline execution complete!")

    return {
        "registered_count": registered_count,
        "registered_units": registered_units,
        "failed_units": failed_units,
        # Number of B-Rep solids actually reconstructed. 0 means reconstruction
        # produced nothing, so validation / clash / registration never had input.
        "brep_count": len(solids),
        # The user-provided location, exactly as supplied (or None when the
        # building was georeferenced by GCPs alone).
        "location": location_meta,
        # State/district codes recorded in the ULPINs, with their provenance.
        "jurisdiction": jurisdiction_meta,
        # Provenance for the Z axis. api.py should surface this so the user
        # sees WHERE the storey count came from and how confident it is,
        # instead of being asked to supply it.
        "vertical_model": {
            "storeys": len(envelope.storeys),
            "floors_above_ground": envelope.floors_above_ground,
            "basements": envelope.basements,
            "median_floor_height": round(envelope.median_floor_height, 3),
            "total_height": round(envelope.total_height, 3),
            "storey_count_source": envelope.storey_count_source,
            "floor_height_source": envelope.floor_height_source,
            "confidence": envelope.confidence,
            "lidar_total_height": envelope.lidar_total_height,
            "evidence": envelope.evidence,
        },
    }


def run_batch_cadastre_pipeline(jobs):
    """
    Registers MULTIPLE buildings -- each with its own blueprint image and
    its own site_lat/site_lon -- in a single call.

    `jobs`: list of dicts, each accepting the same keyword arguments as
    run_unified_cadastre_pipeline (image_path, site_lat, site_lon,
    state_code, district_code, pixel_scale_m or gcp_pixels/gcp_real_world,
    floor_h, floors, is_demolition, lidar_path, include_demo_subsurface),
    plus an optional "label" used only for logging (e.g. a filename or
    building name the caller supplied). Each building in the batch carries
    ITS OWN location and jurisdiction -- there is no shared/default site, so
    buildings in the same batch can be anywhere. A job's GPS location
    (`location` or site_lat/site_lon) is sufficient: its jurisdiction is
    resolved from the SOI district dataset unless the job supplies an explicit
    `jurisdiction` or a manual state_code/district_code override.

    Overlap handling: NOTHING special is needed here for cross-building
    clash detection -- CadastreDatabaseEngine.register_property() already
    checks every incoming 3D volume against the ENTIRE property_spatial_shards
    table (every building, every floor, every tier, previously registered
    OR registered earlier in this same batch), using exact SFCGAL volume
    intersection, serialized against concurrent registrations elsewhere
    (see db_engine.py). Building 2 in this batch will be rejected (or, if
    `is_demolition` is set, will clear the clash) exactly the same way it
    would be if Building 1 had been registered in a completely separate
    request yesterday.

    All buildings in the batch share ONE CadastreDatabaseEngine so we
    don't pay a fresh-connection-plus-schema-check cost per building, and
    the .glb model is exported exactly ONCE at the end of the whole batch
    (each building always sees the fully up-to-date ledger while
    registering, regardless of export timing, since export is a pure
    read of the DB that happens after all registrations finish).
    """
    print("\n==================================================")
    print(f"🏙️  BATCH REGISTRATION: {len(jobs)} building(s)")
    print("==================================================")

    db = CadastreDatabaseEngine()
    results = []
    try:
        for i, job in enumerate(jobs, start=1):
            label = job.get("label") or f"Building {i}/{len(jobs)}"
            print(f"\n--- {label} ---")
            try:
                result = run_unified_cadastre_pipeline(
                    image_path=job["image_path"],
                    site_lat=job.get("site_lat"),
                    site_lon=job.get("site_lon"),
                    location=job.get("location"),
                    state_code=job.get("state_code"),
                    district_code=job.get("district_code"),
                    jurisdiction=job.get("jurisdiction"),
                    pixel_scale_m=job.get("pixel_scale_m"),
                    gcp_pixels=job.get("gcp_pixels"),
                    gcp_real_world=job.get("gcp_real_world"),
                    # FIX: these used to default to 3.0 / 1 here, which meant
                    # a batch job that simply omitted them got a silently
                    # invented single-storey building. None now means "derive
                    # from the blueprint", matching the single-building path.
                    floor_h=job.get("floor_h"),
                    floors=job.get("floors"),
                    require_blueprint_evidence=job.get("require_blueprint_evidence", True),
                    is_demolition=job.get("is_demolition", False),
                    lidar_path=job.get("lidar_path"),
                    include_demo_subsurface=job.get("include_demo_subsurface", False),
                    export_model=False,
                    db=db,
                    building_label=label,
                )
            except Exception as e:
                print(f"❌ {label} failed with an unhandled error: {e}")
                result = {"registered_count": 0, "registered_units": [], "failed_units": [],
                          "error": str(e)}
            result["label"] = label
            results.append(result)
    finally:
        db.close()

    print("\n[BATCH STEP] Exporting WebGL 3D Model (.glb) for the full ledger...")
    export_postgis_to_glb(output_filename="approved_cadastre.glb")

    total_registered = sum(r.get("registered_count", 0) for r in results)
    print(f"\n🎉 BATCH COMPLETE: {total_registered} unit(s) registered across {len(jobs)} building(s).")

    # None when no building reported a B-Rep count (e.g. every job errored out early).
    brep_counts = [r["brep_count"] for r in results if "brep_count" in r]
    return {
        "total_registered_count": total_registered,
        "total_brep_count": sum(brep_counts) if brep_counts else None,
        "buildings": results,
    }


if __name__ == "__main__":
    import argparse
    import json

    # Manual entry point. Location and georeferencing are user inputs given
    # on the command line (or as GCPs in a JSON file) -- not environment
    # variables, and there are no defaults: missing evidence is an error.
    ap = argparse.ArgumentParser(
        description="Register a building in the 3D cadastre. With --lat/--lon the "
                    "jurisdiction is resolved automatically from the SOI district dataset "
                    "(set SOI_DISTRICT_BOUNDARY_SHP); no state/district codes are needed.")
    ap.add_argument("--image", default="drone_sample.jpg", help="blueprint image")
    ap.add_argument("--lat", type=float, help="property latitude (decimal degrees)")
    ap.add_argument("--lon", type=float, help="property longitude (decimal degrees)")
    ap.add_argument("--accuracy-m", type=float, help="optional horizontal accuracy in metres")
    ap.add_argument("--location-source", choices=LOCATION_SOURCES,
                    help="provenance of the location fix (required with --lat/--lon)")
    ap.add_argument("--crs", default="EPSG:4326", help="location CRS (EPSG:4326 only)")
    ap.add_argument("--pixel-scale-m", type=float,
                    help="metres per blueprint pixel (required with --lat/--lon)")
    ap.add_argument("--gcp-json",
                    help='JSON file {"gcp_pixels": [[x,y],...], "gcp_real_world": [[X,Y],...]}')
    ap.add_argument("--state-code",
                    help="MANUAL OVERRIDE (admin/controlled use): authoritative state code; "
                         "normally resolved automatically from the GPS location")
    ap.add_argument("--district-code",
                    help="MANUAL OVERRIDE (admin/controlled use): authoritative district code; "
                         "must be given with --state-code")
    args = ap.parse_args()

    if not os.path.exists(args.image):
        ap.error(f"input image '{args.image}' not found.")
    if bool(args.state_code) != bool(args.district_code):
        ap.error("--state-code and --district-code must be given together (manual override).")

    gcp_pixels = gcp_real_world = None
    if args.gcp_json:
        with open(args.gcp_json, "r", encoding="utf-8") as fh:
            gcp = json.load(fh)
        gcp_pixels, gcp_real_world = gcp.get("gcp_pixels"), gcp.get("gcp_real_world")

    user_location = None
    if args.lat is not None or args.lon is not None:
        if args.lat is None or args.lon is None:
            ap.error("--lat and --lon must be given together.")
        if not args.location_source:
            ap.error("--location-source is required with --lat/--lon.")
        try:
            user_location = PropertyLocation(
                latitude=args.lat, longitude=args.lon, source=args.location_source,
                accuracy_m=args.accuracy_m, crs=args.crs)
        except ValueError as e:
            ap.error(str(e))

    if user_location is None and gcp_pixels is None:
        ap.error("no georeferencing evidence: supply the property location "
                 "(--lat, --lon, --location-source, --pixel-scale-m) or ground-control "
                 "points (--gcp-json). No location is assumed or invented.")
    if user_location is not None and gcp_pixels is None and args.pixel_scale_m is None:
        ap.error("--pixel-scale-m is required with --lat/--lon (or supply --gcp-json).")

    try:
        run_unified_cadastre_pipeline(
            args.image,
            location=user_location,
            pixel_scale_m=args.pixel_scale_m,
            gcp_pixels=gcp_pixels,
            gcp_real_world=gcp_real_world,
            state_code=args.state_code,
            district_code=args.district_code,
        )
    except JurisdictionError as e:
        ap.error(str(e))