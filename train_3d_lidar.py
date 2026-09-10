#!/usr/bin/env python3
"""
train_3d_lidar.py  –  SIH 26011 · 3D Cadastre Z-Axis Elevation Engine
══════════════════════════════════════════════════════════════════════
Processes .las / .laz LiDAR point clouds to extract building heights,
floor-slab elevations, and roof surfaces for conversion into OGC 3D
geometries (POLYHEDRALSURFACE Z, TIN Z).

Pipeline overview
─────────────────
  1.  INGEST    – Read a .las file via laspy; separate Building vs
                  Ground classified points (ASPRS classes 2 & 6).
  2.  FLOOR     – Build a 1-D vertical (Z-axis) point-density
                  histogram and run peak detection to locate the
                  physical Z-elevation of each concrete floor slab.
  3.  ROOF      – Extract the topmost point-cloud slice and:
                    a) Fit a plane via RANSAC  → Ax + By + Cz + D = 0
                       (for standard pitched / flat roofs).
                    b) If the RANSAC residual is too high (curved /
                       organic roof), fall back to Delaunay
                       triangulation → 2.5-D TIN surface.
  4.  OUTPUT    – Return structured results with Z_min, Z_max, floor
                  elevations, roof equation, and OGC-ready WKT
                  fragments (POLYHEDRALSURFACE Z / TIN Z).

Requirements
────────────
  pip install laspy[lazrs] numpy scikit-learn scipy shapely

Usage
─────
  # Process a single .las / .laz file:
  python train_3d_lidar.py --las data/raw_lidar/san_francisco_3dep_sample.laz

  # Override histogram resolution (metres):
  python train_3d_lidar.py --las data/raw_lidar/sample.las --bin-size 0.15

  # Process all .las/.laz files in a directory:
  python train_3d_lidar.py --dir data/raw_lidar/

  # Generate OGC WKT output only (skip the plots):
  python train_3d_lidar.py --las sample.las --no-plot
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import laspy
import numpy as np
from scipy.signal import find_peaks
from scipy.spatial import Delaunay
from shapely.geometry import Polygon as ShapelyPolygon
from sklearn.linear_model import RANSACRegressor

# ────────────────────────────────────────────────────────────────────
# Logging
# ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-7s │ %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("train_3d_lidar")


# ════════════════════════════════════════════════════════════════════
# §1  CONSTANTS & CONFIGURATION
# ════════════════════════════════════════════════════════════════════

# ── ASPRS LAS point classification codes ─────────────────────────
# Standard codes used by virtually all airborne LiDAR vendors.
# See: https://www.asprs.org/wp-content/uploads/2019/03/LAS_1_4_r15.pdf
ASPRS_GROUND: int   = 2     # Class 2 = Ground
ASPRS_BUILDING: int = 6     # Class 6 = Building

# ── Floor slab detection ─────────────────────────────────────────
# Histogram bin width (metres).  0.20 m ≈ the thickness of a
# standard reinforced-concrete slab, so peaks in the histogram
# correspond to physical floor levels.
DEFAULT_BIN_SIZE_M: float = 0.20

# Minimum peak prominence (fraction of the tallest peak).
# Prevents noise from being classified as a floor slab.
PEAK_PROMINENCE_FRAC: float = 0.10

# Minimum vertical distance (metres) between two consecutive
# detected floor slabs.  A typical storey is ≥ 2.5 m.
MIN_FLOOR_SPACING_M: float = 2.0

# ── Roof extraction ──────────────────────────────────────────────
# Thickness (metres) of the "roof slice" taken from the top of the
# building point cloud.  Only points within [Z_max − slice, Z_max]
# are used for plane fitting.
ROOF_SLICE_THICKNESS_M: float = 1.0

# RANSAC parameters for plane fitting.
RANSAC_RESIDUAL_THRESHOLD_M: float = 0.15   # inlier distance (metres)
RANSAC_MAX_TRIALS: int             = 1000

# If the mean absolute RANSAC residual exceeds this value (metres),
# the roof is deemed "curved / organic" and we fall back to a TIN.
CURVED_ROOF_RESIDUAL_THRESHOLD_M: float = 0.30


# ════════════════════════════════════════════════════════════════════
# §2  DATA STRUCTURES
# ════════════════════════════════════════════════════════════════════

@dataclass
class FloorSlabResult:
    """Result of the floor-slab detection stage."""
    z_min: float                      # lowest building point (metres)
    z_max: float                      # highest building point (metres)
    ground_elevation: float           # median ground elevation (metres)
    building_height: float            # z_max − ground_elevation (metres)
    num_floors_detected: int          # count of detected slab peaks
    floor_elevations: list[float]     # Z values of each slab (metres)
    histogram_bins: np.ndarray = field(repr=False)   # bin edges
    histogram_counts: np.ndarray = field(repr=False) # counts per bin


@dataclass
class RoofSurfaceResult:
    """Result of the roof-surface extraction stage."""
    # RANSAC plane: Ax + By + Cz + D = 0
    plane_coefficients: tuple[float, float, float, float] | None  # (A,B,C,D)
    pitch_angle_deg: float | None      # angle between plane normal & vertical
    is_curved: bool                    # True → TIN fallback was used
    ransac_mean_residual: float | None # mean |residual| in metres
    tin_vertices: np.ndarray | None    # (M, 3) array of TIN vertex coords
    tin_simplices: np.ndarray | None   # (T, 3) Delaunay triangle indices
    ogc_wkt: str                       # OGC WKT fragment


@dataclass
class BuildingExtractionResult:
    """Complete output of the Z-axis elevation pipeline."""
    las_path: str
    num_ground_points: int
    num_building_points: int
    floors: FloorSlabResult
    roof: RoofSurfaceResult


# ════════════════════════════════════════════════════════════════════
# §3  LAS FILE INGESTION & POINT CLASSIFICATION FILTER
# ════════════════════════════════════════════════════════════════════

def load_and_filter_las(
    las_path: Path,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Read a .las / .laz file and separate Building vs Ground points.

    Parameters
    ----------
    las_path : Path
        Path to the input LAS/LAZ file.

    Returns
    -------
    ground_xyz : ndarray, shape (N_g, 3)
        XYZ coordinates of points classified as Ground (ASPRS 2).
    building_xyz : ndarray, shape (N_b, 3)
        XYZ coordinates of points classified as Building (ASPRS 6).

    Raises
    ------
    FileNotFoundError
        If *las_path* does not exist.
    ValueError
        If the file contains zero building points (nothing to process).

    Notes
    -----
    ASPRS classification is embedded in the point record's
    ``classification`` field.  Vendors sometimes use non-standard
    codes; this function logs a warning if many points are
    unclassified (class 0 or 1).
    """
    if not las_path.exists():
        raise FileNotFoundError(f"LAS file not found: {las_path}")

    log.info("═══ Loading LAS file ═══")
    log.info("  path: %s", las_path)

    # ── Read the file ────────────────────────────────────────────
    las = laspy.read(str(las_path))
    total_points = len(las.points)
    log.info("  total points: %s", f"{total_points:,}")

    # ── Extract X, Y, Z as a contiguous float64 array ───────────
    # laspy applies the scale + offset automatically when you
    # access .x, .y, .z (they are ScaledArrayView objects).
    x = np.array(las.x, dtype=np.float64)
    y = np.array(las.y, dtype=np.float64)
    z = np.array(las.z, dtype=np.float64)
    classifications = np.array(las.classification, dtype=np.uint8)

    # ── Log class distribution (top 5) ───────────────────────────
    unique_classes, counts = np.unique(classifications, return_counts=True)
    sorted_idx = np.argsort(-counts)
    log.info("  Classification distribution (top classes):")
    for rank, ci in enumerate(sorted_idx[:5]):
        pct = 100.0 * counts[ci] / total_points
        log.info("    class %2d : %10s pts  (%.1f%%)", unique_classes[ci], f"{counts[ci]:,}", pct)

    # ── Filter by ASPRS class ────────────────────────────────────
    ground_mask   = classifications == ASPRS_GROUND
    building_mask = classifications == ASPRS_BUILDING

    ground_xyz   = np.column_stack((x[ground_mask],   y[ground_mask],   z[ground_mask]))
    building_xyz = np.column_stack((x[building_mask], y[building_mask], z[building_mask]))

    log.info("  Ground   (class %d): %s pts", ASPRS_GROUND,   f"{len(ground_xyz):,}")
    log.info("  Building (class %d): %s pts", ASPRS_BUILDING, f"{len(building_xyz):,}")

    # ── Warn if very few classified points ───────────────────────
    unclassified_count = np.sum((classifications == 0) | (classifications == 1))
    if unclassified_count > 0.5 * total_points:
        log.warning(
            "  ⚠️  %.0f%% of points are unclassified (class 0/1). "
            "The LAS file may need pre-processing with PDAL or LAStools.",
            100.0 * unclassified_count / total_points,
        )

    if len(building_xyz) == 0:
        raise ValueError(
            f"No building points (ASPRS class {ASPRS_BUILDING}) found in "
            f"{las_path.name}.  Run a classification tool (e.g., PDAL "
            f"'filters.smrf' + 'filters.hag') first, or verify the file."
        )

    return ground_xyz, building_xyz


# ════════════════════════════════════════════════════════════════════
# §4  FLOOR SLAB DETECTION (1-D Z-AXIS HISTOGRAM + PEAK FINDING)
# ════════════════════════════════════════════════════════════════════

def detect_floor_slabs(
    ground_xyz: np.ndarray,
    building_xyz: np.ndarray,
    bin_size_m: float = DEFAULT_BIN_SIZE_M,
) -> FloorSlabResult:
    """
    Detect physical floor-slab elevations from a building's Z-axis
    point distribution.

    Algorithm
    ---------
    1. Compute a 1-D histogram of all building Z-coordinates, binned
       at ``bin_size_m`` intervals.
    2. Run ``scipy.signal.find_peaks`` on the histogram to locate
       local maxima.
    3. Filter peaks by:
         a) Minimum prominence  (≥ ``PEAK_PROMINENCE_FRAC`` × max peak)
         b) Minimum spacing     (≥ ``MIN_FLOOR_SPACING_M``)
    4. Each surviving peak corresponds to a concrete floor slab whose
       elevation is the centre of that histogram bin.

    Why this works
    ──────────────
    Airborne LiDAR returns dense clusters of points at hard horizontal
    surfaces (ground, concrete slabs, roof decks).  These show up as
    sharp peaks in the vertical histogram.  Walls produce a diffuse
    spread between peaks.

    Parameters
    ----------
    ground_xyz : ndarray, shape (N_g, 3)
        Ground-classified points (used to establish base elevation).
    building_xyz : ndarray, shape (N_b, 3)
        Building-classified points.
    bin_size_m : float
        Histogram bin width in metres.

    Returns
    -------
    FloorSlabResult
        Dataclass containing Z_min, Z_max, detected floor elevations,
        and the raw histogram data (for optional plotting).
    """
    log.info("═══ Floor Slab Detection ═══")

    # ── Ground reference elevation ───────────────────────────────
    # Use the *median* Z of ground points as the reference datum.
    # Median is more robust than mean against outliers (e.g., water
    # returns, noise below terrain).
    ground_z = ground_xyz[:, 2] if len(ground_xyz) > 0 else np.array([0.0])
    ground_elevation = float(np.median(ground_z))
    log.info("  Ground elevation (median): %.2f m", ground_elevation)

    # ── Building Z-extent ────────────────────────────────────────
    bldg_z = building_xyz[:, 2]
    z_min  = float(np.min(bldg_z))
    z_max  = float(np.max(bldg_z))
    building_height = z_max - ground_elevation
    log.info("  Building Z range: %.2f → %.2f m  (height ≈ %.1f m)",
             z_min, z_max, building_height)

    # ── 1-D vertical histogram ───────────────────────────────────
    # Bin the Z-coordinates into slices of `bin_size_m` thickness.
    num_bins = max(int(np.ceil((z_max - z_min) / bin_size_m)), 1)
    counts, bin_edges = np.histogram(bldg_z, bins=num_bins, range=(z_min, z_max))

    # Bin centres (the Z elevation each bin represents)
    bin_centres = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    log.info("  Histogram: %d bins × %.2f m", num_bins, bin_size_m)

    # ── Peak detection ───────────────────────────────────────────
    # `find_peaks` identifies local maxima in the histogram signal.
    #
    # Parameters:
    #   prominence  – a peak must be at least this tall relative to
    #                  its surrounding baseline to count.
    #   distance    – minimum number of bins between any two peaks
    #                  (enforces the minimum storey height).
    min_prominence = PEAK_PROMINENCE_FRAC * float(np.max(counts))
    min_distance_bins = max(int(MIN_FLOOR_SPACING_M / bin_size_m), 1)

    peak_indices, peak_properties = find_peaks(
        counts.astype(np.float64),
        prominence=min_prominence,
        distance=min_distance_bins,
    )

    # The Z-elevation of each detected slab
    floor_elevations = sorted(bin_centres[peak_indices].tolist())

    log.info("  Detected %d floor slab(s):", len(floor_elevations))
    for fi, fz in enumerate(floor_elevations):
        relative = fz - ground_elevation
        log.info("    Floor %d: Z = %.2f m  (%.1f m above ground)", fi, fz, relative)

    return FloorSlabResult(
        z_min=z_min,
        z_max=z_max,
        ground_elevation=ground_elevation,
        building_height=building_height,
        num_floors_detected=len(floor_elevations),
        floor_elevations=floor_elevations,
        histogram_bins=bin_edges,
        histogram_counts=counts,
    )


# ════════════════════════════════════════════════════════════════════
# §5  ROOF SURFACE EXTRACTION
#     a) RANSAC Plane Fitting  → OGC POLYHEDRALSURFACE Z
#     b) Delaunay TIN Fallback → OGC TIN Z
# ════════════════════════════════════════════════════════════════════

# ────────────────────────────────────────────────────────────────────
# §5a  RANSAC Plane Fitting
# ────────────────────────────────────────────────────────────────────

def _fit_plane_ransac(
    points: np.ndarray,
) -> tuple[tuple[float, float, float, float], float, np.ndarray]:
    """
    Fit a plane Ax + By + Cz + D = 0 to 3-D points using RANSAC.

    Implementation note
    ───────────────────
    Scikit-learn's RANSACRegressor operates on 2-D feature → 1-D
    target, so we treat (X, Y) as features and Z as the target:

        Z ≈ aX + bY + c

    This is equivalent to the plane equation:

        aX + bY − Z + c = 0   →   A=a, B=b, C=−1, D=c

    We then normalise so that (A² + B² + C²) = 1.

    Parameters
    ----------
    points : ndarray, shape (N, 3)
        3-D point coordinates.

    Returns
    -------
    (A, B, C, D) : tuple[float, …]
        Normalised plane equation coefficients.
    mean_residual : float
        Mean absolute distance of *all* points from the fitted plane
        (metres).  A large value indicates the surface is not planar.
    residuals : ndarray, shape (N,)
        Per-point signed distance to the plane.
    """
    X_feat = points[:, :2]   # (N, 2) → [X, Y]
    Z_tgt  = points[:, 2]    # (N,)   → Z

    ransac = RANSACRegressor(
        residual_threshold=RANSAC_RESIDUAL_THRESHOLD_M,
        max_trials=RANSAC_MAX_TRIALS,
        random_state=42,
    )
    ransac.fit(X_feat, Z_tgt)

    # Extract coefficients:  Z = a·X + b·Y + c
    a = float(ransac.estimator_.coef_[0])
    b = float(ransac.estimator_.coef_[1])
    c = float(ransac.estimator_.intercept_)

    # Convert to Ax + By + Cz + D = 0 form
    A_raw, B_raw, C_raw, D_raw = a, b, -1.0, c

    # Normalise the normal vector to unit length
    norm = math.sqrt(A_raw**2 + B_raw**2 + C_raw**2)
    A = A_raw / norm
    B = B_raw / norm
    C = C_raw / norm
    D = D_raw / norm

    # Per-point signed distance from the plane
    residuals = (A * points[:, 0] + B * points[:, 1] + C * points[:, 2] + D)
    mean_residual = float(np.mean(np.abs(residuals)))

    return (A, B, C, D), mean_residual, residuals


def _plane_pitch_angle(A: float, B: float, C: float) -> float:
    """
    Compute the roof pitch angle (degrees) from the plane normal.

    The normal vector of the plane Ax + By + Cz + D = 0 is (A, B, C).
    The pitch is the angle between this normal and the vertical
    (Z-axis unit vector [0, 0, 1]):

        cos(θ) = |C| / √(A² + B² + C²)

    For a perfectly flat roof, θ = 0°.
    For a 45° pitched roof, θ = 45°.
    """
    cos_theta = abs(C) / math.sqrt(A**2 + B**2 + C**2)
    # Clamp to avoid numerical issues with acos
    cos_theta = max(-1.0, min(1.0, cos_theta))
    return math.degrees(math.acos(cos_theta))


# ────────────────────────────────────────────────────────────────────
# §5b  Delaunay TIN Surface (fallback for curved / organic roofs)
# ────────────────────────────────────────────────────────────────────

def _build_tin_surface(
    points: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build a 2.5-D Triangulated Irregular Network (TIN) from 3-D
    points by projecting to the XY plane for Delaunay triangulation,
    then lifting the triangles back to their original Z values.

    Parameters
    ----------
    points : ndarray, shape (N, 3)
        3-D roof point coordinates.

    Returns
    -------
    vertices : ndarray, shape (N, 3)
        Same as input *points* (kept for symmetry).
    simplices : ndarray, shape (T, 3)
        Triangle vertex indices from the Delaunay triangulation.

    Notes
    -----
    2.5-D means every (X, Y) has exactly one Z value (no overhangs).
    This is standard for rooftop surfaces derived from nadir LiDAR.
    """
    # Delaunay triangulation operates on the 2-D (X, Y) projection.
    tri = Delaunay(points[:, :2])

    log.info("    TIN: %d triangles from %d vertices", len(tri.simplices), len(points))

    return points, tri.simplices


# ────────────────────────────────────────────────────────────────────
# §5c  OGC WKT Generation (POLYHEDRALSURFACE Z / TIN Z)
# ────────────────────────────────────────────────────────────────────

def _generate_polyhedralsurface_wkt(
    plane: tuple[float, float, float, float],
    building_xyz: np.ndarray,
    floor_elevations: list[float],
    ground_elevation: float,
) -> str:
    """
    Generate an OGC POLYHEDRALSURFACE Z WKT for a building with a
    planar roof.

    Strategy
    --------
    For a simple extruded building the POLYHEDRALSURFACE is a closed
    shell of planar faces:
      • 1 ground polygon   (at ground_elevation)
      • 1 roof polygon     (at z_max, tilted per the RANSAC plane)
      • 4 wall polygons    (vertical, connecting ground to roof)

    The 2-D footprint is approximated by the convex hull of the
    building points projected to XY.  For production use this would
    be replaced by the simplified polygon from train_2d_yolo.py.

    Parameters
    ----------
    plane : tuple
        (A, B, C, D) normalised plane equation.
    building_xyz : ndarray, shape (N, 3)
        Building-classified points.
    floor_elevations : list[float]
        Detected slab Z values.
    ground_elevation : float
        Ground reference Z.

    Returns
    -------
    str
        OGC WKT string:  ``POLYHEDRALSURFACE Z ((...), ...)``
    """
    from shapely.geometry import MultiPoint

    # ── 2-D convex hull of the building footprint ────────────────
    xy = building_xyz[:, :2]
    hull = MultiPoint(xy).convex_hull

    if hull.geom_type != "Polygon":
        # Degenerate case (all points collinear or coincident)
        return "POLYHEDRALSURFACE Z EMPTY"

    # Exterior ring (already closed by Shapely)
    ring_2d = list(hull.exterior.coords)  # [(x,y), …]

    A, B, C, D = plane
    z_max = float(np.max(building_xyz[:, 2]))

    # ── Helper: Z on the roof plane at a given (x, y) ────────────
    def _roof_z(x: float, y: float) -> float:
        """Evaluate the RANSAC plane: z = -(Ax + By + D) / C."""
        if abs(C) < 1e-12:
            return z_max   # near-vertical plane fallback
        return -(A * x + B * y + D) / C

    # ── Build WKT patches ────────────────────────────────────────
    patches: list[str] = []

    # Ground face (horizontal at ground_elevation)
    ground_coords = ", ".join(
        f"{x:.2f} {y:.2f} {ground_elevation:.2f}" for x, y in ring_2d
    )
    patches.append(f"(({ground_coords}))")

    # Roof face (on the RANSAC plane)
    roof_coords = ", ".join(
        f"{x:.2f} {y:.2f} {_roof_z(x, y):.2f}" for x, y in ring_2d
    )
    patches.append(f"(({roof_coords}))")

    # Wall faces (one per edge of the footprint ring)
    for i in range(len(ring_2d) - 1):
        x1, y1 = ring_2d[i]
        x2, y2 = ring_2d[i + 1]
        gz = ground_elevation
        rz1 = _roof_z(x1, y1)
        rz2 = _roof_z(x2, y2)
        # A wall quad: bottom-left → bottom-right → top-right → top-left → close
        wall = (
            f"(({x1:.2f} {y1:.2f} {gz:.2f}, "
            f"{x2:.2f} {y2:.2f} {gz:.2f}, "
            f"{x2:.2f} {y2:.2f} {rz2:.2f}, "
            f"{x1:.2f} {y1:.2f} {rz1:.2f}, "
            f"{x1:.2f} {y1:.2f} {gz:.2f}))"
        )
        patches.append(wall)

    wkt = "POLYHEDRALSURFACE Z (" + ", ".join(patches) + ")"
    return wkt


def _generate_tin_z_wkt(
    vertices: np.ndarray,
    simplices: np.ndarray,
    max_triangles: int = 500,
) -> str:
    """
    Generate an OGC TIN Z WKT from Delaunay triangulation results.

    Parameters
    ----------
    vertices : ndarray, shape (N, 3)
        3-D vertex coordinates.
    simplices : ndarray, shape (T, 3)
        Triangle vertex indices.
    max_triangles : int
        Cap on the number of triangles in the WKT to keep the string
        manageable.  In production the full TIN would be written to a
        binary format (e.g., CityGML, glTF).

    Returns
    -------
    str
        OGC WKT:  ``TIN Z (((…)), ((…)), …)``
    """
    tri_strings: list[str] = []
    for tri_idx in simplices[:max_triangles]:
        # Each triangle is a closed ring of 4 coordinates (3 + repeat first)
        v0, v1, v2 = vertices[tri_idx]
        coords = (
            f"{v0[0]:.2f} {v0[1]:.2f} {v0[2]:.2f}, "
            f"{v1[0]:.2f} {v1[1]:.2f} {v1[2]:.2f}, "
            f"{v2[0]:.2f} {v2[1]:.2f} {v2[2]:.2f}, "
            f"{v0[0]:.2f} {v0[1]:.2f} {v0[2]:.2f}"
        )
        tri_strings.append(f"(({coords}))")

    wkt = "TIN Z (" + ", ".join(tri_strings) + ")"
    return wkt


# ────────────────────────────────────────────────────────────────────
# §5d  Top-level roof extractor
# ────────────────────────────────────────────────────────────────────

def extract_roof_surface(
    building_xyz: np.ndarray,
    floor_elevations: list[float],
    ground_elevation: float,
    roof_slice_m: float = ROOF_SLICE_THICKNESS_M,
) -> RoofSurfaceResult:
    """
    Extract the roof surface from the building point cloud.

    Workflow
    --------
    1. Slice the topmost ``roof_slice_m`` metres of the building
       point cloud.
    2. Attempt RANSAC plane fitting.
    3. If the mean residual is acceptable → planar roof →
       ``POLYHEDRALSURFACE Z``.
    4. If the residual is too high → curved/organic roof →
       Delaunay TIN → ``TIN Z``.

    Parameters
    ----------
    building_xyz : ndarray, shape (N, 3)
        Building-classified points.
    floor_elevations : list[float]
        Detected slab Z values (for WKT generation).
    ground_elevation : float
        Ground reference Z.
    roof_slice_m : float
        Thickness of the roof slice in metres.

    Returns
    -------
    RoofSurfaceResult
    """
    log.info("═══ Roof Surface Extraction ═══")

    z_max = float(np.max(building_xyz[:, 2]))
    z_cut = z_max - roof_slice_m

    # ── Extract the roof slice ───────────────────────────────────
    roof_mask = building_xyz[:, 2] >= z_cut
    roof_pts  = building_xyz[roof_mask]
    log.info("  Roof slice: Z ≥ %.2f m → %s pts", z_cut, f"{len(roof_pts):,}")

    if len(roof_pts) < 10:
        log.warning("  ⚠️  Too few roof points (%d). Cannot fit surface.", len(roof_pts))
        return RoofSurfaceResult(
            plane_coefficients=None,
            pitch_angle_deg=None,
            is_curved=True,
            ransac_mean_residual=None,
            tin_vertices=None,
            tin_simplices=None,
            ogc_wkt="TIN Z EMPTY",
        )

    # ── RANSAC plane fitting ─────────────────────────────────────
    (A, B, C, D), mean_res, residuals = _fit_plane_ransac(roof_pts)
    pitch = _plane_pitch_angle(A, B, C)

    log.info("  RANSAC plane: %.4f·x + %.4f·y + %.4f·z + %.4f = 0", A, B, C, D)
    log.info("  Pitch angle : %.1f°", pitch)
    log.info("  Mean |residual|: %.3f m  (threshold: %.3f m)",
             mean_res, CURVED_ROOF_RESIDUAL_THRESHOLD_M)

    # ── Decision: planar vs curved ───────────────────────────────
    is_curved = mean_res > CURVED_ROOF_RESIDUAL_THRESHOLD_M

    if is_curved:
        # ── Curved / organic roof → Delaunay TIN ────────────────
        log.info("  → Roof classified as CURVED/ORGANIC. Building TIN surface …")
        tin_verts, tin_simps = _build_tin_surface(roof_pts)

        # Sub-sample if the point cloud is massive (TIN WKT can be huge)
        max_tin_pts = 2000
        if len(roof_pts) > max_tin_pts:
            log.info("    Sub-sampling roof points: %d → %d for TIN",
                     len(roof_pts), max_tin_pts)
            rng = np.random.default_rng(42)
            idx = rng.choice(len(roof_pts), size=max_tin_pts, replace=False)
            tin_verts_sub = roof_pts[idx]
            _, tin_simps_sub = _build_tin_surface(tin_verts_sub)
        else:
            tin_verts_sub = tin_verts
            tin_simps_sub = tin_simps

        wkt = _generate_tin_z_wkt(tin_verts_sub, tin_simps_sub)

        return RoofSurfaceResult(
            plane_coefficients=(A, B, C, D),
            pitch_angle_deg=pitch,
            is_curved=True,
            ransac_mean_residual=mean_res,
            tin_vertices=tin_verts,
            tin_simplices=tin_simps,
            ogc_wkt=wkt,
        )
    else:
        # ── Planar roof → POLYHEDRALSURFACE Z ────────────────────
        log.info("  → Roof classified as PLANAR. Generating POLYHEDRALSURFACE Z …")
        wkt = _generate_polyhedralsurface_wkt(
            (A, B, C, D), building_xyz, floor_elevations, ground_elevation,
        )

        return RoofSurfaceResult(
            plane_coefficients=(A, B, C, D),
            pitch_angle_deg=pitch,
            is_curved=False,
            ransac_mean_residual=mean_res,
            tin_vertices=None,
            tin_simplices=None,
            ogc_wkt=wkt,
        )


# ════════════════════════════════════════════════════════════════════
# §6  TOP-LEVEL PROCESSING FUNCTION
# ════════════════════════════════════════════════════════════════════

def process_las_file(
    las_path: Path,
    bin_size_m: float = DEFAULT_BIN_SIZE_M,
    roof_slice_m: float = ROOF_SLICE_THICKNESS_M,
) -> BuildingExtractionResult:
    """
    Full Z-axis elevation extraction pipeline for a single LAS file.

    Steps
    ─────
    1. Load & filter points (Ground vs Building).
    2. Detect floor slabs via Z-histogram peak detection.
    3. Extract roof surface (RANSAC plane / Delaunay TIN).
    4. Package everything into a ``BuildingExtractionResult``.

    Parameters
    ----------
    las_path : Path
        Input .las / .laz file.
    bin_size_m : float
        Histogram bin width for floor detection (metres).
    roof_slice_m : float
        Thickness of the roof slice (metres).

    Returns
    -------
    BuildingExtractionResult
    """
    # ── §3: Ingest ───────────────────────────────────────────────
    ground_xyz, building_xyz = load_and_filter_las(las_path)

    # ── §4: Floor slabs ──────────────────────────────────────────
    floors = detect_floor_slabs(ground_xyz, building_xyz, bin_size_m=bin_size_m)

    # ── §5: Roof surface ─────────────────────────────────────────
    roof = extract_roof_surface(
        building_xyz,
        floor_elevations=floors.floor_elevations,
        ground_elevation=floors.ground_elevation,
        roof_slice_m=roof_slice_m,
    )

    return BuildingExtractionResult(
        las_path=str(las_path),
        num_ground_points=len(ground_xyz),
        num_building_points=len(building_xyz),
        floors=floors,
        roof=roof,
    )


# ════════════════════════════════════════════════════════════════════
# §7  RESULT DISPLAY & EXPORT
# ════════════════════════════════════════════════════════════════════

def print_results(result: BuildingExtractionResult) -> None:
    """Pretty-print the extraction results to stdout."""
    f = result.floors
    r = result.roof

    print("\n" + "═" * 72)
    print("  3D Cadastre · Z-Axis Elevation Extraction Results")
    print("═" * 72)
    print(f"  LAS file         : {result.las_path}")
    print(f"  Ground points    : {result.num_ground_points:,}")
    print(f"  Building points  : {result.num_building_points:,}")

    print(f"\n── Floor Slabs ────────────────────────────────────")
    print(f"  Ground elevation : {f.ground_elevation:.2f} m")
    print(f"  Z_min (building) : {f.z_min:.2f} m")
    print(f"  Z_max (building) : {f.z_max:.2f} m")
    print(f"  Building height  : {f.building_height:.1f} m")
    print(f"  Floors detected  : {f.num_floors_detected}")
    for i, fz in enumerate(f.floor_elevations):
        agl = fz - f.ground_elevation
        print(f"    Floor {i}: Z = {fz:.2f} m  ({agl:.1f} m AGL)")

    print(f"\n── Roof Surface ───────────────────────────────────")
    if r.plane_coefficients:
        A, B, C, D = r.plane_coefficients
        print(f"  Plane equation   : {A:.4f}x + {B:.4f}y + {C:.4f}z + {D:.4f} = 0")
    if r.pitch_angle_deg is not None:
        print(f"  Pitch angle      : {r.pitch_angle_deg:.1f}°")
    if r.ransac_mean_residual is not None:
        print(f"  RANSAC residual  : {r.ransac_mean_residual:.3f} m")
    print(f"  Curved / organic : {'Yes → TIN Z' if r.is_curved else 'No → POLYHEDRALSURFACE Z'}")

    # Truncate WKT for display
    wkt_display = r.ogc_wkt
    if len(wkt_display) > 200:
        wkt_display = wkt_display[:197] + "…"
    print(f"  OGC WKT          : {wkt_display}")
    print("═" * 72)


def export_results_json(
    result: BuildingExtractionResult,
    output_path: Path,
) -> None:
    """
    Serialise extraction results to a JSON file.

    Parameters
    ----------
    result : BuildingExtractionResult
    output_path : Path
        Destination .json file.
    """
    f = result.floors
    r = result.roof

    data = {
        "project": "3D Cadastre – SIH 26011",
        "las_file": result.las_path,
        "points": {
            "ground": result.num_ground_points,
            "building": result.num_building_points,
        },
        "floors": {
            "ground_elevation_m": f.ground_elevation,
            "z_min_m": f.z_min,
            "z_max_m": f.z_max,
            "building_height_m": f.building_height,
            "num_floors": f.num_floors_detected,
            "slab_elevations_m": f.floor_elevations,
        },
        "roof": {
            "plane_equation": (
                {"A": r.plane_coefficients[0], "B": r.plane_coefficients[1],
                 "C": r.plane_coefficients[2], "D": r.plane_coefficients[3]}
                if r.plane_coefficients else None
            ),
            "pitch_angle_deg": r.pitch_angle_deg,
            "ransac_mean_residual_m": r.ransac_mean_residual,
            "is_curved": r.is_curved,
            "geometry_type": "TIN Z" if r.is_curved else "POLYHEDRALSURFACE Z",
            "ogc_wkt": r.ogc_wkt,
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    log.info("📄  Results exported → %s", output_path)


# ════════════════════════════════════════════════════════════════════
# §8  OPTIONAL MATPLOTLIB VISUALISATION
# ════════════════════════════════════════════════════════════════════

def plot_results(
    result: BuildingExtractionResult,
    save_path: Path | None = None,
) -> None:
    """
    Generate a 2-panel diagnostic plot:
      Left  : Z-axis histogram with detected floor-slab peaks.
      Right : 3-D scatter of the roof slice with RANSAC plane / TIN.

    Parameters
    ----------
    result : BuildingExtractionResult
    save_path : Path | None
        If provided, save the figure as a PNG instead of showing it.
    """
    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    except ImportError:
        log.warning("matplotlib not installed – skipping plot.")
        return

    f = result.floors
    r = result.roof

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(
        f"3D Cadastre · Z-Axis Extraction  –  {Path(result.las_path).name}",
        fontsize=13, fontweight="bold",
    )

    # ── Left panel: Z histogram + floor slab markers ─────────────
    ax1 = axes[0]
    bin_centres = 0.5 * (f.histogram_bins[:-1] + f.histogram_bins[1:])
    ax1.barh(bin_centres, f.histogram_counts, height=np.diff(f.histogram_bins)[0],
             color="#4A90D9", alpha=0.7, edgecolor="white", linewidth=0.3)
    for fz in f.floor_elevations:
        ax1.axhline(fz, color="#E74C3C", linewidth=2, linestyle="--", label=f"Slab @ {fz:.1f} m")
    ax1.axhline(f.ground_elevation, color="#2ECC71", linewidth=2,
                linestyle="-", label=f"Ground @ {f.ground_elevation:.1f} m")
    ax1.set_xlabel("Point count")
    ax1.set_ylabel("Elevation Z (m)")
    ax1.set_title("Vertical Point-Density Histogram")
    # De-duplicate legend entries
    handles, labels = ax1.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax1.legend(by_label.values(), by_label.keys(), fontsize=8)

    # ── Right panel: 3-D roof points ─────────────────────────────
    ax2 = fig.add_subplot(1, 2, 2, projection="3d")

    # Re-load roof points for the scatter
    z_max_val = float(np.max(f.histogram_bins))
    z_cut = z_max_val - ROOF_SLICE_THICKNESS_M

    # We don't have building_xyz stored directly; approximate from
    # the result.  For a real pipeline we'd pass it through.
    # For the plot we just indicate the TIN / plane.
    if r.tin_vertices is not None and r.tin_simplices is not None:
        # Plot TIN surface
        verts = r.tin_vertices
        simps = r.tin_simplices
        ax2.plot_trisurf(
            verts[:, 0], verts[:, 1], verts[:, 2],
            triangles=simps[:200],   # cap for performance
            color="#E67E22", alpha=0.5, edgecolor="gray", linewidth=0.2,
        )
        ax2.set_title("Roof TIN Surface (Delaunay)")
    elif r.plane_coefficients is not None:
        A, B, C, D = r.plane_coefficients
        ax2.set_title(f"RANSAC Roof Plane (pitch {r.pitch_angle_deg:.1f}°)")
        ax2.text2D(0.05, 0.95,
                   f"{A:.3f}x + {B:.3f}y + {C:.3f}z + {D:.3f} = 0",
                   transform=ax2.transAxes, fontsize=8, color="red")

    ax2.set_xlabel("X (m)")
    ax2.set_ylabel("Y (m)")
    ax2.set_zlabel("Z (m)")

    plt.tight_layout()
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(str(save_path), dpi=150, bbox_inches="tight")
        log.info("📊  Plot saved → %s", save_path)
    else:
        plt.show()


# ════════════════════════════════════════════════════════════════════
# §9  CLI ENTRY POINT
# ════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "SIH 26011 – Z-Axis Elevation Extraction from LiDAR .las files. "
            "Detects floor slabs and roof surfaces for OGC 3D geometries."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--las", type=Path,
        help="Path to a single .las / .laz file.",
    )
    input_group.add_argument(
        "--dir", type=Path,
        help="Directory containing .las / .laz files (processes all).",
    )

    parser.add_argument(
        "--bin-size", type=float, default=DEFAULT_BIN_SIZE_M,
        help=f"Z-histogram bin width in metres (default: {DEFAULT_BIN_SIZE_M}).",
    )
    parser.add_argument(
        "--roof-slice", type=float, default=ROOF_SLICE_THICKNESS_M,
        help=f"Roof slice thickness in metres (default: {ROOF_SLICE_THICKNESS_M}).",
    )
    parser.add_argument(
        "--json-out", type=Path, default=None,
        help="Export results to a JSON file.",
    )
    parser.add_argument(
        "--no-plot", action="store_true",
        help="Skip the matplotlib visualisation.",
    )
    parser.add_argument(
        "--plot-save", type=Path, default=None,
        help="Save the plot as a PNG instead of displaying it.",
    )

    args = parser.parse_args()

    # ── Collect input file(s) ────────────────────────────────────
    if args.las:
        las_files = [args.las]
    else:
        las_files = sorted(
            list(args.dir.glob("*.las")) + list(args.dir.glob("*.laz"))
        )
        if not las_files:
            log.error("No .las / .laz files found in %s", args.dir)
            sys.exit(1)
        log.info("Found %d LAS file(s) in %s", len(las_files), args.dir)

    # ── Process each file ────────────────────────────────────────
    for las_path in las_files:
        try:
            result = process_las_file(
                las_path,
                bin_size_m=args.bin_size,
                roof_slice_m=args.roof_slice,
            )
        except (FileNotFoundError, ValueError) as exc:
            log.error("  ❌  %s", exc)
            continue

        # ── Display results ──────────────────────────────────────
        print_results(result)

        # ── Export JSON ──────────────────────────────────────────
        if args.json_out:
            json_path = args.json_out
            # If processing multiple files, add a suffix
            if len(las_files) > 1:
                json_path = args.json_out.with_name(
                    f"{args.json_out.stem}_{las_path.stem}{args.json_out.suffix}"
                )
            export_results_json(result, json_path)

        # ── Plot ─────────────────────────────────────────────────
        if not args.no_plot:
            plot_save = args.plot_save
            if plot_save and len(las_files) > 1:
                plot_save = args.plot_save.with_name(
                    f"{args.plot_save.stem}_{las_path.stem}{args.plot_save.suffix}"
                )
            plot_results(result, save_path=plot_save)


if __name__ == "__main__":
    main()
