#!/usr/bin/env python3
"""
train_3d_lidar.py  –  SIH 26011 · 3D Cadastre Z-Axis Elevation Engine
══════════════════════════════════════════════════════════════════════
Processes .las / .laz LiDAR point clouds to extract building heights,
floor-slab elevations (metadata only), and candidate exposed-surface point
evidence (raw XYZ) for downstream OpenCASCADE surface / B-Rep reconstruction.

Pipeline overview
─────────────────
  1.  INGEST    – Read a .las file via laspy; separate Building vs
                  Ground classified points (ASPRS classes 2 & 6).
  2.  FLOOR     – Build a 1-D vertical (Z-axis) point-density
                  histogram and run peak detection to locate the
                  physical Z-elevation of each concrete floor slab.
  3.  SURFACE   – Select candidate exposed-surface evidence: building
                  points on locally surface-like 3-D neighbourhoods
                  (k-NN PCA in XYZ) with an open view on at least one
                  side. Roofs are NOT distinguished from facades or
                  other exterior surfaces. No Z slice and no XY
                  projection; every measured point is kept in
                  ``building_points_xyz``. A RANSAC plane is computed as a
                  DIAGNOSTIC only.
  4.  OUTPUT    – Return the raw measured XYZ of every point positively
                  identified as building evidence by the LAS
                  classification (``building_points_xyz``; sole
                  authoritative geometry, and COMPLETE only when
                  ``completeness.established`` is True) plus candidate
                  surface-evidence subset (not a roof label) and Z_min /
                  Z_max / floor elevations as metadata. No geometry is generated here.

Requirements
────────────
  pip install laspy[lazrs] numpy scipy

Usage
─────
  # Process a single .las / .laz file:
  python train_3d_lidar.py --las data/raw_lidar/san_francisco_3dep_sample.laz

  # Override histogram resolution (metres):
  python train_3d_lidar.py --las data/raw_lidar/sample.las --bin-size 0.15

  # Process all .las/.laz files in a directory:
  python train_3d_lidar.py --dir data/raw_lidar/

  # Skip the plots and save the identified building XYZ (exact float64 .npy):
  python train_3d_lidar.py --las sample.las --no-plot --points-out out/building.npy
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path

import laspy
import numpy as np
from scipy.signal import find_peaks
from scipy.spatial import cKDTree

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
ASPRS_UNCLASSIFIED: tuple[int, ...] = (0, 1)   # never classified / unclassified

# LAS classes accepted as building evidence for the authoritative
# ``building_points_xyz`` cloud. Default is the standard Building class;
# add any vendor-specific building class codes used by the dataset here.
# Ground (class 2) must never be listed.
BUILDING_EVIDENCE_CLASSES: tuple[int, ...] = (ASPRS_BUILDING,)

# Standard LAS classes that DEFINITIVELY mean "not building" (ground,
# vegetation, noise, water, rail, road, wires/towers, bridge deck). Every
# point must fall in ``building_classes`` or a non-building set for the
# building cloud to be established as complete.
# Deliberately NOT listed (building status is dataset-dependent, so they
# stay UNRESOLVED unless the caller configures them explicitly): class 0/1
# (never classified / unclassified), 8 (Model Key-point), 12 (Overlap
# Points), other reserved codes (19+) and user-defined classes. Model
# Key-point and Overlap Points are standard LAS classes, but their
# building status is dataset-dependent, so they are NOT assumed
# non-building.
KNOWN_NON_BUILDING_CLASSES: tuple[int, ...] = (
    2, 3, 4, 5, 7, 9, 10, 11, 13, 14, 15, 16, 17, 18,
)

# EARLY-INGESTION SAFETY CHECK ONLY: if more than this fraction of the
# file is unclassified (class 0/1), ingestion fails outright because the
# classification clearly cannot separate building from ground. Passing it
# does NOT establish completeness; that is decided solely by the full
# unresolved-class check (``CloudCompleteness``).
MAX_UNCLASSIFIED_FRACTION: float = 0.5

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

# ── Candidate surface evidence ────────────────────────────────────────
# A measured point is candidate exposed-surface evidence (NOT a roof label) when its k-NN neighbourhood
# (true 3-D) is surface-like AND at least one side of that local sheet has
# a wide-open view (see below). No slope / Z-orientation rule is applied,
# so steep, vertical, curved and folded surfaces are all eligible. This
# only SELECTS measured points as evidence: none is moved, added, or
# removed from the authoritative ``building_points_xyz`` cloud.
SURFACE_EVIDENCE_KNN: int                = 12
# 3-D exposure test: a side (+n or -n of the local surface normal) is
# "open" when a thin tube marched along the normal AND along a small cone
# of rays tilted around it meets no measured building point. A surface
# with no open side (covered / enclosed: intermediate slab, balcony,
# overhang, room ceiling) is ambiguous and left unlabelled. Tube radius
# is derived
# from the local point spacing (median k-th neighbour distance), the ray
# runs to the building's 3-D bounding-box diagonal (no Z range used).
SURFACE_TUBE_RADIUS_FACTOR: float = 1.0
SURFACE_CONE_HALF_ANGLE_DEG: float = 45.0   # tilt of the extra cone rays
SURFACE_CONE_RAYS: int = 6                  # rays around the normal per side

# RANSAC parameters (DIAGNOSTIC plane fit only; never used as geometry).
RANSAC_RESIDUAL_THRESHOLD_M: float = 0.15   # inlier distance (metres)
RANSAC_MAX_TRIALS: int             = 1000

# If the mean absolute RANSAC residual exceeds this value (metres),
# the evidence set is flagged non-planar (diagnostic metadata only).
CURVED_SURFACE_RESIDUAL_THRESHOLD_M: float = 0.30


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
class SurfaceEvidenceResult:
    """Candidate exposed-surface evidence (raw XYZ); NOT a roof label."""
    # CANDIDATE evidence: unmodified measured points (a subset of the
    # authoritative ``building_points_xyz``); no rounding, resampling,
    # interpolation, or added points.
    surface_evidence_xyz: np.ndarray   # (M, 3) subset of building_points_xyz
    # DIAGNOSTIC / METADATA ONLY (Ax + By + Cz + D = 0 RANSAC fit); never
    # used to generate geometry.
    plane_coefficients: tuple[float, float, float, float] | None  # (A,B,C,D)
    pitch_angle_deg: float | None      # angle between plane normal & vertical
    is_curved: bool | None             # residual above threshold (diagnostic)
    ransac_mean_residual: float | None # mean |residual| in metres
    # Always False: without an external semantic roof label, exposed
    # facades / exterior surfaces cannot be told from roofs (no gravity or
    # Z reference is assumed). Nothing here is authoritative roof geometry;
    # ``building_points_xyz`` is the sole authoritative measured cloud.
    roof_label_authoritative: bool = False


@dataclass
class CloudCompleteness:
    """Whether the input classification establishes building-cloud completeness."""
    # True only if EVERY point in the file is positively assigned either to
    # a building class or to a known non-building class. Nothing is guessed:
    # unresolved points (class 0/1, reserved, unknown) are never treated as
    # building or as non-building.
    established: bool
    unresolved_points: int
    unresolved_classes: dict[int, int]   # class code -> point count
    reason: str


class IncompleteBuildingCloudError(ValueError):
    """Raised when completeness is not established; downstream stops.

    The measured points are carried unmodified for DIAGNOSTIC inspection
    only; they must never feed authoritative reconstruction.
    """

    def __init__(self, message: str, ground_xyz: np.ndarray,
                 building_points_xyz: np.ndarray,
                 completeness: CloudCompleteness) -> None:
        super().__init__(message)
        self.ground_xyz = ground_xyz
        self.building_points_xyz = building_points_xyz   # partial, non-authoritative
        self.completeness = completeness


@dataclass
class BuildingExtractionResult:
    """Complete output of the Z-axis elevation pipeline."""
    las_path: str
    num_ground_points: int
    num_building_points: int
    # AUTHORITATIVE: exact measured XYZ of every point positively
    # identified as building evidence by the LAS classification (no
    # rounding, resampling, or added points). It is the COMPLETE building
    # cloud ONLY if ``completeness.established``; otherwise it is a
    # NON-AUTHORITATIVE partial cloud (unresolved points were left out,
    # neither guessed as building nor discarded from the file). Surface-
    # evidence / floor results below are subsets / metadata and never
    # replace this cloud.
    building_points_xyz: np.ndarray = field(repr=False)   # (N_b, 3)
    completeness: CloudCompleteness
    floors: FloorSlabResult
    surface_evidence: SurfaceEvidenceResult


# ════════════════════════════════════════════════════════════════════
# §3  LAS FILE INGESTION & POINT CLASSIFICATION FILTER
# ════════════════════════════════════════════════════════════════════

def load_and_filter_las(
    las_path: Path,
    building_classes: tuple[int, ...] = BUILDING_EVIDENCE_CLASSES,
    non_building_classes: tuple[int, ...] = KNOWN_NON_BUILDING_CLASSES,
) -> tuple[np.ndarray, np.ndarray, CloudCompleteness]:
    """
    Read a .las / .laz file and separate Building vs Ground points.

    Parameters
    ----------
    las_path : Path
        Path to the input LAS/LAZ file.
    building_classes : tuple[int, ...]
        LAS classes identified as building evidence (default: the
        standard Building class). Must not include Ground (class 2).
    non_building_classes : tuple[int, ...]
        LAS classes definitively known NOT to be building (default:
        ``KNOWN_NON_BUILDING_CLASSES``). Ambiguous classes (e.g. 8 Model Key-point, 12 Overlap Points)
        may be added here by the caller only if the dataset guarantees
        they contain no building geometry; otherwise they remain
        unresolved. Must not overlap ``building_classes``.

    Returns
    -------
    ground_xyz : ndarray, shape (N_g, 3)
        XYZ coordinates of points classified as Ground (ASPRS 2).
    building_xyz : ndarray, shape (N_b, 3)
        Exact measured XYZ of EVERY point whose LAS class is in
        ``building_classes`` (the authoritative building cloud). Points
        of other classes are not building evidence and are never added.
    completeness : CloudCompleteness
        Whether the classification establishes that this cloud is the
        complete building cloud (see ``KNOWN_NON_BUILDING_CLASSES``).
        Below the failure thresholds the cloud is still returned but is
        marked non-authoritative when completeness is not established.

    Raises
    ------
    FileNotFoundError
        If *las_path* does not exist.
    ValueError
        If the LAS classification cannot reliably separate building from
        ground (no ground points, or too many unclassified points), if
        ``building_classes`` includes Ground, or if the file contains zero
        building points (nothing to process).

    Notes
    -----
    ASPRS classification is embedded in the point record's
    ``classification`` field.  Vendors sometimes use non-standard
    codes: list them in ``building_classes``. Unclassified points
    (class 0/1) cannot be identified as building and are excluded from
    both clouds and leave completeness unestablished; their count is
    logged, and ingestion fails when they exceed
    ``MAX_UNCLASSIFIED_FRACTION`` of the file (early safety check only;
    completeness is decided by the unresolved-class check).
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
    if ASPRS_GROUND in building_classes:
        raise ValueError("building_classes must not include Ground (class 2).")
    if set(building_classes) & set(non_building_classes):
        raise ValueError("building_classes and non_building_classes overlap.")
    ground_mask   = classifications == ASPRS_GROUND
    building_mask = np.isin(classifications, building_classes)

    ground_xyz   = np.column_stack((x[ground_mask],   y[ground_mask],   z[ground_mask]))
    building_xyz = np.column_stack((x[building_mask], y[building_mask], z[building_mask]))

    log.info("  Ground   (class %d): %s pts", ASPRS_GROUND,   f"{len(ground_xyz):,}")
    log.info("  Building (classes %s): %s pts", list(building_classes), f"{len(building_xyz):,}")

    # ── Fail explicitly if building-vs-ground classification is unreliable ──
    unclassified_count = int(np.isin(classifications, ASPRS_UNCLASSIFIED).sum())
    if unclassified_count > MAX_UNCLASSIFIED_FRACTION * total_points:
        raise ValueError(
            f"{100.0 * unclassified_count / total_points:.0f}% of points in "
            f"{las_path.name} are unclassified (class 0/1); the LAS "
            f"classification cannot reliably separate building from ground. "
            f"Classify the file (e.g., PDAL / LAStools) first."
        )
    if len(ground_xyz) == 0:
        raise ValueError(
            f"No ground points (ASPRS class {ASPRS_GROUND}) in {las_path.name}; "
            f"building-vs-ground classification is unavailable, so the "
            f"building cloud cannot be treated as complete."
        )
    # ── Completeness: only established if every point is accounted for ──
    resolved = (building_mask
                | np.isin(classifications, non_building_classes)
                | (classifications == ASPRS_GROUND))
    unresolved_classes = {
        int(c): int(n) for c, n in zip(*np.unique(classifications[~resolved],
                                                  return_counts=True))
    }
    unresolved_points = int(sum(unresolved_classes.values()))
    if unresolved_points == 0:
        completeness = CloudCompleteness(
            True, 0, {},
            "every point is assigned to a building or known non-building class",
        )
    else:
        completeness = CloudCompleteness(
            False, unresolved_points, unresolved_classes,
            f"{unresolved_points:,} points have unresolved classes "
            f"{unresolved_classes}; building geometry may exist outside "
            f"the identified building classes",
        )
        log.warning(
            "  ⚠️  Building-cloud completeness NOT established: %s. Points "
            "are neither treated as building nor discarded from the file; "
            "the building cloud is NON-AUTHORITATIVE (partial).",
            completeness.reason,
        )

    if len(building_xyz) == 0:
        raise ValueError(
            f"No building points (LAS classes {list(building_classes)}) found in "
            f"{las_path.name}.  Run a classification tool (e.g., PDAL "
            f"'filters.smrf' + 'filters.hag') first, or verify the file."
        )

    return ground_xyz, building_xyz, completeness


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

    METADATA / PROVENANCE ONLY: this is a 1-D Z histogram. Its outputs
    (z_min, z_max, ground elevation, floor elevations) never generate or
    position geometry.

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
    surfaces (ground, concrete slabs, upper decks).  These show up as
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
# §5  CANDIDATE SURFACE EVIDENCE (measured XYZ; no surface is generated)
# ════════════════════════════════════════════════════════════════════

# ────────────────────────────────────────────────────────────────────
# §5a  RANSAC plane (DIAGNOSTIC / METADATA ONLY)
# ────────────────────────────────────────────────────────────────────

def _fit_plane_ransac(
    points: np.ndarray,
) -> tuple[tuple[float, float, float, float], float, np.ndarray]:
    """
    Fit a plane Ax + By + Cz + D = 0 to 3-D points using RANSAC.

    DIAGNOSTIC ONLY: the plane describes the candidate surface evidence
    for metadata; it is never used to generate or replace geometry, and
    no point is filtered, moved, or deleted (``points`` is read-only).

    True 3-D fit
    ────────────
    The plane is fitted directly in (X, Y, Z) with an orientation-free
    point-to-plane distance, so vertical, slanted, curved and arbitrary-
    orientation surfaces are handled (no Z = f(X, Y) regression, no XY
    projection). Each trial builds a plane from 3 random measured points;
    the plane with the most inliers (distance <=
    ``RANSAC_RESIDUAL_THRESHOLD_M``) is refined by a total-least-squares
    (SVD) fit on those inliers, then normalised so (A² + B² + C²) = 1.

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

    Raises
    ------
    ValueError
        If no non-degenerate plane can be formed (diagnostic unavailable).
    """
    n = len(points)
    if n < 3:
        raise ValueError("Need at least 3 points for a plane fit.")

    rng = np.random.default_rng(42)
    best_inliers = None
    best_count = 0
    for _ in range(RANSAC_MAX_TRIALS):
        p0, p1, p2 = points[rng.choice(n, size=3, replace=False)]
        normal = np.cross(p1 - p0, p2 - p0)
        length = np.linalg.norm(normal)
        if length < 1e-12:                     # collinear sample
            continue
        normal /= length
        inliers = np.abs((points - p0) @ normal) <= RANSAC_RESIDUAL_THRESHOLD_M
        count = int(inliers.sum())
        if count > best_count:
            best_count, best_inliers = count, inliers

    if best_inliers is None or best_count < 3:
        raise ValueError("RANSAC could not find a valid plane.")

    # Total-least-squares refinement on the consensus set (3-D SVD).
    inl = points[best_inliers]
    centroid = inl.mean(axis=0)
    normal = np.linalg.svd(inl - centroid, full_matrices=False)[2][-1]
    A, B, C = (float(v) for v in normal)       # already unit length
    D = float(-normal @ centroid)

    # Per-point signed distance from the plane (all points, read-only)
    residuals = points @ normal + D
    mean_residual = float(np.mean(np.abs(residuals)))

    return (A, B, C, D), mean_residual, residuals


def _plane_pitch_angle(A: float, B: float, C: float) -> float:
    """
    Compute the fitted plane's pitch angle (degrees) from its normal (diagnostic).

    The normal vector of the plane Ax + By + Cz + D = 0 is (A, B, C).
    The pitch is the angle between this normal and the vertical
    (Z-axis unit vector [0, 0, 1]):

        cos(θ) = |C| / √(A² + B² + C²)

    For a horizontal plane, θ = 0°.
    For a 45° inclined plane, θ = 45°.
    """
    cos_theta = abs(C) / math.sqrt(A**2 + B**2 + C**2)
    # Clamp to avoid numerical issues with acos
    cos_theta = max(-1.0, min(1.0, cos_theta))
    return math.degrees(math.acos(cos_theta))


# ────────────────────────────────────────────────────────────────────
# §5b  Candidate exposed-surface evidence selection
# ────────────────────────────────────────────────────────────────────

def _select_surface_evidence_points(
    building_xyz: np.ndarray,
    k: int = SURFACE_EVIDENCE_KNN,
) -> np.ndarray:
    """
    Select measured points on exposed, locally surface-like 3-D sheets
    (candidate evidence only; roofs are not distinguished from facades).

    Each point's own k nearest neighbours (in true XYZ) give a local
    covariance; a point is kept when

      * its neighbourhood is SURFACE-like (2-D dominant, not a line or
        volumetric scatter), and
      * at least one side of that sheet has a wide-open 3-D view (see
        exposure test below).

    No wall-vs-roof classification is made and there is no slope /
    Z-orientation rule: steep, vertical, curved and folded surfaces are
    eligible. Without a gravity or semantic reference an exposed facade
    cannot be told from a roof, so facades may be included; this is
    candidate evidence only, not a roof label.

    Nothing is sliced by elevation and nothing is projected to XY, so
    multiple surface sheets, overhangs, curved / slanted surfaces, and
    several Z values at the same XY all keep their measured points.
    Nothing is connected, triangulated, or fitted: the returned array is
    a subset of ``building_xyz`` with unmodified coordinates, ready for
    downstream OpenCASCADE surface / B-Rep reconstruction.
    """
    pts = building_xyz
    n = len(pts)
    if n < k + 1:
        return pts[:0]

    tree = cKDTree(pts)
    nn_dist, nn_idx = tree.query(pts, k=k + 1)     # column 0 is the point
    keep = np.zeros(n, dtype=bool)
    normals = np.zeros((n, 3))   # unit local normals, sign arbitrary

    chunk = 200_000                                 # bounds memory only
    for lo in range(0, n, chunk):
        hi = min(lo + chunk, n)
        nb = pts[nn_idx[lo:hi]]                     # (m, k+1, 3)
        centred = nb - nb.mean(axis=1, keepdims=True)
        cov = np.einsum("nki,nkj->nij", centred, centred) / k
        evals, evecs = np.linalg.eigh(cov)          # ascending
        sv = np.sqrt(np.clip(evals, 0.0, None))
        s2 = np.where(sv[:, 2] > 1e-12, sv[:, 2], 1e-12)
        a1 = (sv[:, 2] - sv[:, 1]) / s2             # line-like
        a2 = (sv[:, 1] - sv[:, 0]) / s2             # surface-like
        a3 = sv[:, 0] / s2                          # volumetric scatter
        surface_like = (sv[:, 2] > 1e-12) & (a2 >= a1) & (a2 >= a3)
        keep[lo:hi] = surface_like                   # no wall / roof / slope classification
        normals[lo:hi] = evecs[:, :, 0]              # smallest-variance axis

    # ── 3-D exposure / occlusion test (topological context) ──────────
    # For each side of a candidate's own normal, march thin tubes along the
    # normal and along a cone of tilted rays through the FULL measured
    # cloud. A side is open only if every ray is unobstructed; a candidate
    # with no open side is ambiguous and stays unlabelled. The normal's
    # sign is never oriented (no Z or other global direction is assumed).
    # Selection heuristic only: rejected points remain measured points in
    # the ``building_points_xyz`` cloud, and every point
    # (candidate or not) still acts as an occluder. Coordinates untouched.
    rt = SURFACE_TUBE_RADIUS_FACTOR * float(np.median(nn_dist[:, k]))
    if not np.isfinite(rt) or rt <= 0.0:
        return pts[:0]        # no evidence; measured cloud is unaffected
    reach = float(np.linalg.norm(pts.max(axis=0) - pts.min(axis=0)))

    # tangent basis (t1, t2) perpendicular to each normal
    helper = np.where(np.abs(normals[:, :1]) < 0.9,
                      np.array([[1.0, 0.0, 0.0]]), np.array([[0.0, 1.0, 0.0]]))
    t1 = np.cross(normals, helper)
    t1 /= np.linalg.norm(t1, axis=1, keepdims=True).clip(1e-12)
    t2 = np.cross(normals, t1)
    tilt = math.radians(SURFACE_CONE_HALF_ANGLE_DEG)
    phis = np.linspace(0.0, 2.0 * math.pi, SURFACE_CONE_RAYS, endpoint=False)

    def _side_open(cand: np.ndarray, sign: float) -> np.ndarray:
        """Boolean mask over ``cand``: side ``sign`` is fully unobstructed."""
        is_open = np.ones(len(cand), dtype=bool)
        dirs = [sign * normals[cand]]
        for phi in phis:
            dirs.append(sign * (math.cos(tilt) * normals[cand]
                                + math.sin(tilt) * (math.cos(phi) * t1[cand]
                                                    + math.sin(phi) * t2[cand])))
        for d in dirs:
            live = np.flatnonzero(is_open)
            t = 2.0 * rt      # start clear of the candidate's own surface
            while len(live) and t <= reach:
                dist, _ = tree.query(pts[cand[live]] + d[live] * t, k=1,
                                     distance_upper_bound=rt, workers=-1)
                hit = np.isfinite(dist)
                is_open[live[hit]] = False
                live = live[~hit]
                t += rt
        return is_open

    cand = np.flatnonzero(keep)
    open_pos = _side_open(cand, 1.0)
    rest = ~open_pos
    open_neg = np.zeros(len(cand), dtype=bool)
    open_neg[rest] = _side_open(cand[rest], -1.0)
    keep[:] = False
    keep[cand[open_pos | open_neg]] = True

    return pts[keep]

# ────────────────────────────────────────────────────────────────────
# §5c  Top-level surface-evidence extractor
# ────────────────────────────────────────────────────────────────────

def extract_surface_evidence(building_xyz: np.ndarray) -> SurfaceEvidenceResult:
    """
    Extract candidate exposed-surface evidence from the building point cloud.

    ``surface_evidence_xyz`` is a raw measured-XYZ SUBSET (candidate evidence, not a roof label; ``roof_label_authoritative`` is False),
    unmodified. The RANSAC plane, pitch, and residual are diagnostics
    only. No triangulation, hull, extrusion, or WKT is produced here.

    Parameters
    ----------
    building_xyz : ndarray, shape (N, 3)
        Building-classified points.

    Returns
    -------
    SurfaceEvidenceResult
    """
    log.info("═══ Candidate Surface Evidence ═══")

    evidence_pts = _select_surface_evidence_points(building_xyz)
    log.info("  Candidate exposed-surface points: %s of %s building pts",
             f"{len(evidence_pts):,}", f"{len(building_xyz):,}")

    if len(evidence_pts) < 10:
        log.warning("  ⚠️  Too few surface-evidence points (%d). No diagnostics computed.",
                    len(evidence_pts))
        return SurfaceEvidenceResult(
            surface_evidence_xyz=evidence_pts,
            plane_coefficients=None,
            pitch_angle_deg=None,
            is_curved=None,
            ransac_mean_residual=None,
        )

    # ── RANSAC plane: diagnostic only, must never block the evidence ──
    try:
        (A, B, C, D), mean_res, _ = _fit_plane_ransac(evidence_pts)
    except ValueError as exc:
        log.warning("  RANSAC diagnostic unavailable: %s", exc)
        return SurfaceEvidenceResult(
            surface_evidence_xyz=evidence_pts,
            plane_coefficients=None,
            pitch_angle_deg=None,
            is_curved=None,
            ransac_mean_residual=None,
        )
    pitch = _plane_pitch_angle(A, B, C)

    log.info("  [diagnostic] RANSAC plane: %.4f·x + %.4f·y + %.4f·z + %.4f = 0", A, B, C, D)
    log.info("  [diagnostic] Pitch angle : %.1f°", pitch)
    log.info("  [diagnostic] Mean |residual|: %.3f m  (threshold: %.3f m)",
             mean_res, CURVED_SURFACE_RESIDUAL_THRESHOLD_M)

    return SurfaceEvidenceResult(
        surface_evidence_xyz=evidence_pts,
        plane_coefficients=(A, B, C, D),
        pitch_angle_deg=pitch,
        is_curved=mean_res > CURVED_SURFACE_RESIDUAL_THRESHOLD_M,
        ransac_mean_residual=mean_res,
    )


# ════════════════════════════════════════════════════════════════════
# §6  TOP-LEVEL PROCESSING FUNCTION
# ════════════════════════════════════════════════════════════════════

def process_las_file(
    las_path: Path,
    bin_size_m: float = DEFAULT_BIN_SIZE_M,
) -> BuildingExtractionResult:
    """
    Full Z-axis elevation extraction pipeline for a single LAS file.

    Raises ``IncompleteBuildingCloudError`` (a ``ValueError``) before any
    floor / surface extraction if completeness is not established.

    Steps
    ─────
    1. Load & filter points (Ground vs Building).
    2. Detect floor slabs via Z-histogram peak detection.
    3. Select candidate exposed-surface evidence (raw XYZ; RANSAC = diagnostic).
    4. Package everything into a ``BuildingExtractionResult``.

    Parameters
    ----------
    las_path : Path
        Input .las / .laz file.
    bin_size_m : float
        Histogram bin width for floor detection (metres).

    Returns
    -------
    BuildingExtractionResult
    """
    # ── §3: Ingest ───────────────────────────────────────────────
    ground_xyz, building_xyz, completeness = load_and_filter_las(las_path)

    # ── Stop before ANY downstream extraction on a partial cloud ─────
    if not completeness.established:
        raise IncompleteBuildingCloudError(
            f"Building-cloud completeness not established for "
            f"{las_path.name}: {completeness.reason}. Refusing to run "
            f"floor / surface extraction on a partial, non-authoritative "
            f"cloud. Configure building_classes / non_building_classes or "
            f"classify the file.",
            ground_xyz, building_xyz, completeness,
        )

    # ── §4: Floor slabs ──────────────────────────────────────────
    floors = detect_floor_slabs(ground_xyz, building_xyz, bin_size_m=bin_size_m)

    # ── §5: Candidate surface evidence ─────────────────────────────────────────
    surface_evidence = extract_surface_evidence(building_xyz)

    return BuildingExtractionResult(
        las_path=str(las_path),
        num_ground_points=len(ground_xyz),
        num_building_points=len(building_xyz),
        building_points_xyz=building_xyz,
        completeness=completeness,
        floors=floors,
        surface_evidence=surface_evidence,
    )


# ════════════════════════════════════════════════════════════════════
# §7  RESULT DISPLAY & EXPORT
# ════════════════════════════════════════════════════════════════════

def print_results(result: BuildingExtractionResult) -> None:
    """Pretty-print the extraction results to stdout."""
    f = result.floors
    r = result.surface_evidence

    print("\n" + "═" * 72)
    print("  3D Cadastre · Z-Axis Elevation Extraction Results")
    print("═" * 72)
    print(f"  LAS file         : {result.las_path}")
    print(f"  Ground points    : {result.num_ground_points:,}")
    print(f"  Building points  : {result.num_building_points:,} (identified as building)")
    c = result.completeness
    print(f"  Cloud complete   : {'YES (by classification)' if c.established else 'NOT ESTABLISHED – non-authoritative'}")

    print(f"\n── Floor Slabs ────────────────────────────────────")
    print(f"  Ground elevation : {f.ground_elevation:.2f} m")
    print(f"  Z_min (building) : {f.z_min:.2f} m")
    print(f"  Z_max (building) : {f.z_max:.2f} m")
    print(f"  Building height  : {f.building_height:.1f} m")
    print(f"  Floors detected  : {f.num_floors_detected}")
    for i, fz in enumerate(f.floor_elevations):
        agl = fz - f.ground_elevation
        print(f"    Floor {i}: Z = {fz:.2f} m  ({agl:.1f} m AGL)")

    print(f"\n── Candidate Surface Evidence (measured XYZ) ───────────")
    print(f"  Surface evidence : {len(r.surface_evidence_xyz):,} (candidate, not a roof label)")
    if r.plane_coefficients:
        A, B, C, D = r.plane_coefficients
        print(f"  [diagnostic] plane : {A:.4f}x + {B:.4f}y + {C:.4f}z + {D:.4f} = 0")
    if r.pitch_angle_deg is not None:
        print(f"  [diagnostic] pitch : {r.pitch_angle_deg:.1f}°")
    if r.ransac_mean_residual is not None:
        print(f"  [diagnostic] RANSAC residual : {r.ransac_mean_residual:.3f} m")
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
    r = result.surface_evidence

    data = {
        "project": "3D Cadastre – SIH 26011",
        "las_file": result.las_path,
        "points": {
            "ground": result.num_ground_points,
            "building": result.num_building_points,
        },
        "building_cloud": {
            "authoritative_complete": result.completeness.established,
            "unresolved_points": result.completeness.unresolved_points,
            "unresolved_classes": {str(k): v for k, v in
                                   result.completeness.unresolved_classes.items()},
            "reason": result.completeness.reason,
        },
        "floors": {
            "ground_elevation_m": f.ground_elevation,
            "z_min_m": f.z_min,
            "z_max_m": f.z_max,
            "building_height_m": f.building_height,
            "num_floors": f.num_floors_detected,
            "slab_elevations_m": f.floor_elevations,
        },
        "surface_evidence": {
            "roof_label_authoritative": False,   # no external roof semantics
            "plane_equation": (
                {"A": r.plane_coefficients[0], "B": r.plane_coefficients[1],
                 "C": r.plane_coefficients[2], "D": r.plane_coefficients[3]}
                if r.plane_coefficients else None
            ),
            "pitch_angle_deg": r.pitch_angle_deg,
            "ransac_mean_residual_m": r.ransac_mean_residual,
            "is_curved": r.is_curved,
            "evidence_points": int(len(r.surface_evidence_xyz)),
            "diagnostic_only": True,   # plane / pitch / residual: not geometry
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
      Right : 3-D scatter of the candidate surface evidence (display only).

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
    r = result.surface_evidence

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

    # ── Right panel: candidate surface-evidence points (display only) ─
    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    pts = r.surface_evidence_xyz
    if len(pts):
        shown = pts[:: max(1, len(pts) // 20000)]   # plot-only thinning
        ax2.scatter(shown[:, 0], shown[:, 1], shown[:, 2], s=1, color="#E67E22")
    ax2.set_title("Candidate surface evidence (raw XYZ subset)")

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
            "Detects floor slabs (metadata) and candidate exposed-surface points."
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
        "--points-out", type=Path, default=None,
        help="Save the identified building XYZ cloud (exact float64) as .npy; complete only if completeness is established.",
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

        # ── Save identified building XYZ (authoritative only if completeness established) ────────
        if args.points_out:
            points_path = args.points_out
            if len(las_files) > 1:
                points_path = args.points_out.with_name(
                    f"{args.points_out.stem}_{las_path.stem}{args.points_out.suffix}"
                )
            points_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(points_path, result.building_points_xyz)
            log.info("📐  Identified building XYZ saved (%s) → %s",
                     "complete" if result.completeness.established
                     else "NON-AUTHORITATIVE, completeness not established", points_path)

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