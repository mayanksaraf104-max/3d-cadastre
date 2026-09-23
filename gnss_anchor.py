"""
gnss_anchor.py -- strictly 2D GNSS/CORS georeferencing.

CONTRACT
--------
* Maps image pixel coordinates (x, y) to real-world planar coordinates (X, Y)
  (e.g. UTM meters) using a 2D affine transform fitted to Ground Control Points.
* This module is 2D ONLY. It never generates, infers, or carries Z, height,
  floor elevation, z_min/z_max, extrusion, or any OpenCASCADE / 3D geometry.
  Inputs with a Z dimension are rejected.
* Output of `anchor_polygon()` is REFERENCE / SEMANTIC / VALIDATION / INDEXING
  geometry only (e.g. spatial indexing, cross-checks, semantic association).
  It is NEVER authoritative 3D geometry and must not be used as the source
  for building solids, extrusions, or the final cadastre model.
"""
import numpy as np
from shapely.geometry import Polygon

# Minimum ratio of the 2nd to 1st singular value of the centered GCP set.
# Below this the GCPs are (near-)collinear or coincident and the affine fit
# is not uniquely determined.
_MIN_GCP_SPREAD_RATIO = 1e-6


class GNSSCoordinateAnchor:
    def __init__(self, gcp_pixels, gcp_real_world):
        """
        Initializes the GNSS Anchor by calculating the 2D Affine Transformation Matrix.
        Requires at least 3 non-degenerate (non-collinear) Ground Control Points (GCPs)
        to calculate Scaling, Rotation, and Translation.

        2D only: pixel (x, y) -> real-world (X, Y). No Z/height/elevation is produced.

        Raises:
            ValueError: if the GCPs have mismatched/invalid shapes, contain
                non-finite values, number fewer than 3, or are degenerate.
        """
        pixels, real_world = self._validate_gcps(gcp_pixels, gcp_real_world)
        self.matrix = self._calculate_affine_matrix(pixels, real_world)

    @staticmethod
    def _validate_gcps(gcp_pixels, gcp_real_world):
        """Validate GCP inputs and return them as float (N, 2) arrays."""
        try:
            pixels = np.asarray(gcp_pixels, dtype=float)
            real_world = np.asarray(gcp_real_world, dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"GCPs must be numeric arrays: {exc}") from exc

        for name, arr in (("gcp_pixels", pixels), ("gcp_real_world", real_world)):
            if arr.ndim != 2 or arr.shape[1] != 2:
                raise ValueError(
                    f"{name} must have shape (N, 2) (2D only); got {arr.shape}"
                )
        if pixels.shape != real_world.shape:
            raise ValueError(
                f"GCP shape mismatch: pixels {pixels.shape} vs real-world {real_world.shape}"
            )
        if pixels.shape[0] < 3:
            raise ValueError(
                f"At least 3 GCPs are required; got {pixels.shape[0]}"
            )
        for name, arr in (("gcp_pixels", pixels), ("gcp_real_world", real_world)):
            if not np.all(np.isfinite(arr)):
                raise ValueError(f"{name} contains NaN or infinite values")

        # Non-degeneracy: the points must span 2D (not coincident, not collinear).
        # Checked on both sides so the fitted transform cannot collapse.
        for name, arr in (("gcp_pixels", pixels), ("gcp_real_world", real_world)):
            s = np.linalg.svd(arr - arr.mean(axis=0), compute_uv=False)
            if s[0] <= 0.0 or (s[1] / s[0]) < _MIN_GCP_SPREAD_RATIO:
                raise ValueError(
                    f"{name} are degenerate (coincident or collinear); "
                    "need at least 3 non-collinear GCPs"
                )
        return pixels, real_world

    def _calculate_affine_matrix(self, pixels, real_world):
        """
        Uses Ordinary Least Squares (OLS) to solve the 2D Affine Transformation:
        X_real = a*x_pixel + b*y_pixel + c
        Y_real = d*x_pixel + e*y_pixel + f
        """
        print("🛰️ Calculating GNSS/CORS Affine Transformation Matrix...")
        
        # Build the A matrix from pixel coordinates [x, y, 1]
        A = np.c_[pixels, np.ones(pixels.shape[0])]
        
        # Real-world X and Y vectors
        X_real = real_world[:, 0]
        Y_real = real_world[:, 1]
        
        # Solve for coefficients [a, b, c] and [d, e, f]
        coef_X, _, _, _ = np.linalg.lstsq(A, X_real, rcond=None)
        coef_Y, _, _, _ = np.linalg.lstsq(A, Y_real, rcond=None)
        
        return {
            'a': coef_X[0], 'b': coef_X[1], 'c': coef_X[2],
            'd': coef_Y[0], 'e': coef_Y[1], 'f': coef_Y[2]
        }

    def anchor_polygon(self, pixel_polygon):
        """
        Takes a 2D Shapely polygon in pixel space and mathematically warps it 
        to real-world planar coordinates (e.g. UTM meters).

        CONTRACT: the result is 2D REFERENCE / SEMANTIC / VALIDATION / INDEXING
        geometry only. It is never authoritative 3D geometry, carries no Z,
        height, elevation, or extrusion data, and must not be used to build 3D
        solids. Only the exterior ring is transformed.

        Raises:
            ValueError: if the input polygon has a Z dimension.
        """
        if pixel_polygon.has_z:
            raise ValueError(
                "anchor_polygon() is strictly 2D; input polygon must not have Z coordinates"
            )
        real_world_coords = []
        for x, y in pixel_polygon.exterior.coords:
            # Apply the transformation matrix to every single vertex
            X = self.matrix['a'] * x + self.matrix['b'] * y + self.matrix['c']
            Y = self.matrix['d'] * x + self.matrix['e'] * y + self.matrix['f']
            real_world_coords.append((X, Y))
            
        return Polygon(real_world_coords)

# --- Hackathon Proof Test ---
if __name__ == "__main__":
    print("\n--- GNSS / CORS Anchoring Test ---")
    
    # 1. Simulated Pixel Coordinates from your Blueprint (e.g., corners of the image)
    pixels = np.array([
        [0, 0],         # Top Left
        [1000, 0],      # Top Right
        [0, 1000]       # Bottom Left
    ])
    
    # 2. Simulated Real-World CORS GPS Data (UTM Zone 10N - San Francisco)
    # Planar meters (2D only; no elevation is represented here).
    utm_coords = np.array([
        [552000.0, 4182000.0],  # Real-world Top Left
        [552500.0, 4182000.0],  # Real-world Top Right (500m wide)
        [552000.0, 4181500.0]   # Real-world Bottom Left (500m tall)
    ])
    
    # Initialize the anchor
    gnss_rover = GNSSCoordinateAnchor(pixels, utm_coords)
    
    # Test it on an AI-extracted room
    dummy_ai_room = Polygon([(100, 100), (300, 100), (300, 300), (100, 300), (100, 100)])
    print(f"\n📏 Local Pixel Room Coordinates:\n   {list(dummy_ai_room.exterior.coords)[:2]}...")
    
    anchored_room = gnss_rover.anchor_polygon(dummy_ai_room)
    print(f"\n🌍 Anchored Global UTM Coordinates (2D reference only):\n   {list(anchored_room.exterior.coords)[:2]}...")