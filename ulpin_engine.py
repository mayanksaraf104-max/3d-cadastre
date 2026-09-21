import hashlib
import math


class ULPINEngine:
    """
    Custom TRUE-3D property identifier generator for this 3D cadastre.

    NOT THE OFFICIAL ULPIN/PNIU: the official Unique Land Parcel Identification
    Number (ULPIN, also called PNIU) issued by the Government of India is a
    separate 2D cadastral identifier based on the georeferenced parcel
    vertices. Nothing here implements or approximates that algorithm; the
    identifier below is an internal 3D property ID only.

    TRUE 3D PHILOSOPHY
    ------------------
    A ULPIN encodes the MEASURED 3D position of a property. In a true 3D
    cadastre, Z is treated identically to X and Y -- it is a coordinate,
    not an attribute. The spatial hash is computed from the OGC 3D centroid
    of the solid geometry (computed by PostGIS from the actual polyhedral
    surface), not from user-supplied floor numbers or bounding-box extents.

    Why centroid, not z_min/z_max:
      A dome, slanted roof, or irregular shell has a z_max at its peak and
      a z_min at its base, but the actual volume occupies a shape between
      them that z_min/z_max cannot describe. Two different solids can share
      the same (z_min, z_max) range while being completely disjoint, or
      differ in (z_min, z_max) while substantially overlapping. Treating
      z_min/z_max as spatial proxies is bounding-box reasoning -- 2.5D.

      The 3D centroid is a single (x, y, z) point derived from the full
      geometry by OGC-standard functions (ST_Centroid for X/Y, volumetric
      midpoint for Z). It represents the geometric center of mass, and two
      geometries with different centroids are genuinely at different spatial
      positions in 3D space.

    Overlap detection uses the full solid geometry (ST_3DIntersection +
    ST_Volume in db_engine.py), never the centroid or this ID. The ID is
    an identifier, not a spatial query tool.
    """

    @staticmethod
    def _derive_tier(centroid_z, floor_level=0):
        """
        Derives the tier type from the OGC 3D centroid's Z coordinate.

        The centroid Z is the volumetric center of mass of the solid, computed
        by PostGIS from the actual geometry. A solid whose center of mass is
        below ground is primarily subsurface (even if a ventilation shaft pokes
        above). A solid whose center of mass is above ground is surface or
        air_rights.

        This replaces the old approach of comparing z_min/z_max ranges against
        fixed thresholds, which failed for irregular geometries: a building
        with a deep foundation and a tall tower would span both ranges, and
        the bounding box gave no clear classification.

        Rules:
          - centroid_z < 0: SUBSURFACE (center of mass is below ground)
          - centroid_z >= 0 AND floor_level > 0: AIR_RIGHTS
          - Everything else: SURFACE

        A measured centroid_z is REQUIRED (ValueError otherwise). There is no
        floor_level-only fallback: floor_level never stands in for missing Z.
        """
        if centroid_z is None or not math.isfinite(centroid_z):
            raise ValueError("A measured, finite centroid_z is required to derive the tier; "
                             "floor_level cannot substitute for missing Z.")
        if centroid_z < 0:
            return "SUBSURFACE"
        if centroid_z > 0 and floor_level > 0:
            return "AIR_RIGHTS"
        return "SURFACE"

    @staticmethod
    def generate_3d_property_id(centroid_x=0.0, centroid_y=0.0,
                                tier_type="SURFACE", floor_level=0, nonce=0,
                                centroid_z=None):
        """
        Generates this cadastre's custom 5-digit 3D property identifier,
        derived SOLELY from the measured 3D centroid (centroid_x, centroid_y,
        centroid_z) plus the collision-retry `nonce`. It carries no
        state/district code, tier or floor.

        This is NOT the official ULPIN/PNIU (a separate 2D cadastral
        identifier based on georeferenced parcel vertices); that algorithm is
        not implemented or approximated here.

        TRUE 3D
        -------
        The spatial hash is computed from the OGC 3D centroid (x, y, z),
        where Z is treated identically to X and Y -- all three are coordinates
        of the same OGC-derived point, hashed together into the spatial block.

        This ensures two units at the same (X, Y) but different Z (stacked
        flats, basement vs. ground floor) get naturally distinct ULPINs
        without any bounding-box or z_min/z_max reasoning.

        Parameters
        ----------
        centroid_x, centroid_y : float
            OGC centroid X and Y, computed by PostGIS ST_Centroid() from
            the actual solid geometry. These are real spatial coordinates
            in CADASTRE_SRID, not hash projections.
        centroid_z : float (REQUIRED)
            OGC centroid Z, computed as the volumetric midpoint of the solid.
            Z is included in the spatial hash identically to X and Y. Passing
            None (or a non-finite value) raises ValueError: no identifier is
            generated from X/Y plus floor_level or tier_type when measured Z
            is unavailable. The
            parameter keeps its default of None only so a missing Z fails
            with a clear error instead of a positional-argument mismatch.
        tier_type : str
            Ignored. Accepted only so existing callers do not break; it has
            no influence on the identifier.
        floor_level : int
            Ignored. Accepted only so existing callers do not break; it has
            no influence on the identifier and never substitutes for Z.
        nonce : int
            Retry counter for collision resolution.

        Returns
        -------
        str : fixed-width 5-digit spatial hash of (x, y, z, nonce).

        Raises
        ------
        ValueError : if centroid_z is None or not finite.
        """
        if centroid_z is None or not math.isfinite(centroid_z):
            raise ValueError("centroid_z (the measured OGC 3D centroid Z) is required; a 3D ULPIN "
                             "is never generated from X/Y plus floor_level or tier_type.")

        # 1. Quantize the OGC 3D centroid into a fixed 5-digit spatial hash.
        #
        # TRUE 3D: X, Y, and Z are hashed together as equal coordinates.
        # Two units at the same (X, Y) but different Z produce different
        # hashes naturally, without any bounding-box reasoning.
        coord_str = f"{centroid_x:.4f},{centroid_y:.4f},{centroid_z:.4f},{nonce}"

        spatial_hash = int(hashlib.md5(coord_str.encode()).hexdigest(), 16) % 100000
        spatial_part = f"{spatial_hash:05d}"

        # The identifier is the spatial hash alone: derived solely from the
        # measured 3D centroid (+ nonce). tier_type / floor_level play no part.
        return spatial_part