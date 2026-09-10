import hashlib


class ULPINEngine:
    @staticmethod
    def generate_3d_ulpin(state_code="10", district_code="05", centroid_x=0.0, centroid_y=0.0,
                           tier_type="SURFACE", floor_level=0, nonce=0):
        """
        Generates a standardized 14-digit 3D ULPIN incorporating geographic hierarchy,
        spatial quantization, vertical tier categorization, and floor level.

        FIX: added an optional `nonce`. The spatial component is still a lossy
        5-digit hash of the centroid (100,000 buckets), so two genuinely
        different properties CAN still map to the same bucket -- this alone
        does not make collisions impossible. What it does is give the caller
        (CadastreDatabaseEngine.register_property) a way to deterministically
        request a *different* candidate ULPIN for the same real-world unit
        when it detects that the first candidate is already taken by a
        DIFFERENT unit_id, so it can retry instead of silently merging two
        properties under one legal identifier. nonce=0 reproduces the
        original behavior exactly, so re-registering the same unit with the
        same inputs still yields the same ULPIN (idempotent) unless a retry
        was needed.
        """
        # 1. Quantize coordinates (+ retry nonce) into a fixed 5-digit spatial hash block
        coord_str = f"{centroid_x:.4f},{centroid_y:.4f},{nonce}"
        spatial_hash = int(hashlib.md5(coord_str.encode()).hexdigest(), 16) % 100000
        spatial_part = f"{spatial_hash:05d}"

        # 2. Map tier type code
        tier_mapping = {
            "SURFACE": "01",
            "SUBSURFACE": "02",
            "AIR_RIGHTS": "03"
        }
        tier_code = tier_mapping.get(tier_type.upper(), "01")

        # 3. Format floor level
        if floor_level < 0:
            floor_part = f"B{abs(floor_level):02d}"
        else:
            floor_part = f"{floor_level:03d}"

        # Assemble standard 14-character unique alphanumeric identifier
        ulpin_14 = f"{state_code}{district_code}{spatial_part}{tier_code}{floor_part}"
        return ulpin_14