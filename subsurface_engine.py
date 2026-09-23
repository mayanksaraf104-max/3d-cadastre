import numpy as np
from OCC.Core.BRepExtrema import BRepExtrema_DistShapeShape
from OCC.Core.BRepBndLib import brepbndlib_Add
from OCC.Core.BRepCheck import BRepCheck_Analyzer
from OCC.Core.Bnd import Bnd_Box
from OCC.Core.TopAbs import TopAbs_SOLID
from OCC.Core.TopExp import TopExp_Explorer


class True3DSubsurfaceEngine:
    def __init__(self, min_clearance_meters=1.5):
        self.min_clearance = min_clearance_meters

    def require_measured_solid(self, shape, label="geometry"):
        """
        Gate for independently measured / validated 3D B-Rep (XYZ-derived)
        solids. This engine never generates, sweeps, extrudes, repositions
        or reconstructs geometry from waypoints, radii, footprints, z-ranges
        or bounding boxes: the solid must be supplied as-is, or it is rejected.

        Raises ValueError if the geometry is missing, contains no B-Rep
        solid, or fails OCC's topological/geometric validity check.
        """
        if shape is None or shape.IsNull():
            raise ValueError(f"{label}: missing geometry.")
        if not TopExp_Explorer(shape, TopAbs_SOLID).More():
            raise ValueError(f"{label}: not a B-Rep solid.")
        if not BRepCheck_Analyzer(shape).IsValid():
            raise ValueError(f"{label}: invalid B-Rep solid.")
        return shape

    def validate_3d_clearance(self, asset_id, new_solid, existing_solids, surface_z=None):
        """
        Validates underground infrastructure using True 3D Euclidean algorithms.

        surface_z: optional, ADVISORY screening hint only (scalar elevation in
        the same vertical datum as the solids). It is not authoritative
        terrain/subsurface geometry and never changes APPROVED/REJECTED; a
        suspected breach is only reported under the "screening" key.
        Authoritative surface checks need a supplied measured 3D terrain B-Rep.
        """
        print(f"\n🚇 Running True 3D Subsurface Clearance Protocol for: {asset_id}")

        # 0. Reject missing / invalid geometry (nothing is repaired or rebuilt)
        try:
            self.require_measured_solid(new_solid, asset_id)
            for existing_id, existing_solid in existing_solids.items():
                self.require_measured_solid(existing_solid, existing_id)
        except ValueError as e:
            return {"status": "REJECTED", "reason": f"Invalid geometry: {e}"}

        # 1. Advisory screening only (non-authoritative; never rejects)
        screening = []
        if surface_z is not None:
            bbox = Bnd_Box()
            brepbndlib_Add(new_solid, bbox)
            xmin, ymin, zmin, xmax, ymax, zmax = bbox.Get()

            if zmax > surface_z:
                screening.append(f"Possible breach of scalar surface_z {surface_z:.2f}m (bbox top: {zmax:.2f}m); verify against measured 3D terrain.")

        # 2. Rule 2: 3D Multi-Tier Clearance Buffer
        for existing_id, existing_solid in existing_solids.items():
            # Calculate the mathematically exact 3D distance between the complex irregular geometries
            dist_calculator = BRepExtrema_DistShapeShape(new_solid, existing_solid)
            dist_calculator.Perform()

            if dist_calculator.IsDone():
                exact_3d_distance = dist_calculator.Value()

                if exact_3d_distance < self.min_clearance:
                    return {
                        "status": "REJECTED",
                        "reason": f"Clearance Violation: Too close to {existing_id} (Distance: {exact_3d_distance:.2f}m < Req: {self.min_clearance}m)"
                    }

        return {"status": "APPROVED", "reason": "Passed strict 3D volumetric clearance rules.", "screening": screening}


# --- Integration Test (measured solids only; no synthetic geometry) ---
# Usage: python subsurface_engine.py NEW.step ID=EXISTING.step [ID=EXISTING.step ...] [--surface-z Z]
if __name__ == "__main__":
    import argparse
    from OCC.Core.STEPControl import STEPControl_Reader
    from OCC.Core.IFSelect import IFSelect_RetDone

    def load_step(path):
        reader = STEPControl_Reader()
        if reader.ReadFile(path) != IFSelect_RetDone:
            raise SystemExit(f"Cannot read STEP file: {path}")
        reader.TransferRoots()
        return reader.OneShape()

    ap = argparse.ArgumentParser()
    ap.add_argument("new_step")
    ap.add_argument("existing", nargs="+", help="ID=path.step")
    ap.add_argument("--surface-z", type=float, default=None)
    args = ap.parse_args()

    engine = True3DSubsurfaceEngine(min_clearance_meters=2.0)
    existing_assets = {}
    for item in args.existing:
        asset_id, path = item.split("=", 1)
        existing_assets[asset_id] = load_step(path)

    result = engine.validate_3d_clearance("NEW_ASSET", load_step(args.new_step), existing_assets, args.surface_z)
    print(f"Result: {result['status']} – {result['reason']}")