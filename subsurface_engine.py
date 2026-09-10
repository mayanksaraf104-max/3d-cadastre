import numpy as np
from OCC.Core.gp import gp_Pnt, gp_Vec, gp_Dir, gp_Ax2, gp_Circ
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeEdge, BRepBuilderAPI_MakeWire, BRepBuilderAPI_MakeFace
from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakePrism
from OCC.Core.BRepOffsetAPI import BRepOffsetAPI_MakePipe
from OCC.Core.TColgp import TColgp_Array1OfPnt
from OCC.Core.GeomAPI import GeomAPI_PointsToBSpline
from OCC.Core.BRepExtrema import BRepExtrema_DistShapeShape
from OCC.Core.BRepBndLib import brepbndlib_Add
from OCC.Core.Bnd import Bnd_Box
from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh


class True3DSubsurfaceEngine:
    def __init__(self, min_clearance_meters=1.5):
        self.min_clearance = min_clearance_meters

    def generate_cylindrical_tunnel(self, waypoints_3d, radius):
        """
        Generates a True 3D curved cylinder (like a subway tunnel or pipeline)
        and tessellates it for OGC PolyhedralSurface Z compliance.
        """
        # 1. Weave a mathematical B-Spline curve through the 3D waypoints
        pt_array = TColgp_Array1OfPnt(1, len(waypoints_3d))
        for i, (x, y, z) in enumerate(waypoints_3d, start=1):
            pt_array.SetValue(i, gp_Pnt(float(x), float(y), float(z)))

        spline_curve = GeomAPI_PointsToBSpline(pt_array).Curve()
        spline_edge = BRepBuilderAPI_MakeEdge(spline_curve).Edge()
        spline_wire = BRepBuilderAPI_MakeWire(spline_edge).Wire()

        # 2. Create the circular profile at the start of the tunnel.
        #
        # FIX: the profile's normal was previously hardcoded to the
        # global X-axis (gp_Dir(1, 0, 0)) regardless of which direction
        # the tunnel actually runs. BRepOffsetAPI_MakePipe expects the
        # profile face to sit roughly transverse to the spine's tangent
        # at the start point -- for anything that isn't running along
        # global X (e.g. this pipeline's diagonal waypoints), that
        # produces a skewed profile and risks a self-intersecting or
        # degenerate swept solid (which then makes downstream volume /
        # distance / intersection checks on that solid unreliable).
        #
        # This now derives the actual tangent direction from the curve's
        # first derivative at its start parameter, so the profile plane
        # is genuinely perpendicular to the path.
        start_pt = gp_Pnt()
        start_tangent = gp_Vec()
        spline_curve.D1(spline_curve.FirstParameter(), start_pt, start_tangent)

        if start_tangent.Magnitude() < 1e-9:
            # Degenerate tangent (e.g. duplicate leading waypoints) --
            # fall back to the old default rather than crashing on
            # gp_Dir's zero-vector check.
            tangent_dir = gp_Dir(1, 0, 0)
        else:
            tangent_dir = gp_Dir(start_tangent)

        circle_axis = gp_Ax2(start_pt, tangent_dir)
        circle = gp_Circ(circle_axis, float(radius))

        circle_edge = BRepBuilderAPI_MakeEdge(circle).Edge()
        circle_wire = BRepBuilderAPI_MakeWire(circle_edge).Wire()
        profile_face = BRepBuilderAPI_MakeFace(circle_wire).Face()

        # 3. Sweep the circular face along the 3D curved wire!
        tunnel_solid = BRepOffsetAPI_MakePipe(spline_wire, profile_face).Shape()

        # 4. Tessellate into OGC Triangles (TIN Z)
        BRepMesh_IncrementalMesh(tunnel_solid, 0.5)

        return tunnel_solid

    def validate_3d_clearance(self, asset_id, new_solid, existing_solids):
        """
        Validates underground infrastructure using True 3D Euclidean algorithms.
        """
        print(f"\n🚇 Running True 3D Subsurface Clearance Protocol for: {asset_id}")

        # 1. Rule 1: Must Reside Subsurface (Z <= 0)
        bbox = Bnd_Box()
        brepbndlib_Add(new_solid, bbox)
        xmin, ymin, zmin, xmax, ymax, zmax = bbox.Get()

        if zmax > 0:
            return {"status": "REJECTED", "reason": f"Breaches surface ground level (Highest Point: {zmax:.2f}m)."}

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

        return {"status": "APPROVED", "reason": "Passed strict 3D volumetric clearance rules."}


# --- Integration Test ---
if __name__ == "__main__":
    engine = True3DSubsurfaceEngine(min_clearance_meters=2.0)

    # 1. Generate an existing curved METRO TUNNEL dipping deep underground
    metro_waypoints = [(0, 50, -10), (25, 50, -15), (50, 50, -10)]
    existing_metro = engine.generate_cylindrical_tunnel(metro_waypoints, radius=4.0)
    existing_assets = {"METRO_LINE_01": existing_metro}

    # 2. Generate a new straight WATER PIPELINE intersecting directly above it
    pipe_waypoints = [(25, 0, -8), (25, 25, -8), (25, 100, -8)]
    new_pipe = engine.generate_cylindrical_tunnel(pipe_waypoints, radius=1.0)

    print("\n[TEST 1] Testing close-proximity water pipe vs existing metro line...")
    result = engine.validate_3d_clearance("WATER_PIPE_04", new_pipe, existing_assets)
    print(f"Result: {result['status']} – {result['reason']}")