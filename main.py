import os
import cv2
import numpy as np
from shapely.geometry import Polygon
from ai_to_ogc import extract_ogc_boundaries
from z_engine import calculate_z_bounds
from db_engine import CadastreDatabaseEngine
from export_ledger import export_postgis_to_glb
from subsurface_engine import True3DSubsurfaceEngine

# Advanced OpenCASCADE Imports
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopAbs import TopAbs_FACE, TopAbs_REVERSED
from OCC.Core.TopoDS import topods
from OCC.Core.BRep import BRep_Tool
from OCC.Core.TopLoc import TopLoc_Location
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakePolygon, BRepBuilderAPI_MakeFace, BRepBuilderAPI_GTransform
from OCC.Core.gp import gp_Pnt, gp_Vec, gp_GTrsf, gp_Mat, gp_XYZ
from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakePrism, BRepPrimAPI_MakeHalfSpace
from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Common
from OCC.Core.TColgp import TColgp_Array2OfPnt
from OCC.Core.GeomAPI import GeomAPI_PointsToBSplineSurface
from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh


class GNSSCoordinateAnchor:
    """Calculates the Affine Transformation Matrix using Ground Control Points (GCPs)"""
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
        """
        FIX: the pipeline used to run calculate_z_bounds() and build all
        OCC geometry directly on the RAW PIXEL polygon (e.g. coordinates
        in the 0-1000 range from the source image), and only applied the
        GNSS affine transform at the very end, to the finished 3D solid.

        But z_engine.get_intersecting_lidar_tiles() queries PostGIS
        assuming the polygon it's given is already in real-world
        coordinates (it tags it `SRID=...` and intersects it against
        lidar_tile_index, which is indexed in real-world coordinates).
        A pixel-space polygon interpreted that way lands nowhere near the
        actual tiles, so the LiDAR lookup silently returned zero tiles
        every time and every roof fell back to a flat prism -- regardless
        of how good the LiDAR coverage actually was.

        The fix: transform the footprint polygon into global coordinates
        FIRST, then do LiDAR lookup / roof fitting / OCC geometry
        construction entirely in global space. There is then no need to
        GTransform the finished solid at the end (except for geometry,
        like the subsurface tunnel, that intentionally builds directly
        from already-global waypoints).
        """
        coords = [self.transform_xy(x, y) for x, y in poly.exterior.coords]
        holes = [
            [self.transform_xy(x, y) for x, y in interior.coords]
            for interior in poly.interiors
        ]
        return Polygon(coords, holes)


def robust_solid_to_wkt(shape, global_mirror=False):
    """
    Extracts the tessellated mesh triangles directly from a True 3D solid
    and guarantees a perfectly formatted OGC POLYHEDRALSURFACE Z string.

    FIX (part 1): BRep_Tool.Triangulation() returns each face's raw mesh in
    the winding order of its underlying surface parametrization -- it does
    NOT respect the face's TopAbs_Orientation() flag. Faces OCC marked
    REVERSED (common as a byproduct of BRepAlgoAPI_Common, run per-unit to
    intersect the tall prism with the roof half-space) need their winding
    flipped on export.

    FIX (part 2): `global_mirror` should be True if this solid was produced
    by a transform with a NEGATIVE determinant (a reflection, not just a
    rotation/translation) -- which is exactly what this pipeline's GNSS
    affine does, since mapping image +Y (down) to UTM northing (up) is
    mathematically a mirror. A mirror flips a solid's handedness globally,
    on top of whatever each face's local Orientation() flag already says.
    The two effects are combined with XOR: a face needs its winding
    flipped if exactly one of (locally reversed, globally mirrored) is
    true, and left alone if both or neither are true.
    """
    polygons = []
    explorer = TopExp_Explorer(shape, TopAbs_FACE)

    while explorer.More():
        face = topods.Face(explorer.Current())
        is_reversed = face.Orientation() == TopAbs_REVERSED
        needs_flip = is_reversed != global_mirror  # XOR

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

                # A valid OGC polygon MUST close on itself (P1 -> P2 -> P3 -> P1)
                poly = (f"(({p1.X():.4f} {p1.Y():.4f} {p1.Z():.4f}, "
                        f"{p2.X():.4f} {p2.Y():.4f} {p2.Z():.4f}, "
                        f"{p3.X():.4f} {p3.Y():.4f} {p3.Z():.4f}, "
                        f"{p1.X():.4f} {p1.Y():.4f} {p1.Z():.4f}))")
                polygons.append(poly)

        explorer.Next()

    if not polygons:
        return None

    return "POLYHEDRALSURFACE Z (" + ", ".join(polygons) + ")"


def extract_smart_boundaries(image_path):
    """
    Intelligent Extractor: Tries AI for standard blueprints.
    If the AI fails, it dynamically falls back to CV for Drone Aerial Imagery.
    """
    print("\n[STEP 1] Running Intelligent Vectorization...")
    ai_units = extract_ogc_boundaries(image_path)

    if ai_units and len(ai_units) > 0:
        print(f"✅ AI successfully extracted {len(ai_units)} spatial boundaries.")
        return ai_units

    print("⚠️ Blueprint AI model detected zero units. Engaging Drone/Aerial CV Fallback...")
    img = cv2.imread(image_path)
    if img is None:
        print(f"❌ Error: Could not read image at {image_path}")
        return []

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    fallback_units = []
    for cnt in contours:
        # Approximate the contour to preserve irregular/wavy building boundaries
        epsilon = 0.02 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)

        if len(approx) >= 3 and cv2.contourArea(cnt) > 1000:
            coords = [(float(pt[0][0]), float(pt[0][1])) for pt in approx]
            coords.append(coords[0])
            
            # --- THE FIX: Sharp Corners (Mitre Join) & Vertex Decimation ---
            raw_poly = Polygon(coords)
            
            # join_style=2 forces sharp corners. simplify(2.0) strips out jagged OpenCV pixel noise.
            clean_poly = raw_poly.buffer(-5.0, join_style=2).simplify(2.0, preserve_topology=True)
            
            if not clean_poly.is_empty and clean_poly.geom_type == 'Polygon':
                fallback_units.append({"polygon": clean_poly, "confidence": 0.95})
    print(f"✅ Drone CV Fallback extracted {len(fallback_units)} physical building footprints.")
    return fallback_units


def _build_polygon_wire(coords, z):
    """Builds a single closed OCC wire from a list of (x, y) coords at height z."""
    mk = BRepBuilderAPI_MakePolygon()
    for x, y in coords[:-1] if coords[0] == coords[-1] else coords:
        mk.Add(gp_Pnt(float(x), float(y), float(z)))
    mk.Close()
    return mk.Wire()


def build_face_with_holes(poly: Polygon, z: float):
    """
    FIX: the original pipeline only ever read `poly.exterior.coords` when
    building the extrusion face. shapely polygons produced by
    ai_to_ogc.make_valid() can legitimately have interior rings (holes) --
    e.g. a courtyard or light well -- and those were silently dropped,
    so any unit with a real void in its footprint got extruded as a
    solid, filled-in prism.

    This builds the outer wire as before, then adds each interior ring as
    a hole (reversed, as OCC expects for a face's inner wires) so voids
    survive into the 3D solid.
    """
    outer_wire = _build_polygon_wire(list(poly.exterior.coords), z)
    face_builder = BRepBuilderAPI_MakeFace(outer_wire)

    for interior in poly.interiors:
        hole_wire = _build_polygon_wire(list(interior.coords), z)
        face_builder.Add(hole_wire.Reversed())

    return face_builder.Face()


def run_unified_cadastre_pipeline(image_path, lidar_path=None, include_demo_subsurface=False):
    """
    `lidar_path`: optional explicit .laz file (e.g. from run_drone.py). If
    given, every unit's Z/roof fitting uses this file directly instead of
    querying the PostGIS lidar_tile_index for overlapping tiles.

    `include_demo_subsurface`: if True, also generates and registers a
    fixed demo tunnel ("METRO_LINE_A") using hardcoded waypoints, as a
    showcase of the subsurface engine. Defaults to False so real
    pipeline runs on actual floor plans/drone imagery don't have an
    unrelated fake tunnel injected into every output. Pass True
    explicitly (e.g. for a demo/pitch run) to include it.
    """
    print("\n==================================================")
    print("🚀 RUNNING FULL 3D VERTICAL CADASTRE PIPELINE")
    print("==================================================")

    # ==========================================================
    # STEP 0: Establish GNSS / CORS Global Anchor
    # ==========================================================
    print("\n[STEP 0] Calibrating GNSS/CORS Global Coordinate Anchor...")
    pixels = np.array([[0, 0], [1000, 0], [0, 1000]])
    utm_coords = np.array([[552000.0, 4182000.0], [552500.0, 4182000.0], [552000.0, 4181500.0]])
    gnss = GNSSCoordinateAnchor(pixels, utm_coords)
    mat_vals = gnss.matrix

    # Create the OpenCASCADE General Affine Transformation. Still used for
    # geometry built directly with OCC primitives before we know the final
    # global coordinates (kept for reference / any future 3D-only inputs);
    # 2D footprints are now pre-transformed via gnss.transform_polygon().
    occ_mat = gp_Mat(mat_vals['a'], mat_vals['b'], 0.0,
                      mat_vals['d'], mat_vals['e'], 0.0,
                      0.0, 0.0, 1.0)
    global_trsf = gp_GTrsf()
    global_trsf.SetVectorialPart(occ_mat)
    global_trsf.SetTranslationPart(gp_XYZ(mat_vals['c'], mat_vals['f'], 0.0))
    print("   ✅ Affine Matrix Initialized: Translating Local Pixels to Global UTM.")

    # Detect whether this GCP fit is a reflection (negative determinant) --
    # e.g. mapping image +Y (down) to UTM northing (up), which is geographically
    # correct but flips solid handedness on export. Any solid built from
    # already-global-space coordinates below (which is now all of them,
    # since we transform the footprint up front) still needs this flag
    # passed to robust_solid_to_wkt at export time, because the mirroring
    # is baked into the coordinates themselves, not applied as a later OCC
    # transform.
    transform_is_mirror = (mat_vals['a'] * mat_vals['e'] - mat_vals['b'] * mat_vals['d']) < 0
    if transform_is_mirror:
        print("   ⚠️ GCP fit has negative determinant (reflection) — will compensate on WKT export.")

    raw_units = extract_smart_boundaries(image_path)
    if not raw_units:
        print("❌ Pipeline Aborted: No valid geometries found.")
        return

    solids = []
    unit_metadata = []

    print("\n[STEP 2] Building & Anchoring Multi-Storey Geometry...")
    for i, unit in enumerate(raw_units, start=1):
        # FIX: transform the footprint into GLOBAL coordinates BEFORE doing
        # anything with it -- LiDAR lookup, roof fitting, and OCC geometry
        # construction all now happen in global space, consistently.
        poly_pixel = unit["polygon"]
        poly_global = gnss.transform_polygon(poly_pixel)

        unit_id = f"AI_Unit_{i}_Floor_1"
        coords = list(poly_global.exterior.coords)

        # Dynamically calculate Z: queries PostGIS (or uses lidar_path
        # directly) for overlapping LiDAR tiles. poly_global is now in the
        # same coordinate space as lidar_tile_index, so this can actually
        # find matches.
        z_dem, poly_model, poly_features = calculate_z_bounds(poly_global, lidar_path=lidar_path)

        try:
            floor_face = build_face_with_holes(poly_global, z_dem)
            tall_prism = BRepPrimAPI_MakePrism(floor_face, gp_Vec(0, 0, 50.0)).Shape()

            if poly_model is not None:
                minx, miny, maxx, maxy = poly_global.bounds
                grid_size = 10
                grid_pts = TColgp_Array2OfPnt(1, grid_size, 1, grid_size)

                x_vals = np.linspace(minx - 2, maxx + 2, grid_size)
                y_vals = np.linspace(miny - 2, maxy + 2, grid_size)

                for row, x in enumerate(x_vals, start=1):
                    for col, y in enumerate(y_vals, start=1):
                        xy_input = poly_features.transform([[x, y]])
                        z_curve = poly_model.predict(xy_input)[0]
                        grid_pts.SetValue(row, col, gp_Pnt(float(x), float(y), float(z_curve)))

                bspline = GeomAPI_PointsToBSplineSurface(grid_pts).Surface()
                roof_face = BRepBuilderAPI_MakeFace(bspline, 1e-6).Face()

                ref_point = gp_Pnt(minx, miny, z_dem - 100.0)
                half_space = BRepPrimAPI_MakeHalfSpace(roof_face, ref_point).Solid()
                final_solid = BRepAlgoAPI_Common(tall_prism, half_space).Shape()
            else:
                # FIX: no LiDAR roof model available (typical for indoor
                # AI-blueprint units, which have no aerial LiDAR coverage).
                # Previously this always extruded a hardcoded 3.0m flat
                # box, silently discarding whatever z_top the AI model
                # actually predicted for the room/corridor. Now it uses
                # the unit's own z_top (falling back to 3.0 only if the
                # unit never provided one, e.g. the CV fallback path).
                ceiling_height = float(unit.get("z_top", z_dem + 3.0)) - z_dem
                if ceiling_height <= 0:
                    ceiling_height = 3.0
                final_solid = BRepPrimAPI_MakePrism(floor_face, gp_Vec(0, 0, ceiling_height)).Shape()

            # No GTransform needed here anymore -- floor_face was already
            # built directly in global coordinates.
            #
            # FIX: 0.5 deflection produces hundreds of small/sliver
            # triangles per solid once meshed -- fine for a smooth-looking
            # GLB, but this SAME triangulated geometry is also what gets
            # stored in PostGIS and fed into the ST_3DIntersection/
            # ST_MakeSolid clash check in db_engine.py. SFCGAL's exact
            # volumetric boolean cost scales badly with face count, and a
            # coarser mesh here (fewer, larger flat faces, same overall
            # shape) keeps clash checks fast without visibly changing the
            # unit's boundary. If you need a smoother GLB later, mesh a
            # SEPARATE fine copy for export_ledger.py rather than lowering
            # this value again.
            BRepMesh_IncrementalMesh(final_solid, 2.0)

            solids.append(final_solid)
            unit_metadata.append({"id": unit_id, "z_base": z_dem, "tier": "SURFACE", "mirror": transform_is_mirror})
            print(f"   ✔️ Extruded & Georeferenced: {unit_id}")

            # Stack a 2nd Floor on Unit 1
            if i == 1:
                air_rights_id = "AI_Unit_1_Floor_2"
                face_f2 = build_face_with_holes(poly_global, 20.0)
                solid_f2 = BRepPrimAPI_MakePrism(face_f2, gp_Vec(0, 0, 4.0)).Shape()

                BRepMesh_IncrementalMesh(solid_f2, 0.5)

                solids.append(solid_f2)
                unit_metadata.append({"id": air_rights_id, "z_base": 20.0, "tier": "AIR_RIGHTS", "mirror": transform_is_mirror})

        except Exception as e:
            print(f"⚠️ Failed to process {unit_id}: {e}")

    # ==========================================================
    # SUBSURFACE: Add a Curved Underground Tunnel (DEMO ONLY)
    # ==========================================================
    # This block was unconditional -- it ran on every single pipeline
    # call regardless of image_path, injecting the same fixed-waypoint
    # tunnel ("METRO_LINE_A") into every generated model even when the
    # input had nothing to do with a metro/subsurface asset. Now gated
    # behind include_demo_subsurface so it only appears when explicitly
    # requested (e.g. a demo/pitch run showcasing the subsurface engine).
    if include_demo_subsurface:
        print("\n[STEP 3] Generating True 3D Subsurface Infrastructure (demo)...")
        sub_engine = True3DSubsurfaceEngine(min_clearance_meters=2.0)

        tunnel_waypoints = [(100, 100, -10), (500, 500, -25), (900, 900, -10)]

        try:
            global_waypoints = []
            for x, y, z in tunnel_waypoints:
                global_X, global_Y = gnss.transform_xy(x, y)
                global_waypoints.append((global_X, global_Y, z))

            global_metro_solid = sub_engine.generate_cylindrical_tunnel(global_waypoints, radius=6.0)

            # FIX: use a coarser mesh deflection (2.0 instead of 0.5) for the
            # tunnel. The tunnel solid is large (radius=6, spanning ~1100m of
            # UTM space) so 0.5m deflection produces thousands of tiny triangles.
            # When that mesh is stored in PostGIS and later fed to ST_MakeSolid,
            # SFCGAL rejects it as "self-intersects" because many of those sliver
            # triangles are nearly degenerate at float64 precision. 2.0m deflection
            # produces far fewer, cleaner faces that SFCGAL can handle.
            BRepMesh_IncrementalMesh(global_metro_solid, 2.0)

            # Validate the WKT before pushing to DB — the tunnel is the most
            # likely geometry to produce a self-intersecting PolyhedralSurface.
            metro_wkt = robust_solid_to_wkt(global_metro_solid, global_mirror=False)
            if metro_wkt and "()" not in metro_wkt:
                solids.append(global_metro_solid)
                unit_metadata.append({"id": "METRO_LINE_A", "z_base": -25.0, "tier": "SUBSURFACE", "mirror": False})
                print("   🚇 Swept & Georeferenced: METRO_LINE_A")
            else:
                print("   ⚠️ METRO_LINE_A produced empty/invalid WKT — skipping registration.")
        except Exception as e:
            print(f"⚠️ Failed to generate Subsurface Tunnel: {e}")

    print("\n[STEP 4] Registering into National PostGIS Cadastre Ledger...")
    # FIX: use as a context manager so the connection is always closed --
    # including if an exception is raised partway through -- instead of
    # leaking a psycopg2 connection every time this pipeline runs (see
    # CadastreDatabaseEngine.close() for why that matters).
    registered_count = 0
    with CadastreDatabaseEngine() as db:
        for idx, solid in enumerate(solids):
            if idx >= len(unit_metadata):
                break
            unit_info = unit_metadata[idx]
            unit_id = unit_info["id"]
            tier = unit_info["tier"]

            wkt_string = robust_solid_to_wkt(solid, global_mirror=unit_info.get("mirror", False))

            if wkt_string and "()" not in wkt_string:
                floor_level = int(unit_info["z_base"] // 3) if tier != "SUBSURFACE" else -1
                ulpin = db.register_property(unit_id, wkt_string, tier_type=tier, floor_level=floor_level)
                if ulpin:
                    registered_count += 1

    print(f"\n🎉 Successfully registered {registered_count} Multi-Tier Georeferenced 3D assets.")

    print("\n[STEP 5] Exporting WebGL 3D Model (.glb)...")
    export_postgis_to_glb(output_filename="approved_cadastre.glb")
    print("✨ Pipeline execution fully complete!")


if __name__ == "__main__":
    input_image = "drone_sample.jpg"  # Or use "data/floor_plans/images/val/sample_0.png"

    if os.path.exists(input_image):
        run_unified_cadastre_pipeline(input_image)
    else:
        print(f"❌ Input image '{input_image}' not found.")