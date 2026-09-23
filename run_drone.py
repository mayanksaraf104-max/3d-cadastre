"""
run_drone.py -- drone-imagery driver for the cadastre pipeline.

TRUE-3D CONTRACT
----------------
* Drone rooftop polygons (from the AI/OGC model OR the OpenCV fallback) are 2D
  reference / semantic / validation evidence ONLY. They never create, position,
  trim, extrude, or determine any 3D geometry or Z; they are kept for
  attribution and validation.
* Authoritative 3D geometry comes only from measured LiDAR XYZ. With no such
  source the run is aborted rather than producing a 2D-only result.
"""
import argparse
import os
import cv2
import numpy as np
from shapely.geometry import Polygon

# We import main as a full module so we can override its internal namespace
import main
import ai_to_ogc

# The OpenCV threshold fallback is a crude heuristic, not a trained model and
# certainly not measured 3D evidence, so it gets an explicitly low confidence.
CV_FALLBACK_CONFIDENCE = 0.3


def extract_drone_rooftops(image_path):
    """
    Returns 2D rooftop polygons (pixel space) as reference/semantic/validation
    evidence ONLY -- never 3D geometry and never a source of Z.
    """
    print("\n🚁 [DRONE MODE] Scanning Aerial Orthomosaic...")

    # 1. Try the AI model first
    ai_units = ai_to_ogc.extract_ogc_boundaries(image_path)
    if ai_units and len(ai_units) > 0:
        print("✅ AI successfully segmented drone rooftops!")
        return ai_units

    print("⚠️ Blueprint AI model didn't recognize real rooftops. Engaging CV Fallback...")

    # 2. Computer Vision Fallback
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    drone_units = []
    for cnt in contours:
        epsilon = 0.02 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)

        # Filter for building-sized polygons
        if len(approx) >= 3 and cv2.contourArea(cnt) > 1000:
            coords = [(float(pt[0][0]), float(pt[0][1])) for pt in approx]
            coords.append(coords[0])
            drone_units.append({
                "polygon": Polygon(coords),
                "confidence": CV_FALLBACK_CONFIDENCE,
                "source": "cv_threshold_heuristic",   # CV-derived, not measured 3D evidence
            })

    print(f"✅ Extracted {len(drone_units)} physical building footprints from Drone Image.")
    return drone_units


if __name__ == "__main__":
    # main.run_unified_cadastre_pipeline() has NO default location, scale or
    # jurisdiction: it raises ValueError without them, so they are required here.
    parser = argparse.ArgumentParser(description="Drone imagery -> 3D cadastre run")
    parser.add_argument("--image", default="drone_sample.jpg")
    parser.add_argument("--lidar", default="data/raw_lidar/san_francisco_3dep_sample.laz")
    parser.add_argument("--site-lat", type=float, required=True,
                        help="latitude of the image's top-left pixel (0,0)")
    parser.add_argument("--site-lon", type=float, required=True,
                        help="longitude of the image's top-left pixel (0,0)")
    parser.add_argument("--pixel-scale-m", type=float, required=True,
                        help="real-world metres per image pixel (image assumed north-up)")
    parser.add_argument("--state-code", required=True, help="ULPIN state code")
    parser.add_argument("--district-code", required=True, help="ULPIN district code")
    args = parser.parse_args()
    drone_img, drone_laz = args.image, args.lidar

    if not os.path.exists(drone_img):
        print(f"❌ Please place a '{drone_img}' in your main folder first!")
    elif not os.path.exists(drone_laz):
        # No measured XYZ source: refuse a 2D-only result.
        raise SystemExit(
            f"❌ No measured LiDAR XYZ source at '{drone_laz}'. Aborting: drone "
            f"polygons are 2D reference evidence only and cannot yield a 3D cadastral run."
        )
    else:
        rooftops = extract_drone_rooftops(drone_img)

        if rooftops:
            # ==========================================================
            # 🛑 THE FIX: Patch the function directly inside `main`!
            # ==========================================================
            # (`rooftops` are 2D reference evidence only; measured LiDAR XYZ
            # is the sole authority for any 3D geometry.)
            main.extract_ogc_boundaries = lambda x: rooftops

            # FIX: main.run_unified_cadastre_pipeline() only ever accepted
            # ONE positional argument (image_path). Calling it here with
            # a second positional arg (the .laz path) raised a TypeError
            # before the pipeline did anything, which is why results
            # looked broken. main.py now accepts an explicit optional
            # `lidar_path` keyword, which z_engine.calculate_z_bounds()
            # uses directly instead of querying the PostGIS spatial index
            # -- so this actually feeds your real LiDAR file into the
            # roof-fitting step now, instead of being silently dropped.
            lidar_path = drone_laz
            # No floors/floor_h override is passed: vertical structure must
            # come from evidence, never from a typed-in guess.
            result = main.run_unified_cadastre_pipeline(
                drone_img,
                site_lat=args.site_lat, site_lon=args.site_lon,
                pixel_scale_m=args.pixel_scale_m,
                state_code=args.state_code, district_code=args.district_code,
                lidar_path=lidar_path,
            )

            # Report what the pipeline actually did instead of always claiming success.
            if not result or result.get("registered_count", 0) == 0:
                raise SystemExit(
                    "❌ DRONE RUN REGISTERED NOTHING: "
                    f"{(result or {}).get('error', 'no units were reconstructed from measured evidence')} "
                    f"(failed_units={(result or {}).get('failed_units', [])})"
                )
            print(f"\n🚁 DRONE RUN COMPLETE: registered {result['registered_count']} unit(s); "
                  f"failed: {result['failed_units']}. Check approved_cadastre.glb.")
        else:
            raise SystemExit("❌ No rooftop polygons were extracted from the drone image.")