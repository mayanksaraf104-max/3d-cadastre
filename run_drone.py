import os
import cv2
import numpy as np
from shapely.geometry import Polygon

# We import main as a full module so we can override its internal namespace
import main
import ai_to_ogc


def extract_drone_rooftops(image_path):
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
            drone_units.append({"polygon": Polygon(coords), "confidence": 0.95})

    print(f"✅ Extracted {len(drone_units)} physical building footprints from Drone Image.")
    return drone_units


if __name__ == "__main__":
    drone_img = "drone_sample.jpg"
    drone_laz = "data/raw_lidar/san_francisco_3dep_sample.laz"

    if not os.path.exists(drone_img):
        print(f"❌ Please place a '{drone_img}' in your main folder first!")
    else:
        rooftops = extract_drone_rooftops(drone_img)

        if rooftops:
            # ==========================================================
            # 🛑 THE FIX: Patch the function directly inside `main`!
            # ==========================================================
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
            lidar_path = drone_laz if os.path.exists(drone_laz) else None
            main.run_unified_cadastre_pipeline(drone_img, lidar_path=lidar_path)

            print("\n🚁 DRONE TEST COMPLETE: Check approved_cadastre.glb for your real-world buildings!")