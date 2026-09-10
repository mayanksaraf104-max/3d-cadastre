import os
import glob
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from ultralytics import YOLO

def find_and_test_model():
    print("🔍 Searching for trained YOLOv8 weights...")
    
    # Dynamically find any 'best.pt' file inside the runs/segment directory
    weight_matches = glob.glob("runs/segment/**/best.pt", recursive=True)
    
    if not weight_matches:
        print("❌ Error: No trained 'best.pt' weights found anywhere under 'runs/segment/'. Did training complete?")
        return
        
    weights_path = weight_matches[-1] # Pick the most recent one
    print(f"✅ Found weights at: {weights_path}")
    
    model = YOLO(weights_path)
    
    test_dir = "floor-plan-object-detection/*" # or your test image path
    image_paths = glob.glob("data/floor_plans/images/val/*")
    
    if not image_paths:
        print("⚠️ No test images found in validation folder.")
        return
        
    print(f"🖼️ Running inference on {len(image_paths)} images...")
    results = model(image_paths, save=True, show=False)
    print("✅ Testing complete! Check 'runs/predict/' for outputs.")

if __name__ == "__main__":
    find_and_test_model()