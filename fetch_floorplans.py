import os
import cv2
import numpy as np
import yaml

def create_yolo_dataset():
    print("🚀 Creating local Floor Plan dataset for YOLOv8...")
    
    # 1. Setup YOLO Directory Structure
    dirs = [
        "data/floor_plans/images/train",
        "data/floor_plans/images/val",
        "data/floor_plans/labels/train",
        "data/floor_plans/labels/val"
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
        
    # 2. Generate Synthetic Architectural Floor Plans
    def generate_sample(img_path, txt_path):
        img = np.ones((512, 512, 3), dtype=np.uint8) * 240
        labels = []
        
        # Room 1 (RESIDENTIAL) - Class 0
        cv2.rectangle(img, (50, 50), (200, 250), (200, 220, 240), -1)
        labels.append("0 0.097 0.097 0.390 0.097 0.390 0.488 0.097 0.488")
        
        # Corridor (COMMERCIAL/COMMON) - Class 1
        cv2.rectangle(img, (200, 50), (250, 450), (240, 230, 200), -1)
        labels.append("1 0.390 0.097 0.488 0.097 0.488 0.878 0.390 0.878")
        
        # Room 2 (RESIDENTIAL) - Class 0
        cv2.rectangle(img, (250, 50), (450, 200), (200, 220, 240), -1)
        labels.append("0 0.488 0.097 0.878 0.097 0.878 0.390 0.488 0.390")
        
        # Draw structural walls (Black Lines)
        cv2.rectangle(img, (50, 50), (450, 450), (50, 50, 50), 4)
        
        # Save image and label file
        cv2.imwrite(img_path, img)
        with open(txt_path, "w") as f:
            f.write("\n".join(labels))

    print("📥 Generating training images...")
    for i in range(25):
        generate_sample(f"data/floor_plans/images/train/sample_{i}.png", 
                        f"data/floor_plans/labels/train/sample_{i}.txt")
                        
    print("📥 Generating validation images...")
    for i in range(5):
        generate_sample(f"data/floor_plans/images/val/sample_{i}.png", 
                        f"data/floor_plans/labels/val/sample_{i}.txt")

    # 3. Create YOLOv8 configuration file (data.yaml)
    yaml_data = {
        "path": os.path.abspath("data/floor_plans"),
        "train": "images/train",
        "val": "images/val",
        "nc": 2,
        "names": ["residential_room", "common_corridor"]
    }
    
    with open("data/floor_plans/data.yaml", "w") as f:
        yaml.dump(yaml_data, f, default_flow_style=False)
        
    print("✅ Dataset successfully generated at data/floor_plans!")
    print("✅ data.yaml is ready for YOLOv8 training.")

if __name__ == "__main__":
    create_yolo_dataset()