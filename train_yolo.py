import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from ultralytics import YOLO

def train_cadastre_model():
    print("🚀 Initializing YOLOv8 model for architectural footprint segmentation...")
    
    model = YOLO("yolov8n-seg.pt")
    
    results = model.train(
        data="data/floor_plans/data.yaml",  
        epochs=50,                          
        imgsz=640,                          
        batch=8,                            
        device=0,                       
        name="cadastre_room_segmentation"   
    )
    
    print("✅ Training successfully completed!")

if __name__ == "__main__":
    train_cadastre_model()