from ultralytics import YOLO


def train_hybrid_model():
  # Load your previous GPU weights to fine-tune on the combined dataset
  model = YOLO("runs/segment/runs/sih_gpu_model/weights/best.pt")
  yaml_path = "data/floor_plans/data.yaml"
  print("started")
  results = model.train(
      data=yaml_path,
      epochs=50,  # Additional fine-tuning epochs on mixed data
      imgsz=640,
      batch=16,
      device=0,  # Targets your RTX 4060 GPU
      box=7.5,
      cls=1.5,
      project="runs",
      name="sih_hybrid_model",
  )

if __name__ == "__main__":
  train_hybrid_model()