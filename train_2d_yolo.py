import os

import yaml
from ultralytics import YOLO

import config

CHECKPOINT = config.YOLO_TRAINING_CHECKPOINT_PATH
EXPECTED_NAMES = ["residential_room", "common_corridor", "wall"]


def train_hybrid_model():
  # Load your previous GPU weights to fine-tune on the combined dataset
  if not os.path.isfile(CHECKPOINT):
    raise FileNotFoundError(
        f"Checkpoint not found: {CHECKPOINT} (refusing to train from scratch)"
    )
  model = YOLO(CHECKPOINT)
  yaml_path = "data/floor_plans/data.yaml"

  # Guard: a class/task mismatch would make Ultralytics silently rebuild the
  # head instead of fine-tuning the existing 3-class segmentation model.
  with open(yaml_path) as f:
    data_names = list(yaml.safe_load(f)["names"])
  model_names = list(model.names.values())
  if model.task != "segment" or model_names != EXPECTED_NAMES \
      or data_names != EXPECTED_NAMES:
    raise ValueError(
        f"Class/task mismatch: task={model.task}, model={model_names}, "
        f"data.yaml={data_names}, expected segment/{EXPECTED_NAMES}"
    )

  print("started")
  results = model.train(
      data=yaml_path,
      epochs=50,  # Additional fine-tuning epochs on mixed data
      imgsz=640,
      batch=16,
      device=0,  # Targets your RTX 4060 GPU
      box=7.5,
      cls=1.5,
      seed=42,  # Reproducible run
      deterministic=True,
      resume=False,  # Fine-tune from best.pt; not a resume of an old run
      project="runs",
      name="sih_hybrid_model",  # Separate from sih_gpu_model
      exist_ok=False,  # Never reuse/overwrite an existing run directory
  )

if __name__ == "__main__":
  train_hybrid_model()