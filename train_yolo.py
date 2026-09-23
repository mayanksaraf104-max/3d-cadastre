import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import yaml
from ultralytics import YOLO

import config

CHECKPOINT = config.YOLO_TRAINING_CHECKPOINT_PATH
DATA_YAML = "data/floor_plans/data.yaml"
EXPECTED_NAMES = ["residential_room", "common_corridor", "wall"]

def train_cadastre_model():
    print("🚀 Initializing YOLOv8 fine-tuning from existing cadastre checkpoint...")

    if not os.path.isfile(CHECKPOINT):
        raise FileNotFoundError(
            f"Checkpoint not found: {CHECKPOINT} (refusing to train from scratch)"
        )
    model = YOLO(CHECKPOINT)

    # Verify segmentation task and class names before training; a mismatch
    # would make Ultralytics rebuild the head instead of fine-tuning.
    with open(DATA_YAML) as f:
        data_names = list(yaml.safe_load(f)["names"])
    model_names = list(model.names.values())
    if (model.task != "segment" or model_names != data_names
            or model_names != EXPECTED_NAMES):
        raise ValueError(
            f"Checkpoint/dataset mismatch: task={model.task}, "
            f"model={model_names}, data.yaml={data_names}, "
            f"expected segment/{EXPECTED_NAMES}"
        )

    results = model.train(
        data=DATA_YAML,
        epochs=50,
        imgsz=640,
        batch=8,
        device=0,
        resume=False,                          # new fine-tuning run
        name="cadastre_room_finetune",         # separate from sih_gpu_model
        exist_ok=False                         # never reuse/overwrite a run dir
    )

    print("✅ Training successfully completed!")

if __name__ == "__main__":
    train_cadastre_model()