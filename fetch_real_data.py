"""2D floor-plan training-data ingestion (CubiCasa5k, COCO format).

Floor-plan images and their COCO annotations (rooms / walls / etc.) are
saved UNMODIFIED as 2D semantic / reference training data only. They are
NOT measured 3D geometry and carry no height, elevation, or floor-count
evidence; nothing here generates or infers any 3D information.
"""

import json
import os
from datasets import Image as HFImage, load_dataset
from tqdm import tqdm

def download_2d_floorplan_training_data():
  print(
      "🌍 Connecting to Hugging Face to download 2D floor-plan"
      " training/reference images and their COCO annotations..."
  )

  os.makedirs("data/floor_plans/images/train", exist_ok=True)
  os.makedirs("data/floor_plans/images/val", exist_ok=True)
  os.makedirs("data/floor_plans/annotations/train", exist_ok=True)

  try:
    # Pulling a 2D floor-plan image stream from a public repository
    dataset = load_dataset(
        "phungpx/cubicassa5k-coco", split="train", streaming=True
    )
    # decode=False -> raw stored image bytes (no decode / re-encode / recolor)
    dataset = dataset.cast_column("image", HFImage(decode=False))

    count = 0
    skipped = 0
    warned_no_annotations = False
    max_samples = 5000

    print("📥 Saving raw 2D floor-plan images + COCO annotations into training directories...")
    for sample in tqdm(dataset):
      if count >= max_samples:
        break

      image = sample.get("image") or {}
      raw = image.get("bytes")
      if not raw:
        skipped += 1
        continue

      # Everything except the image (e.g. `annotations`: category_id, bbox,
      # segmentation, ...) is kept verbatim as 2D semantic labels.
      annotations = {k: v for k, v in sample.items() if k != "image"}
      if not annotations and not warned_no_annotations:
        print("⚠️  Dataset rows contain no annotation fields; saving images only.")
        warned_no_annotations = True

      stem = f"real_cadastre_{count}"
      ext = os.path.splitext(image.get("path") or "")[1].lower()
      if not ext:
        ext = ".png" if raw.startswith(b"\x89PNG") else ".jpg"

      with open(f"data/floor_plans/images/train/{stem}{ext}", "wb") as f:
        f.write(raw)
      if annotations:
        with open(
            f"data/floor_plans/annotations/train/{stem}.json", "w"
        ) as f:
          json.dump(annotations, f, default=str)
      count += 1

    print(
        f"✅ Saved {count} raw 2D floor-plan training/reference images with"
        f" COCO annotations ({skipped} skipped: no image bytes). This is 2D"
        " semantic data only, not 3D geometry or height evidence."
    )

  except Exception as e:
    print(f"❌ Error connecting to data source: {e}. No data was downloaded.")


if __name__ == "__main__":
  download_2d_floorplan_training_data()