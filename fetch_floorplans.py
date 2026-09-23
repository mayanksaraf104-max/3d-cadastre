import os
import cv2
import numpy as np
import yaml

# Strictly 2D synthetic training data: 2D floor-plan images + normalized YOLO
# polygon labels. No 3D geometry, heights, floor counts or extrusion here.

IMG_SIZE = 512
N_TRAIN, N_VAL = 200, 40          # more, varied samples (was 25 / 5)
MIN_ROOM = 60                     # minimum room side before wall inset (px)
INCLUDE_WALL_CLASS = True         # label walls as their own class (2)

CLASS_RESIDENTIAL, CLASS_CORRIDOR, CLASS_WALL = 0, 1, 2


def _rect(x0, y0, x1, y1):
    return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=float)


def _split(rng, x0, y0, x1, y1, depth, rooms, cuts):
    """Randomly BSP-subdivide a region; leaves = rooms, cut lines = walls."""
    w, h = x1 - x0, y1 - y0
    can_x, can_y = w >= 2 * MIN_ROOM, h >= 2 * MIN_ROOM
    if depth >= 3 or not (can_x or can_y) or (depth >= 1 and rng.random() < 0.3):
        rooms.append((x0, y0, x1, y1))
        return
    split_x = can_x and (not can_y or rng.random() < w / (w + h))
    if split_x:
        c = x0 + float(np.clip(rng.uniform(0.35, 0.65) * w, MIN_ROOM, w - MIN_ROOM))
        cuts.append(((c, y0), (c, y1)))
        _split(rng, x0, y0, c, y1, depth + 1, rooms, cuts)
        _split(rng, c, y0, x1, y1, depth + 1, rooms, cuts)
    else:
        c = y0 + float(np.clip(rng.uniform(0.35, 0.65) * h, MIN_ROOM, h - MIN_ROOM))
        cuts.append(((x0, c), (x1, c)))
        _split(rng, x0, y0, x1, c, depth + 1, rooms, cuts)
        _split(rng, x0, c, x1, y1, depth + 1, rooms, cuts)


def _build_layout(rng):
    """Return (polygons, W, H) in local 2D pixel coords; polygons are
    [(class_id, 4x2 array)] with rooms/corridor inset so walls never overlap."""
    W, H = int(rng.integers(240, 441)), int(rng.integers(240, 441))
    t = float(rng.integers(4, 9))          # wall thickness
    hw = t / 2.0
    rooms, cuts, corridor = [], [], None

    if rng.random() < 0.75:                # corridor layout
        cw = int(rng.integers(24, 49))
        cy = int(rng.integers(MIN_ROOM + 10, H - cw - MIN_ROOM - 10 + 1))
        corridor = (0, cy, W, cy + cw)
        cuts += [((0, cy), (W, cy)), ((0, cy + cw), (W, cy + cw))]
        _split(rng, 0, 0, W, cy, 0, rooms, cuts)
        _split(rng, 0, cy + cw, W, H, 0, rooms, cuts)
    else:                                  # open layout, no corridor
        _split(rng, 0, 0, W, H, 0, rooms, cuts)

    polys = [(CLASS_RESIDENTIAL, _rect(x0 + hw, y0 + hw, x1 - hw, y1 - hw))
             for x0, y0, x1, y1 in rooms]
    if corridor:
        x0, y0, x1, y1 = corridor
        polys.append((CLASS_CORRIDOR, _rect(x0 + hw, y0 + hw, x1 - hw, y1 - hw)))

    ext = [((0, 0), (W, 0)), ((W, 0), (W, H)), ((W, H), (0, H)), ((0, H), (0, 0))]
    for (ax, ay), (bx, by) in ext + cuts:  # wall = segment thickened by t
        polys.append((CLASS_WALL, _rect(min(ax, bx) - hw, min(ay, by) - hw,
                                        max(ax, bx) + hw, max(ay, by) + hw)))
    return polys, W, H


def _place(rng, polys, W, H):
    """Random 90-degree orientation (+ optional small tilt) and position,
    applied identically to every polygon so image and labels stay aligned."""
    base = float(rng.choice([0, 90, 180, 270]))
    for _ in range(10):
        tilt = float(rng.uniform(-12, 12)) if rng.random() < 0.5 else 0.0
        a = np.deg2rad(base + tilt)
        R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
        pts = [(c, (p - [W / 2, H / 2]) @ R.T) for c, p in polys]
        allp = np.vstack([p for _, p in pts])
        lo, hi = allp.min(axis=0), allp.max(axis=0)
        if (hi - lo).max() <= IMG_SIZE - 16:
            break
    else:                                  # fall back to axis-aligned
        a = np.deg2rad(base)
        R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
        pts = [(c, (p - [W / 2, H / 2]) @ R.T) for c, p in polys]
        allp = np.vstack([p for _, p in pts])
        lo, hi = allp.min(axis=0), allp.max(axis=0)
    span = hi - lo
    off = np.array([rng.uniform(8, IMG_SIZE - 8 - span[i]) for i in range(2)]) - lo
    return [(c, np.clip(np.rint(p + off), 0, IMG_SIZE - 1).astype(np.int32))
            for c, p in pts]


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

    # 2. Generate Synthetic 2D Architectural Floor Plans (varied layouts)
    def generate_sample(rng, img_path, txt_path):
        bg = int(240 + rng.integers(-8, 9))
        img = np.ones((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8) * bg
        colors = {
            CLASS_RESIDENTIAL: np.array([200, 220, 240]),
            CLASS_CORRIDOR: np.array([240, 230, 200]),
            CLASS_WALL: np.array([50, 50, 50]),
        }
        placed = _place(rng, *_build_layout(rng))

        labels = []
        # Wall pixels win: rasterize all wall polygons once, then subtract
        # them from every room/corridor mask so no pixel has two classes.
        wall_mask = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)
        for c, p in placed:                # one call per polygon: a single
            if c == CLASS_WALL:            # multi-polygon fillPoly uses
                cv2.fillPoly(wall_mask, [p.reshape(-1, 1, 2)], 255)  # even-odd

        def jittered(cls):
            color = np.clip(colors[cls] + rng.integers(-10, 11, 3), 0, 255)
            return tuple(int(v) for v in color)

        def label(cls, pts):
            coords = (pts / IMG_SIZE).reshape(-1)
            labels.append(f"{cls} " + " ".join(f"{v:.6f}" for v in coords))

        # Rooms/corridor: fill and label from the FINAL mask (walls removed).
        for cls, pts in placed:
            if cls == CLASS_WALL:
                continue
            m = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)
            cv2.fillPoly(m, [pts.reshape(-1, 1, 2)], 255)
            m[wall_mask > 0] = 0
            img[m > 0] = jittered(cls)
            contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
            for c in contours:
                c = c.reshape(-1, 2)
                if len(c) >= 3 and cv2.contourArea(c) >= 4:
                    label(cls, c.astype(float))

        # Walls drawn last; each wall polygon is its own rasterized shape.
        for cls, pts in placed:
            if cls != CLASS_WALL:
                continue
            cv2.fillPoly(img, [pts.reshape(-1, 1, 2)], jittered(cls))
            if INCLUDE_WALL_CLASS:
                label(cls, pts)

        cv2.imwrite(img_path, img)
        with open(txt_path, "w") as f:
            f.write("\n".join(labels))

    print("📥 Generating training images...")
    rng = np.random.default_rng(0)
    for i in range(N_TRAIN):
        generate_sample(rng, f"data/floor_plans/images/train/sample_{i}.png",
                        f"data/floor_plans/labels/train/sample_{i}.txt")

    print("📥 Generating validation images...")
    rng = np.random.default_rng(1)         # separate stream: no train/val overlap
    for i in range(N_VAL):
        generate_sample(rng, f"data/floor_plans/images/val/sample_{i}.png",
                        f"data/floor_plans/labels/val/sample_{i}.txt")

    # 3. Create YOLOv8 configuration file (data.yaml)
    names = ["residential_room", "common_corridor"]
    if INCLUDE_WALL_CLASS:
        names.append("wall")
    yaml_data = {
        "path": os.path.abspath("data/floor_plans"),
        "train": "images/train",
        "val": "images/val",
        "nc": len(names),
        "names": names
    }

    with open("data/floor_plans/data.yaml", "w") as f:
        yaml.dump(yaml_data, f, default_flow_style=False)

    print("✅ Dataset successfully generated at data/floor_plans!")
    print("✅ data.yaml is ready for YOLOv8 training.")

if __name__ == "__main__":
    create_yolo_dataset()