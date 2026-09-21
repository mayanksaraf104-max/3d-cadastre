import os
import shapely.geometry as sg
from shapely.validation import make_valid
from ultralytics import YOLO

import config

def extract_ogc_boundaries(
    image_path,
    weights_path=config.YOLO_INFERENCE_MODEL_PATH,
    tolerance=1.5,
    max_aspect_ratio=10.0  # Rejects degenerate slivers (see NOTE below)
):
    """
    Loads the trained YOLOv8 model, runs inference on a floor plan,
    and extracts simplified OGC Polygons.

    Z-AXIS FIX: this function used to hardcode z_base=0.0 / z_top=3.0 as
    default parameters and stamp every single detected unit with those
    exact same two numbers. That's fabricated elevation, not measured
    data -- and it's why every floor came out perfectly flat regardless
    of what LiDAR was indexed: this function was never given real Z
    information to begin with, so it just made two numbers up.

    This function only ever sees the floor-plan IMAGE, so its polygons
    are in PIXEL space -- they have no real-world elevation, and
    (per get_intersecting_lidar_tiles' docstring in z_engine.py) can't be
    used to query LiDAR directly; they first need the GNSS affine
    transform into real-world / CADASTRE_SRID coordinates, which happens
    downstream in main.py. So rather than fabricate z_base/z_top here,
    each unit now carries `floor_points_xyz: None` and `z_top: None` as
    explicit placeholders, to be filled in downstream.

    Downstream, main.py transforms the polygon to global coordinates
    (GNSS affine) and obtains authoritative measured XYZ surface evidence
    from the current z_engine pipeline. The YOLO polygon is only 2D
    reference/semantic evidence; it never supplies elevation, and nothing
    here extrudes it or generates geometry.

    `floor_points_xyz` is an (N, 3) array of real floor-surface points
    (see z_engine.get_floor_points_xyz), not a scalar -- it preserves
    slopes, dips, and split-levels instead of flattening the floor to one
    number. Feed it directly into SectionProfile; don't reduce it to a
    scalar (e.g. its own min/mean) before doing so, or you reintroduce
    the exact flat-floor problem this fix removes. If a future blueprint
    annotation format supplies floor elevation hints directly from the
    2D drawing (e.g. contour/spot-height labels), those should also be
    threaded through as points here rather than collapsed to a constant.

    NOTE: `clean_poly` can legitimately be a Polygon WITH interior rings
    (holes) -- e.g. a courtyard, atrium, or light well inside a footprint.
    make_valid() and the MultiPolygon-selection branch below both preserve
    `clean_poly.interiors` correctly; nothing here drops them. main.py
    walks `poly.interiors` too, so those holes come through as real voids
    in the extruded solid instead of being lost.

    NOTE on `max_aspect_ratio`: the YOLO mask -> simplify -> make_valid
    chain occasionally emits a long, near-zero-width sliver instead of a
    real room/corridor -- usually two mask edges that almost touch along
    most of their length, so simplify() collapses the shape into a thin
    rectangle that can stretch across the entire floor plan (visually it
    looks like a stray line -- a "metro line" -- cutting through the 3D
    model). A real corridor is elongated too, but its width stays roughly
    proportional to its length; a mask-noise sliver does not. We measure
    elongation via the polygon's minimum rotated bounding rectangle and
    drop anything thinner than `max_aspect_ratio` allows, regardless of
    what class_name says, since a corridor label on a degenerate sliver
    is itself part of the bug. Tune `max_aspect_ratio` down if legitimate
    long/narrow corridors start getting rejected, or up if slivers still
    slip through.
    """
    if not os.path.exists(weights_path):
        print(f"Error: Trained weights not found at {weights_path}")
        return []

    print(f"Loading trained AI model from: {weights_path}")
    model = YOLO(weights_path)

    print(f"Running inference on blueprint: {image_path}")
    results = model.predict(source=image_path, conf=0.4, verbose=False)

    ogc_units = []

    for r in results:
        if r.masks is None:
            print("No boundaries detected in the image.")
            continue

        # Extract polygon coordinates from the predicted masks
        for seg_idx, mask_coords in enumerate(r.masks.xy):
            class_id = int(r.boxes.cls[seg_idx])
            class_name = model.names[class_id]

            if len(mask_coords) >= 3:
                # 1. Create raw polygon
                raw_poly = sg.Polygon(mask_coords)

                # 2. OGC Optimization & Safety Validation
                simplified_poly = raw_poly.simplify(tolerance=tolerance, preserve_topology=True)
                clean_poly = make_valid(simplified_poly)  # Untangles self-intersecting AI errors

                # Ignore empty geometries or tiny noise artifacts
                if clean_poly.is_empty or clean_poly.area < 5.0:
                    continue

                # Handle MultiPolygons if make_valid split a bow-tie into two shapes.
                # The winning sub-polygon's own interiors (holes) are preserved.
                if clean_poly.geom_type == 'MultiPolygon':
                    clean_poly = max(clean_poly.geoms, key=lambda a: a.area)

                # 2b. Reject degenerate slivers (the "metro line" bug -- see
                # docstring NOTE above). min_rect is the smallest rectangle
                # that fully contains the polygon; its two edge lengths give
                # us length/width without caring about absolute pixel scale.
                min_rect = clean_poly.minimum_rotated_rectangle
                rect_coords = list(min_rect.exterior.coords)
                edge_lengths = [
                    sg.Point(rect_coords[i]).distance(sg.Point(rect_coords[i + 1]))
                    for i in range(4)
                ]
                long_side, short_side = max(edge_lengths), min(edge_lengths)
                aspect_ratio = long_side / short_side if short_side > 1e-6 else float("inf")

                if aspect_ratio > max_aspect_ratio:
                    print(
                        f"⚠️ Rejected sliver at seg {seg_idx+1}: aspect ratio "
                        f"{aspect_ratio:.1f}:1 exceeds max_aspect_ratio={max_aspect_ratio} "
                        f"(class_name='{class_name}') -- likely a mask-noise artifact, not a real unit."
                    )
                    continue

                # 3. Classify into legal cadastral unit types
                cadastre_type = "COMM_CORRIDOR" if "corridor" in class_name.lower() else "RES_ROOM"

                # 4. Formatted for overlap_engine.py / main.py.
                # floor_points_xyz / z_top are intentionally None here --
                # this function only has pixel-space geometry, no real
                # elevation. main.py obtains authoritative measured XYZ
                # surface evidence from the current z_engine pipeline.
                # See the module docstring above for why a fabricated
                # uniform value here was the actual bug.
                ogc_units.append({
                    "id": f"AI_Unit_{seg_idx+1}",       # Matches overlap_engine
                    "type": cadastre_type,
                    "polygon": clean_poly,              # The cleaned Shapely object (holes preserved)
                    "floor_points_xyz": None,           # (N,3) real floor evidence -- filled downstream
                    "z_top": None,                      # measured roof Z -- filled downstream
                    "raw_point_count": len(mask_coords),
                    "ogc_point_count": len(clean_poly.exterior.coords),
                    "hole_count": len(clean_poly.interiors),
                    "wkt_ogc": clean_poly.wkt,
                })

    return ogc_units


if __name__ == "__main__":
    # Ensure this path matches an actual image in your system
    test_image = "data/floor_plans/images/val/sample_0.png"

    if not os.path.exists(test_image):
        print(f"⚠️ Test image not found at {test_image}. Cannot run test.")
    else:
        units = extract_ogc_boundaries(test_image)

        print(f"\n✅ Extracted {len(units)} OGC-compliant property units from the neural net:\n")

        for u in units:
            print(f"[{u['id']}] Type: {u['type']}")
            print(f"  Points: Reduced from {u['raw_point_count']} pixel vertices -> {u['ogc_point_count']} OGC vertices")
            print(f"  Holes: {u['hole_count']}")
            print(f"  Floor evidence / Z-top: pending downstream LiDAR fusion "
                  f"(floor_points_xyz={u['floor_points_xyz']}, z_top={u['z_top']})")
            print(f"  OGC 2D WKT: {u['wkt_ogc'][:60]}...\n")