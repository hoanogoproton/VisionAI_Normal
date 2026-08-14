#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import math
import heapq
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO
from skimage.morphology import skeletonize


# 8-neighborhood with geometric cost
OFFSETS_8 = [
    (-1,  0, 1.0),
    ( 1,  0, 1.0),
    ( 0, -1, 1.0),
    ( 0,  1, 1.0),
    (-1, -1, math.sqrt(2)),
    (-1,  1, math.sqrt(2)),
    ( 1, -1, math.sqrt(2)),
    ( 1,  1, math.sqrt(2)),
]

# BGR colors
PALETTE = [
    (0, 255, 0),
    (255, 0, 0),
    (0, 200, 255),
    (255, 0, 255),
    (0, 128, 255),
    (128, 255, 0),
    (255, 128, 0),
]


def get_aruco_dictionary(dict_name):
    """
    Get OpenCV ArUco predefined dictionary by name.
    Example:
        DICT_4X4_50
        DICT_5X5_100
        DICT_6X6_250
    """
    if not hasattr(cv2, "aruco"):
        raise RuntimeError(
            "cv2.aruco not found. Please install opencv-contrib-python:\n"
            "pip install opencv-contrib-python"
        )

    if not hasattr(cv2.aruco, dict_name):
        valid = [x for x in dir(cv2.aruco) if x.startswith("DICT_")]
        raise ValueError(
            f"Unknown ArUco dictionary: {dict_name}. "
            f"Available dictionaries include: {valid[:30]}"
        )

    return cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dict_name))


def detect_aruco_markers(image, dict_name="DICT_4X4_50"):
    """
    Detect ArUco markers.
    Compatible with newer and older OpenCV ArUco APIs.
    """
    aruco_dict = get_aruco_dictionary(dict_name)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    aruco = cv2.aruco

    # OpenCV >= 4.7
    if hasattr(aruco, "ArucoDetector"):
        params = aruco.DetectorParameters()
        detector = aruco.ArucoDetector(aruco_dict, params)
        corners, ids, rejected = detector.detectMarkers(gray)

    # Older OpenCV
    else:
        params = aruco.DetectorParameters_create()
        corners, ids, rejected = aruco.detectMarkers(
            gray,
            aruco_dict,
            parameters=params,
        )

    return corners, ids, rejected


def detect_aruco_scale(
    image,
    marker_size_mm,
    dict_name="DICT_4X4_50",
    marker_id=None,
):
    """
    Detect ArUco marker and calculate pixel/mm scale.

    Args:
        image: BGR image
        marker_size_mm: real marker side length in millimeters
        dict_name: ArUco dictionary name
        marker_id: if not None, only use this marker id

    Returns:
        dict with px_per_mm and mm_per_px, or None if marker not found.
    """
    if marker_size_mm is None or marker_size_mm <= 0:
        return None

    corners, ids, _ = detect_aruco_markers(image, dict_name=dict_name)

    if ids is None or len(corners) == 0:
        return None

    ids_flat = ids.reshape(-1)

    px_per_mm_values = []
    side_px_values = []
    used_corners = []
    used_ids = []

    for c, mid in zip(corners, ids_flat):
        mid = int(mid)

        if marker_id is not None and mid != int(marker_id):
            continue

        pts = c.reshape(4, 2).astype(np.float32)

        # Four marker side lengths in pixels
        side_lengths_px = [
            np.linalg.norm(pts[1] - pts[0]),
            np.linalg.norm(pts[2] - pts[1]),
            np.linalg.norm(pts[3] - pts[2]),
            np.linalg.norm(pts[0] - pts[3]),
        ]

        side_px = float(np.mean(side_lengths_px))

        if side_px <= 0:
            continue

        px_per_mm = side_px / float(marker_size_mm)

        px_per_mm_values.append(px_per_mm)
        side_px_values.append(side_px)
        used_corners.append(c.astype(np.float32))
        used_ids.append(mid)

    if len(px_per_mm_values) == 0:
        return None

    # Use median for robustness if multiple markers are detected
    px_per_mm = float(np.median(px_per_mm_values))
    mm_per_px = 1.0 / px_per_mm
    side_px_median = float(np.median(side_px_values))

    return {
        "px_per_mm": px_per_mm,
        "mm_per_px": mm_per_px,
        "marker_size_mm": float(marker_size_mm),
        "dict_name": dict_name,
        "marker_id": None if marker_id is None else int(marker_id),
        "detected_ids": used_ids,
        "side_px_median": side_px_median,
        "corners": used_corners,
    }


def draw_aruco_scale(image, aruco_info):
    """
    Draw detected ArUco marker and scale text on image.
    """
    annotated = image.copy()

    if aruco_info is None:
        return annotated

    corners = aruco_info.get("corners", [])
    ids = aruco_info.get("detected_ids", [])

    if len(corners) > 0 and len(ids) > 0 and hasattr(cv2, "aruco"):
        ids_np = np.array(ids, dtype=np.int32).reshape(-1, 1)
        cv2.aruco.drawDetectedMarkers(
            annotated,
            corners,
            ids_np,
            (0, 255, 255),
        )

    text = (
        f"ArUco scale: {aruco_info['px_per_mm']:.4f} px/mm | "
        f"{aruco_info['mm_per_px']:.6f} mm/px"
    )

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.4
    thickness = 2

    margin = 8
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)

    x1, y1 = 5, 5
    x2 = min(annotated.shape[1] - 1, x1 + tw + margin * 2)
    y2 = min(annotated.shape[0] - 1, y1 + th + baseline + margin * 2)

    cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 0), -1)
    cv2.putText(
        annotated,
        text,
        (x1 + margin, y1 + margin + th),
        font,
        font_scale,
        (255, 255, 255),
        thickness,
        lineType=cv2.LINE_AA,
    )

    return annotated


def get_class_name(names, cls_id):
    if isinstance(names, dict):
        return str(names.get(cls_id, cls_id))
    if isinstance(names, (list, tuple)):
        if 0 <= cls_id < len(names):
            return str(names[cls_id])
    return str(cls_id)


def count_result_items(result):
    num_boxes = 0 if result.boxes is None else len(result.boxes)
    num_masks = 0 if (result.masks is None or getattr(result.masks, "data", None) is None) else len(result.masks.data)
    return num_boxes, num_masks


def fill_holes(binary_mask):
    """
    Fill holes inside a binary mask.
    Input: binary mask 0/255
    Output: binary mask 0/255
    """
    h, w = binary_mask.shape[:2]
    flood = binary_mask.copy()
    flood_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)

    # Flood-fill background from corner
    cv2.floodFill(flood, flood_mask, seedPoint=(0, 0), newVal=255)

    flood_inv = cv2.bitwise_not(flood)
    filled = cv2.bitwise_or(binary_mask, flood_inv)
    return filled


def largest_component(binary_mask):
    """
    Keep only the largest connected component.
    Input: binary mask 0/255
    Output: binary mask 0/255
    """
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
    if num_labels <= 1:
        return binary_mask

    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    out = np.zeros_like(binary_mask)
    out[labels == largest_label] = 255
    return out


def clean_binary_mask(mask, close_kernel=5, open_kernel=3):
    """
    Clean raw YOLO mask:
    - binarize
    - close
    - open
    - fill holes
    - keep largest component
    """
    m = (mask > 0).astype(np.uint8) * 255

    if close_kernel > 0:
        k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_kernel, close_kernel))
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k_close, iterations=1)

    if open_kernel > 0:
        k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_kernel, open_kernel))
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k_open, iterations=1)

    m = fill_holes(m)
    m = largest_component(m)
    return m


def skeleton_endpoints(skel_bool):
    """
    endpoint = skeleton pixel with exactly 1 neighbor in 8-neighborhood
    """
    s = skel_bool.astype(np.uint8)
    neighbor_count = cv2.filter2D(
        s,
        ddepth=cv2.CV_16S,
        kernel=np.ones((3, 3), dtype=np.uint8),
        borderType=cv2.BORDER_CONSTANT,
    ) - s

    end_mask = (skel_bool > 0) & (neighbor_count == 1)
    ys, xs = np.where(end_mask)
    return list(zip(ys.tolist(), xs.tolist()))


def dijkstra_on_skeleton(skel_bool, start_yx):
    """
    Shortest-path distance map constrained on skeleton pixels.
    """
    h, w = skel_bool.shape
    dist = np.full((h, w), np.inf, dtype=np.float32)

    sy, sx = start_yx
    dist[sy, sx] = 0.0
    pq = [(0.0, sy, sx)]

    while pq:
        d, y, x = heapq.heappop(pq)
        if d > dist[y, x]:
            continue

        for dy, dx, cost in OFFSETS_8:
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w and skel_bool[ny, nx]:
                nd = d + cost
                if nd < dist[ny, nx]:
                    dist[ny, nx] = nd
                    heapq.heappush(pq, (nd, ny, nx))

    return dist


def trim_skeleton_ends(skel_bool, trim_fraction=0.10):
    """
    Trim a fraction from both skeleton ends to avoid unstable width near endpoints.
    Example: trim_fraction=0.10 means cut 10% from each end.

    If endpoints cannot be found reliably, return original skeleton.
    """
    if trim_fraction <= 0:
        return skel_bool

    endpoints = skeleton_endpoints(skel_bool)
    if len(endpoints) < 2:
        return skel_bool

    # Choose the farthest endpoint pair by Euclidean distance
    best_pair = None
    best_d2 = -1.0
    for i in range(len(endpoints)):
        y1, x1 = endpoints[i]
        for j in range(i + 1, len(endpoints)):
            y2, x2 = endpoints[j]
            d2 = (y2 - y1) ** 2 + (x2 - x1) ** 2
            if d2 > best_d2:
                best_d2 = d2
                best_pair = (endpoints[i], endpoints[j])

    if best_pair is None:
        return skel_bool

    p1, p2 = best_pair
    d1 = dijkstra_on_skeleton(skel_bool, p1)
    d2 = dijkstra_on_skeleton(skel_bool, p2)

    total_len = float(d1[p2[0], p2[1]])
    if not np.isfinite(total_len) or total_len <= 0:
        return skel_bool

    keep = skel_bool & (d1 > trim_fraction * total_len) & (d2 > trim_fraction * total_len)

    if np.count_nonzero(keep) < 5:
        return skel_bool

    return keep


def bbox_from_mask(binary_mask):
    ys, xs = np.where(binary_mask > 0)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def mask_from_result(result, idx, image_shape_hw):
    """
    Extract one mask from Ultralytics result and resize to original image size if needed.
    Output: binary mask 0/255 in original image coordinates
    """
    mask = result.masks.data[idx].detach().cpu().numpy()
    mask = (mask > 0.5).astype(np.uint8) * 255

    h, w = image_shape_hw
    if mask.shape != (h, w):
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

    return mask


def measure_width_from_mask(mask, trim_fraction=0.10, p_low=10, p_high=90):
    """
    raw mask -> clean -> skeleton -> trim ends -> distance transform -> widths = 2 * dist(skeleton)
    """
    clean = clean_binary_mask(mask)

    if np.count_nonzero(clean) == 0:
        return None

    skel = skeletonize(clean > 0)
    if np.count_nonzero(skel) == 0:
        return None

    skel = trim_skeleton_ends(skel, trim_fraction=trim_fraction)
    if np.count_nonzero(skel) == 0:
        return None

    dist = cv2.distanceTransform(clean, cv2.DIST_L2, 5)
    widths = 2.0 * dist[skel]

    widths = widths[np.isfinite(widths) & (widths > 0)]
    if widths.size == 0:
        return None

    if widths.size >= 10:
        lo, hi = np.percentile(widths, [p_low, p_high])
        widths_filtered = widths[(widths >= lo) & (widths <= hi)]
        if widths_filtered.size >= 5:
            widths = widths_filtered

    return {
        "clean_mask": clean,
        "skeleton": skel,
        "distance_map": dist,
        "median_px": float(np.median(widths)),
        "mean_px": float(np.mean(widths)),
        "std_px": float(np.std(widths)),
        "min_px": float(np.min(widths)),
        "max_px": float(np.max(widths)),
        "num_points": int(widths.size),
    }


def draw_measurement(image, measurement, label, color=(0, 255, 0), alpha=0.35, max_circles=20):
    """
    Draw:
    - filled clean mask
    - contour
    - skeleton
    - sampled DT circles
    - label text
    """
    annotated = image.copy()
    clean = measurement["clean_mask"]
    skel = measurement["skeleton"]
    dist = measurement["distance_map"]

    # Transparent mask overlay
    mask_idx = clean > 0
    color_arr = np.array(color, dtype=np.float32)
    annotated[mask_idx] = (
        annotated[mask_idx].astype(np.float32) * (1.0 - alpha) + color_arr * alpha
    ).astype(np.uint8)

    # Contour
    contours, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(annotated, contours, -1, color, 2, lineType=cv2.LINE_AA)

    # Skeleton in red
    annotated[skel] = (0, 0, 255)

    # Sample circles from distance transform
    ys, xs = np.where(skel)
    if len(xs) > 0:
        sample_n = min(max_circles, len(xs))
        sample_ids = np.linspace(0, len(xs) - 1, num=sample_n, dtype=int)
        for sid in sample_ids:
            y, x = int(ys[sid]), int(xs[sid])
            r = int(round(dist[y, x]))
            if r >= 1:
                cv2.circle(annotated, (x, y), r, (255, 255, 0), 1, lineType=cv2.LINE_AA)

    bbox = bbox_from_mask(clean)
    if bbox is not None:
        x1, y1, x2, y2 = bbox

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.4
        thickness = 2

        (tw, th), baseline = cv2.getTextSize(label, font, font_scale, thickness)

        tx = max(0, x1)
        ty = y1 - 8
        if ty - th - baseline < 0:
            ty = min(image.shape[0] - 5, y2 + th + 8)

        x_left = tx
        y_top = max(0, ty - th - baseline - 4)
        x_right = min(image.shape[1] - 1, tx + tw + 6)
        y_bottom = min(image.shape[0] - 1, ty + baseline + 2)

        cv2.rectangle(annotated, (x_left, y_top), (x_right, y_bottom), (0, 0, 0), -1)
        cv2.putText(
            annotated,
            label,
            (tx + 3, ty),
            font,
            font_scale,
            (255, 255, 255),
            thickness,
            lineType=cv2.LINE_AA,
        )

    return annotated


def is_target_class(cls_id, cls_name, target_class_id=None, target_class_name=None):
    if target_class_id is not None and cls_id != target_class_id:
        return False
    if target_class_name is not None and cls_name != target_class_name:
        return False
    return True


def box_iou_xyxy(box1, box2):
    """
    box format: [x1, y1, x2, y2]
    """
    x1 = max(float(box1[0]), float(box2[0]))
    y1 = max(float(box1[1]), float(box2[1]))
    x2 = min(float(box1[2]), float(box2[2]))
    y2 = min(float(box1[3]), float(box2[3]))

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter = inter_w * inter_h

    area1 = max(0.0, float(box1[2]) - float(box1[0])) * max(0.0, float(box1[3]) - float(box1[1]))
    area2 = max(0.0, float(box2[2]) - float(box2[0])) * max(0.0, float(box2[3]) - float(box2[1]))
    union = area1 + area2 - inter

    if union <= 0:
        return 0.0
    return inter / union


def mask_iou_and_overlap(mask1, mask2):
    """
    Return:
    - mask_iou = intersection / union
    - overlap_min = intersection / min(area1, area2)
    """
    a = mask1 > 0
    b = mask2 > 0

    area_a = int(a.sum())
    area_b = int(b.sum())
    if area_a == 0 or area_b == 0:
        return 0.0, 0.0

    inter = int(np.logical_and(a, b).sum())
    if inter == 0:
        return 0.0, 0.0

    union = area_a + area_b - inter
    mask_iou = inter / max(1, union)
    overlap_min = inter / max(1, min(area_a, area_b))
    return float(mask_iou), float(overlap_min)


def build_unique_instances(
    result,
    image_shape_hw,
    names,
    target_class_id=None,
    target_class_name=None,
    dedup_box_iou=0.70,
    dedup_mask_iou=0.50,
    dedup_overlap=0.85,
    top1=False,
):
    """
    Build instance list and remove near-duplicate detections.
    Keep highest-confidence instance first.
    """
    num_boxes, num_masks = count_result_items(result)
    if num_boxes == 0 or num_masks == 0:
        return []

    n = min(num_boxes, num_masks)
    candidates = []

    for i in range(n):
        cls_id = int(result.boxes.cls[i].item())
        score = float(result.boxes.conf[i].item())
        cls_name = get_class_name(names, cls_id)

        if not is_target_class(
            cls_id=cls_id,
            cls_name=cls_name,
            target_class_id=target_class_id,
            target_class_name=target_class_name,
        ):
            continue

        raw_mask = mask_from_result(result, i, image_shape_hw)
        box_xyxy = result.boxes.xyxy[i].detach().cpu().numpy().tolist()

        candidates.append({
            "orig_index": i,
            "class_id": cls_id,
            "class_name": cls_name,
            "confidence": score,
            "box_xyxy": box_xyxy,
            "raw_mask": raw_mask,
        })

    candidates.sort(key=lambda x: x["confidence"], reverse=True)

    kept = []
    for cand in candidates:
        is_duplicate = False

        for prev in kept:
            if prev["class_id"] != cand["class_id"]:
                continue

            biou = box_iou_xyxy(cand["box_xyxy"], prev["box_xyxy"])
            miou, overlap_min = mask_iou_and_overlap(cand["raw_mask"], prev["raw_mask"])

            if (
                biou >= dedup_box_iou
                or miou >= dedup_mask_iou
                or overlap_min >= dedup_overlap
            ):
                is_duplicate = True
                break

        if not is_duplicate:
            kept.append(cand)

    if top1 and len(kept) > 0:
        return [kept[0]]

    return kept


def process_image(
    model_path,
    image_path,
    output_path,
    target_class_id=None,
    target_class_name=None,
    conf=0.25,
    imgsz=448,
    iou=0.30,
    trim_fraction=0.10,
    p_low=10,
    p_high=90,
    dedup_box_iou=0.70,
    dedup_mask_iou=0.50,
    dedup_overlap=0.85,
    top1=False,
    aruco_marker_size_mm=None,
    aruco_dict="DICT_4X4_50",
    aruco_id=None,
    require_aruco=False,
):
    image_path = str(image_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    h, w = image.shape[:2]

    # Detect ArUco scale
    aruco_info = None

    if require_aruco and aruco_marker_size_mm is None:
        raise ValueError("--require-aruco requires --aruco-marker-size-mm")

    if aruco_marker_size_mm is not None:
        aruco_info = detect_aruco_scale(
            image=image,
            marker_size_mm=aruco_marker_size_mm,
            dict_name=aruco_dict,
            marker_id=aruco_id,
        )

        if aruco_info is None:
            msg = (
                "[WARN] ArUco marker not detected. "
                "Width conversion to mm will be disabled."
            )
            if require_aruco:
                raise RuntimeError(msg)
            print(msg)
        else:
            print(
                f"[INFO] ArUco scale: "
                f"{aruco_info['px_per_mm']:.4f} px/mm, "
                f"{aruco_info['mm_per_px']:.6f} mm/px, "
                f"side_px={aruco_info['side_px_median']:.2f}, "
                f"ids={aruco_info['detected_ids']}"
            )

    model = YOLO(model_path)

    results = model.predict(
        source=image_path,
        conf=conf,
        iou=iou,
        imgsz=imgsz,
        retina_masks=True,
        verbose=False,
    )

    if len(results) == 0:
        raise RuntimeError("YOLO returned no result.")

    result = results[0]
    names = result.names if hasattr(result, "names") else model.names
    num_boxes, num_masks = count_result_items(result)

    if num_boxes == 0:
        print("[INFO] No detections found.")
        out_img = draw_aruco_scale(image, aruco_info) if aruco_info is not None else image
        cv2.imwrite(str(output_path), out_img)
        with open(output_path.with_suffix(".json"), "w", encoding="utf-8") as f:
            json.dump([], f, ensure_ascii=False, indent=2)
        return []

    if num_masks == 0:
        print("[INFO] Detections found but no segmentation masks found.")
        out_img = draw_aruco_scale(image, aruco_info) if aruco_info is not None else image
        cv2.imwrite(str(output_path), out_img)
        with open(output_path.with_suffix(".json"), "w", encoding="utf-8") as f:
            json.dump([], f, ensure_ascii=False, indent=2)
        return []

    annotated = image.copy()
    records = []

    instances = build_unique_instances(
        result=result,
        image_shape_hw=(h, w),
        names=names,
        target_class_id=target_class_id,
        target_class_name=target_class_name,
        dedup_box_iou=dedup_box_iou,
        dedup_mask_iou=dedup_mask_iou,
        dedup_overlap=dedup_overlap,
        top1=top1,
    )

    print(f"[DEBUG] raw_instances={num_masks}, kept_after_dedup={len(instances)}")

    for k, inst in enumerate(instances):
        cls_id = inst["class_id"]
        score = inst["confidence"]
        cls_name = inst["class_name"]
        raw_mask = inst["raw_mask"]
        src_idx = inst["orig_index"]

        measurement = measure_width_from_mask(
            raw_mask,
            trim_fraction=trim_fraction,
            p_low=p_low,
            p_high=p_high,
        )

        if measurement is None:
            print(f"[WARN] Instance {src_idx}: cannot measure width.")
            continue

        color = PALETTE[k % len(PALETTE)]

        if aruco_info is not None:
            median_mm = measurement["median_px"] * aruco_info["mm_per_px"]
            label = (
                f"{cls_name} #{src_idx} | "
                f"median={measurement['median_px']:.2f}px "
                f"({median_mm:.3f}mm)"
            )
        else:
            label = f"{cls_name} #{src_idx} | median={measurement['median_px']:.2f}px"

        annotated = draw_measurement(
            annotated,
            measurement,
            label=label,
            color=color,
            alpha=0.35,
            max_circles=20,
        )

        bbox = bbox_from_mask(measurement["clean_mask"])
        x1, y1, x2, y2 = bbox if bbox is not None else (None, None, None, None)

        record = {
            "instance_id": src_idx,
            "class_id": cls_id,
            "class_name": cls_name,
            "confidence": score,
            "bbox_xyxy": [x1, y1, x2, y2],

            "median_width_px": measurement["median_px"],
            "mean_width_px": measurement["mean_px"],
            "std_width_px": measurement["std_px"],
            "min_width_px": measurement["min_px"],
            "max_width_px": measurement["max_px"],

            "median_width_mm": None,
            "mean_width_mm": None,
            "std_width_mm": None,
            "min_width_mm": None,
            "max_width_mm": None,

            "scale_px_per_mm": None,
            "scale_mm_per_px": None,
            "aruco_dict": None,
            "aruco_marker_size_mm": None,
            "aruco_marker_id_filter": None,
            "aruco_marker_ids": None,
            "aruco_marker_side_px_median": None,

            "num_skeleton_points_used": measurement["num_points"],
        }

        if aruco_info is not None:
            mpp = float(aruco_info["mm_per_px"])

            record.update({
                "median_width_mm": measurement["median_px"] * mpp,
                "mean_width_mm": measurement["mean_px"] * mpp,
                "std_width_mm": measurement["std_px"] * mpp,
                "min_width_mm": measurement["min_px"] * mpp,
                "max_width_mm": measurement["max_px"] * mpp,

                "scale_px_per_mm": float(aruco_info["px_per_mm"]),
                "scale_mm_per_px": float(aruco_info["mm_per_px"]),
                "aruco_dict": aruco_info["dict_name"],
                "aruco_marker_size_mm": aruco_info["marker_size_mm"],
                "aruco_marker_id_filter": aruco_info["marker_id"],
                "aruco_marker_ids": aruco_info["detected_ids"],
                "aruco_marker_side_px_median": aruco_info["side_px_median"],
            })

        records.append(record)

    if aruco_info is not None:
        annotated = draw_aruco_scale(annotated, aruco_info)

    cv2.imwrite(str(output_path), annotated)

    json_path = output_path.with_suffix(".json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    return records


def parse_args():
    parser = argparse.ArgumentParser(
        description="Measure wire thickness in pixels/mm from YOLO segmentation mask using skeleton + distance transform + ArUco scale."
    )

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to YOLO segmentation model, e.g. best.pt",
    )
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to input image",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output/annotated.png",
        help="Path to output annotated image",
    )

    parser.add_argument(
        "--class-id",
        type=int,
        default=None,
        help="Only process this class id",
    )
    parser.add_argument(
        "--class-name",
        type=str,
        default=None,
        help="Only process this class name, e.g. DoCot",
    )

    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="YOLO confidence threshold",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.30,
        help="YOLO NMS IoU threshold",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=448,
        help="YOLO inference image size",
    )

    parser.add_argument(
        "--trim-frac",
        type=float,
        default=0.10,
        help="Trim fraction at both skeleton ends",
    )
    parser.add_argument(
        "--p-low",
        type=float,
        default=10,
        help="Low percentile for outlier filtering",
    )
    parser.add_argument(
        "--p-high",
        type=float,
        default=90,
        help="High percentile for outlier filtering",
    )

    parser.add_argument(
        "--dedup-box-iou",
        type=float,
        default=0.70,
        help="Extra duplicate suppression by box IoU",
    )
    parser.add_argument(
        "--dedup-mask-iou",
        type=float,
        default=0.50,
        help="Extra duplicate suppression by mask IoU",
    )
    parser.add_argument(
        "--dedup-overlap",
        type=float,
        default=0.85,
        help="Extra duplicate suppression by mask overlap",
    )
    parser.add_argument(
        "--top1",
        action="store_true",
        help="Keep only highest-confidence instance after filtering",
    )

    # ArUco arguments
    parser.add_argument(
        "--aruco-marker-size-mm",
        type=float,
        default=None,
        help="Real ArUco marker side length in millimeters. If omitted, mm conversion is disabled.",
    )
    parser.add_argument(
        "--aruco-dict",
        type=str,
        default="DICT_4X4_50",
        help="OpenCV ArUco dictionary name, e.g. DICT_4X4_50, DICT_5X5_100, DICT_6X6_250.",
    )
    parser.add_argument(
        "--aruco-id",
        type=int,
        default=None,
        help="Use only this ArUco marker id for scale. If omitted, use all detected markers.",
    )
    parser.add_argument(
        "--require-aruco",
        action="store_true",
        help="Raise error if ArUco marker is not detected.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    records = process_image(
        model_path=args.model,
        image_path=args.image,
        output_path=args.output,
        target_class_id=args.class_id,
        target_class_name=args.class_name,
        conf=args.conf,
        imgsz=args.imgsz,
        iou=args.iou,
        trim_fraction=args.trim_frac,
        p_low=args.p_low,
        p_high=args.p_high,
        dedup_box_iou=args.dedup_box_iou,
        dedup_mask_iou=args.dedup_mask_iou,
        dedup_overlap=args.dedup_overlap,
        top1=args.top1,
        aruco_marker_size_mm=args.aruco_marker_size_mm,
        aruco_dict=args.aruco_dict,
        aruco_id=args.aruco_id,
        require_aruco=args.require_aruco,
    )

    print("\n=== Measurement result ===")
    if len(records) == 0:
        print("No valid instance measured.")
    else:
        for r in records:
            if r.get("median_width_mm") is not None:
                print(
                    f"[instance {r['instance_id']}] "
                    f"class={r['class_name']} "
                    f"conf={r['confidence']:.3f} "
                    f"median={r['median_width_px']:.2f}px/{r['median_width_mm']:.3f}mm "
                    f"mean={r['mean_width_px']:.2f}px/{r['mean_width_mm']:.3f}mm "
                    f"std={r['std_width_px']:.2f}px/{r['std_width_mm']:.3f}mm"
                )
            else:
                print(
                    f"[instance {r['instance_id']}] "
                    f"class={r['class_name']} "
                    f"conf={r['confidence']:.3f} "
                    f"median={r['median_width_px']:.2f}px "
                    f"mean={r['mean_width_px']:.2f}px "
                    f"std={r['std_width_px']:.2f}px"
                )

    print("\nSaved annotated image and JSON successfully.")


if __name__ == "__main__":
    main()