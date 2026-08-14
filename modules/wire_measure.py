import math
import heapq

import cv2
import numpy as np
from skimage.morphology import skeletonize


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

PALETTE = [
    (0, 255, 0),
    (255, 0, 0),
    (0, 200, 255),
    (255, 0, 255),
    (0, 128, 255),
    (128, 255, 0),
    (255, 128, 0),
]


def fill_holes(binary_mask):
    h, w = binary_mask.shape[:2]
    flood = binary_mask.copy()
    flood_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
    cv2.floodFill(flood, flood_mask, seedPoint=(0, 0), newVal=255)
    flood_inv = cv2.bitwise_not(flood)
    filled = cv2.bitwise_or(binary_mask, flood_inv)
    return filled


def largest_component(binary_mask):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
    if num_labels <= 1:
        return binary_mask
    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    out = np.zeros_like(binary_mask)
    out[labels == largest_label] = 255
    return out


def clean_binary_mask(mask, close_kernel=5, open_kernel=3):
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
    if trim_fraction <= 0:
        return skel_bool

    endpoints = skeleton_endpoints(skel_bool)
    if len(endpoints) < 2:
        return skel_bool

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


def measure_width_from_mask(mask, trim_fraction=0.10, p_low=10, p_high=90):
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


def bbox_from_mask(binary_mask):
    ys, xs = np.where(binary_mask > 0)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def mask_from_result(result, idx, image_shape_hw):
    mask = result.masks.data[idx].detach().cpu().numpy()
    mask = (mask > 0.5).astype(np.uint8) * 255

    h, w = image_shape_hw
    if mask.shape != (h, w):
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

    return mask


def get_class_name(names, cls_id):
    if isinstance(names, dict):
        return str(names.get(cls_id, cls_id))
    if isinstance(names, (list, tuple)):
        if 0 <= cls_id < len(names):
            return str(names[cls_id])
    return str(cls_id)


def is_target_class(cls_id, cls_name, target_class_id=None, target_class_name=None):
    if target_class_id is not None and cls_id != target_class_id:
        return False
    if target_class_name is not None and cls_name != target_class_name:
        return False
    return True


def detect_aruco_markers(image, dict_name="DICT_4X4_50"):
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

    aruco_dict = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dict_name))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    aruco = cv2.aruco
    if hasattr(aruco, "ArucoDetector"):
        params = aruco.DetectorParameters()
        detector = aruco.ArucoDetector(aruco_dict, params)
        corners, ids, rejected = detector.detectMarkers(gray)
    else:
        params = aruco.DetectorParameters_create()
        corners, ids, rejected = aruco.detectMarkers(
            gray, aruco_dict, parameters=params,
        )

    return corners, ids, rejected


def detect_aruco_scale(image, marker_size_mm, dict_name="DICT_4X4_50", marker_id=None):
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
    annotated = image.copy()

    if aruco_info is None:
        return annotated

    corners = aruco_info.get("corners", [])
    ids = aruco_info.get("detected_ids", [])

    if len(corners) > 0 and len(ids) > 0 and hasattr(cv2, "aruco"):
        ids_np = np.array(ids, dtype=np.int32).reshape(-1, 1)
        cv2.aruco.drawDetectedMarkers(annotated, corners, ids_np, (0, 255, 255))

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
        annotated, text, (x1 + margin, y1 + margin + th),
        font, font_scale, (255, 255, 255), thickness, lineType=cv2.LINE_AA,
    )

    return annotated


def draw_measurement(image, measurement, label, color=(0, 255, 0), alpha=0.35, max_circles=20):
    annotated = image.copy()
    clean = measurement["clean_mask"]
    skel = measurement["skeleton"]
    dist = measurement["distance_map"]

    mask_idx = clean > 0
    color_arr = np.array(color, dtype=np.float32)
    annotated[mask_idx] = (
        annotated[mask_idx].astype(np.float32) * (1.0 - alpha) + color_arr * alpha
    ).astype(np.uint8)

    contours, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(annotated, contours, -1, color, 2, lineType=cv2.LINE_AA)

    annotated[skel] = (0, 0, 255)

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
            annotated, label, (tx + 3, ty),
            font, font_scale, (255, 255, 255), thickness, lineType=cv2.LINE_AA,
        )

    return annotated


def _count_result_items(result):
    num_boxes = 0 if result.boxes is None else len(result.boxes)
    num_masks = 0 if (result.masks is None or getattr(result.masks, "data", None) is None) else len(result.masks.data)
    return num_boxes, num_masks


def measure_wire_from_crop(seg_model, crop_image, rule, aruco_info):
    """
    Run YOLO segmentation on crop image, extract mask, measure width.

    Args:
        seg_model: YOLO model (segmentation-capable)
        crop_image: numpy array (detect_roi crop)
        rule: dict with class_id, class_name, confidence, trim_fraction, p_low, p_high, etc.
        aruco_info: dict from detect_aruco_scale() or None

    Returns:
        {
            "found": bool,
            "measurement": dict or None,
            "median_width_mm": float or None,
            "mean_width_mm": float or None,
            "record": dict
        }
    """
    class_id = rule.get("class_id")
    class_name = rule.get("class_name")
    conf_thres = rule.get("confidence", 0.25)
    trim_fraction = rule.get("trim_fraction", 0.10)
    p_low = rule.get("p_low", 10)
    p_high = rule.get("p_high", 90)

    results = seg_model.predict(
        source=crop_image,
        conf=conf_thres,
        iou=0.30,
        imgsz=448,
        retina_masks=True,
        verbose=False,
    )

    if len(results) == 0:
        return {"found": False, "measurement": None, "median_width_mm": None, "mean_width_mm": None, "record": {}}

    result = results[0]
    names = result.names if hasattr(result, "names") else seg_model.names
    num_boxes, num_masks = _count_result_items(result)

    if num_boxes == 0 or num_masks == 0:
        return {"found": False, "measurement": None, "median_width_mm": None, "mean_width_mm": None, "record": {}}

    h, w = crop_image.shape[:2]
    n = min(num_boxes, num_masks)

    best = None
    best_conf = -1.0

    for i in range(n):
        cls_id = int(result.boxes.cls[i].item())
        score = float(result.boxes.conf[i].item())
        cls_name = get_class_name(names, cls_id)

        if not is_target_class(
            cls_id=cls_id,
            cls_name=cls_name,
            target_class_id=class_id,
            target_class_name=class_name,
        ):
            continue

        if score > best_conf:
            best_conf = score
            best = {"idx": i, "cls_id": cls_id, "cls_name": cls_name, "conf": score}

    if best is None:
        return {"found": False, "measurement": None, "median_width_mm": None, "mean_width_mm": None, "record": {}}

    raw_mask = mask_from_result(result, best["idx"], (h, w))

    measurement = measure_width_from_mask(
        raw_mask,
        trim_fraction=trim_fraction,
        p_low=p_low,
        p_high=p_high,
    )

    if measurement is None:
        return {
            "found": True,
            "measurement": None,
            "median_width_mm": None,
            "mean_width_mm": None,
            "record": {
                "class_id": best["cls_id"],
                "class_name": best["cls_name"],
                "confidence": best["conf"],
                "mask_available": True,
                "measurement_failed": True,
            },
        }

    record = {
        "class_id": best["cls_id"],
        "class_name": best["cls_name"],
        "confidence": best["conf"],
        "median_width_px": measurement["median_px"],
        "mean_width_px": measurement["mean_px"],
        "std_width_px": measurement["std_px"],
        "min_width_px": measurement["min_px"],
        "max_width_px": measurement["max_px"],
        "num_skeleton_points": measurement["num_points"],
        "median_width_mm": None,
        "mean_width_mm": None,
        "std_width_mm": None,
        "min_width_mm": None,
        "max_width_mm": None,
        "scale_px_per_mm": None,
        "scale_mm_per_px": None,
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
        })

    return {
        "found": True,
        "measurement": measurement,
        "median_width_mm": record["median_width_mm"],
        "mean_width_mm": record["mean_width_mm"],
        "record": record,
    }