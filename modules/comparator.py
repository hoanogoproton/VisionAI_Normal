import math


def is_center_inside_roi(bbox, roi):
    """
    Kiểm tra tâm của bounding box có nằm trong vùng ROI hay không.
    bbox: [x1, y1, x2, y2]
    roi:  [rx1, ry1, rx2, ry2]
    """
    bx1, by1, bx2, by2 = bbox
    rx1, ry1, rx2, ry2 = roi

    # Tính toán tọa độ tâm (center point)
    center_x = (bx1 + bx2) / 2
    center_y = (by1 + by2) / 2

    # Kiểm tra tâm có nằm trong giới hạn của ROI
    return rx1 <= center_x <= rx2 and ry1 <= center_y <= ry2


def compare_detection(detect_result, compare_roi):
    """
    Hàm chính để so sánh kết quả detect với vùng ROI.
    """
    detections = detect_result.get("detections")
    if detections is not None:
        if not detections:
            return False, "NOT_FOUND"
        best_det = max(detections, key=lambda det: det.get("confidence", 0.0))
        bbox = best_det.get("bbox")
    else:
        if not detect_result.get("found", False):
            return False, "NOT_FOUND"
        bbox = detect_result.get("bbox")
    if not bbox:
        return False, "INVALID_BBOX"

    if not is_center_inside_roi(bbox, compare_roi):
        return False, "OUT_OF_COMPARE_ROI"

    return True, "OK"


def compare_angle(detect_result, keypoint_idx_1, keypoint_idx_2,
                  expected_angle, angle_tolerance):
    """
    Kiem tra goc giua 2 keypoints co nam trong khoang cho phep khong.
    
    Args:
        detect_result: Ket qua tu detect_object() - phai co "keypoints" hoac "detections"
        keypoint_idx_1: Index keypoint goc (VD: 0)
        keypoint_idx_2: Index keypoint ngon (VD: 3)
        expected_angle: Goc mong muoi (do, 0-360)
        angle_tolerance: Sai so cho phep (do, VD: 20 = +/-20)
    
    Returns:
        (passed, reason, angle_info)
        - passed: True/False
        - reason: "ANGLE_OK" / "WRONG_ANGLE(xxx.x)" / "NO_KEYPOINTS"
        - angle_info: dict chua thong tin goc (de ve len anh)
    
    Vi du:
        expected_angle = 120, angle_tolerance = 20
        measured = 130 -> diff = 10 < 20 -> OK
        measured = 80  -> diff = 40 > 20 -> NG
    """
    try:
        if (isinstance(keypoint_idx_1, bool) or isinstance(keypoint_idx_2, bool)
                or not isinstance(keypoint_idx_1, int)
                or not isinstance(keypoint_idx_2, int)
                or keypoint_idx_1 < 0 or keypoint_idx_2 < 0
                or keypoint_idx_1 == keypoint_idx_2):
            return False, "INVALID_ANGLE_CONFIG", {}
        expected_angle = float(expected_angle)
        angle_tolerance = float(angle_tolerance)
        if (not math.isfinite(expected_angle) or not math.isfinite(angle_tolerance)
                or not 0 <= angle_tolerance <= 180):
            return False, "INVALID_ANGLE_CONFIG", {}
        expected_angle %= 360
    except (TypeError, ValueError, OverflowError):
        return False, "INVALID_ANGLE_CONFIG", {}

    detections = detect_result.get("detections")
    if detections is not None:
        if not detections:
            return False, "NOT_FOUND", {}
        best_det = max(detections, key=lambda det: det.get("confidence", 0.0))
        keypoints = best_det.get("keypoints")
    else:
        if not detect_result.get("found"):
            return False, "NOT_FOUND", {}
        keypoints = detect_result.get("keypoints")
    if keypoints is None or len(keypoints) == 0:
        return False, "NO_KEYPOINTS", {}
    
    if keypoint_idx_1 >= len(keypoints) or keypoint_idx_2 >= len(keypoints):
        return False, "KEYPOINT_INDEX_OUT_OF_RANGE", {}
    
    try:
        kp1 = keypoints[keypoint_idx_1]  # (x, y, conf)
        kp2 = keypoints[keypoint_idx_2]
        x1, y1, x2, y2 = map(float, (kp1[0], kp1[1], kp2[0], kp2[1]))
        if not all(math.isfinite(v) for v in (x1, y1, x2, y2)):
            return False, "INVALID_KEYPOINTS", {}
    except (TypeError, ValueError, IndexError, OverflowError):
        return False, "INVALID_KEYPOINTS", {}

    # Image coordinates: clockwise direction because y increases downwards.
    dx, dy = x2 - x1, y2 - y1
    if dx == 0 and dy == 0:
        return False, "DEGENERATE_KEYPOINTS", {}
    angle = math.degrees(math.atan2(dy, dx))
    
    # Normalize ve [0, 360)
    if angle < 0:
        angle += 360
    
    # Tinh sai lech goc (xu ly truong hop qua 0/360)
    angle_diff = abs(angle - expected_angle)
    if angle_diff > 180:
        angle_diff = 360 - angle_diff
    
    # Thong tin goc (de ve len anh)
    angle_info = {
        "measured_angle": angle,
        "expected_angle": expected_angle,
        "angle_tolerance": angle_tolerance,
        "angle_diff": angle_diff,
        "keypoint_idx_1": keypoint_idx_1,
        "keypoint_idx_2": keypoint_idx_2,
        "kp1": (kp1[0], kp1[1]),
        "kp2": (kp2[0], kp2[1]),
    }
    
    if angle_diff <= angle_tolerance:
        return True, "ANGLE_OK", angle_info
    else:
        return False, f"WRONG_ANGLE({angle:.1f})", angle_info


def compare_count_and_angle(detect_result, compare_roi, min_count=None,
                            max_count=None, keypoint_idx_1=None,
                            keypoint_idx_2=None, expected_angle=None,
                            angle_tolerance=None):
    """Count centers in compare_roi AND check every counted object's angle.

    All angle fields empty: count only (backward compatible).
    Any angle field supplied: all four are required.
    Count is based on detections AFTER detector class/confidence filtering/NMS,
    never on the number of objects that pass the angle check.

    Returns (passed, reason, details). details["angle_results"] contains an
    entry for EVERY counted object, including failures with no measurable angle.
    """
    detections = detect_result.get("detections")
    if detections is None:
        detections = [detect_result] if detect_result.get("found") else []
    valid = []
    for det in detections:
        bbox = det.get("bbox")
        try:
            if (bbox is None or len(bbox) != 4
                    or not all(math.isfinite(float(v)) for v in bbox)
                    or bbox[0] > bbox[2] or bbox[1] > bbox[3]):
                return False, "INVALID_BBOX", {
                    "detected_count": len(valid), "angle_results": []}
            if is_center_inside_roi(bbox, compare_roi):
                valid.append(det)
        except (TypeError, ValueError, OverflowError):
            return False, "INVALID_BBOX", {
                "detected_count": len(valid), "angle_results": []}

    detected_count = len(valid)
    required_min = min_count if min_count is not None else 1
    details = {"detected_count": detected_count, "angle_results": []}
    if (isinstance(required_min, bool) or not isinstance(required_min, int)
            or required_min < 0
            or (max_count is not None and (
                isinstance(max_count, bool) or not isinstance(max_count, int)
                or max_count < required_min))):
        return False, "INVALID_COUNT_CONFIG", details

    if max_count is None:
        count_passed = detected_count >= required_min
        count_reason = (
            f"COUNT_OK({detected_count}>={required_min})" if count_passed
            else f"COUNT_NG({detected_count}<{required_min})")
    else:
        count_passed = required_min <= detected_count <= max_count
        count_reason = (
            f"COUNT_OK({detected_count}/{required_min}-{max_count})" if count_passed
            else f"COUNT_NG({detected_count}/{required_min}-{max_count})")

    angle_fields = (keypoint_idx_1, keypoint_idx_2,
                    expected_angle, angle_tolerance)
    if all(value is None for value in angle_fields):
        return count_passed, count_reason, details
    if any(value is None for value in angle_fields):
        return False, "INVALID_ANGLE_CONFIG", details

    # Validate even when zero objects were allowed by min_count=0.
    try:
        config_valid = (
            isinstance(keypoint_idx_1, int) and not isinstance(keypoint_idx_1, bool)
            and isinstance(keypoint_idx_2, int) and not isinstance(keypoint_idx_2, bool)
            and keypoint_idx_1 >= 0 and keypoint_idx_2 >= 0
            and keypoint_idx_1 != keypoint_idx_2
            and math.isfinite(float(expected_angle))
            and math.isfinite(float(angle_tolerance))
            and 0 <= float(angle_tolerance) <= 180)
    except (TypeError, ValueError, OverflowError):
        config_valid = False
    if not config_valid:
        return False, "INVALID_ANGLE_CONFIG", details

    for index, det in enumerate(valid, start=1):
        angle_passed, angle_reason, angle_info = compare_angle(
            {"found": True, "keypoints": det.get("keypoints")},
            keypoint_idx_1, keypoint_idx_2, expected_angle, angle_tolerance)
        details["angle_results"].append({
            "object_index": index,
            "passed": angle_passed,
            "reason": angle_reason,
            "angle_info": angle_info,
            "detection": det,
        })

    # Preserve count failure as primary reason, but still measure all angles.
    if not count_passed:
        return False, count_reason, details
    failures = [item for item in details["angle_results"] if not item["passed"]]
    if failures:
        first = failures[0]
        return False, (
            f"COUNT_ANGLE_NG(obj{first['object_index']}:{first['reason']};"
            f"failed={len(failures)}/{detected_count})"), details
    return True, f"COUNT_ANGLE_OK({detected_count})", details


def compare_classification(classified_class: str, class_confidence: float, 
                          expected_class: str, confidence_threshold: float = 0.5) -> tuple:
    """
    So sanh ket qua classification voi class mong doi
    
    Args:
        classified_class: Tên class thực tế từ classifier (str) - VD: "clear"
        class_confidence: Độ tin cậy từ classifier (float) - VD: 0.85
        expected_class: Class mong đợi (str) - VD: "clear"
        confidence_threshold: Ngưỡng tin cậy tối thiểu (default: 0.5)
    
    Returns:
        (passed, reason)
        - passed: True/False
        - reason: "CLASS_OK(clear:0.85)" / "CLASS_NG(defect)" / "LOW_CONFIDENCE(0.45<0.5)"
    """
    
    # Check confidence threshold first
    if class_confidence < confidence_threshold:
        return False, f"LOW_CONFIDENCE({class_confidence:.2f}<{confidence_threshold})"
    
    # Check if class matches expected
    if classified_class.lower() == expected_class.lower():
        return True, f"CLASS_OK({classified_class}:{class_confidence:.2f})"
    else:
        return False, f"CLASS_NG({classified_class}!={expected_class})"


# --- PHẦN TEST NHANH ---
if __name__ == "__main__":
    # Vùng ROI ví dụ (Hình vuông từ 100 đến 500)
    roi_area = [100, 100, 500, 500]

    test_cases = [
        {
            "name": "Trường hợp 1: Tâm nằm trong ROI (Hợp lệ)",
            "detect": {"found": True, "bbox": [80, 80, 150, 150]}, # Tâm là (115, 115) -> Trong ROI
        },
        {
            "name": "Trường hợp 2: Toàn bộ box ở ngoài (Không hợp lệ)",
            "detect": {"found": True, "bbox": [0, 0, 50, 50]},    # Tâm là (25, 25) -> Ngoài ROI
        },
        {
            "name": "Trường hợp 3: Không tìm thấy đối tượng",
            "detect": {"found": False, "bbox": None},
        },
        {
            "name": "Trường hợp 4: Box lớn trùm ROI nhưng tâm ở ngoài",
            "detect": {"found": True, "bbox": [0, 0, 100, 100]},  # Tâm là (50, 50) -> Ngoài ROI
        }
    ]

    print(f"{'Test Case':<40} | {'Result':<10} | {'Status'}")
    print("-" * 70)
    
    for case in test_cases:
        res, msg = compare_detection(case["detect"], roi_area)
        print(f"{case['name']:<40} | {str(res):<10} | {msg}")