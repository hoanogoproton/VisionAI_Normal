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
    
    kp1 = keypoints[keypoint_idx_1]  # (x, y, conf)
    kp2 = keypoints[keypoint_idx_2]  # (x, y, conf)
    
    # Tinh goc: huong tu kp1 -> kp2
    dx = kp2[0] - kp1[0]
    dy = kp2[1] - kp1[1]
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