"""
Module: result_visualizer
Chuc nang: Ve bbox + ROI len anh de hien thi ket qua
           + Luu ket qua batch thanh anh (giong GUI)
Khong phu thuoc: Chi import cv2, numpy (opencv-python)
"""

import cv2
import os
import numpy as np
from typing import Dict, Tuple, Any, Optional, List
from modules.wire_measure import draw_measurement


def draw_roi(image: Any, roi: Tuple[int, int, int, int], 
             color: Tuple[int, int, int], thickness: int = 2, 
             label: str = "") -> None:
    """
    Ve hinh chu nhat ROI len anh
    
    Args:
        image: Anh (numpy array tu cv2.imread)
        roi: Toa do (x_min, y_min, x_max, y_max)
        color: Mau BGR (B, G, R), VD: (0, 255, 0) = green
        thickness: Do day duong ve
        label: Nhan van ban (VD: "detect_roi", "compare_roi")
    """
    x_min, y_min, x_max, y_max = roi
    cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, thickness)
    
    if label:
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        font_thickness = 1
        text_size = cv2.getTextSize(label, font, font_scale, font_thickness)[0]
        # Ve nhan tren canh trai-tren
        bg_x1 = x_min
        bg_y1 = y_min - text_size[1] - 5
        bg_x2 = x_min + text_size[0] + 5
        bg_y2 = y_min
        cv2.rectangle(image, (bg_x1, bg_y1), (bg_x2, bg_y2), color, -1)
        cv2.putText(image, label, (x_min + 2, y_min - 5), font, 
                   font_scale, (0, 0, 0), font_thickness)


def draw_bbox(image: Any, bbox: Tuple[int, int, int, int], 
              color: Tuple[int, int, int], thickness: int = 2, 
              confidence: float = 0.0) -> None:
    """
    Ve hinh chu nhat bbox tim duoc
    
    Args:
        image: Anh
        bbox: Toa do (x_min, y_min, x_max, y_max)
        color: Mau BGR
        thickness: Do day duong ve
        confidence: Diem tin cay (0.0-1.0), se hien thi tren anh
    """
    x_min, y_min, x_max, y_max = bbox
    cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, thickness)
    
    # Ve diem tin cay
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    font_thickness = 2
    conf_text = f"Conf: {confidence:.2f}"
    text_size = cv2.getTextSize(conf_text, font, font_scale, font_thickness)[0]
    
    bg_x1 = x_min
    bg_y1 = y_max + 5
    bg_x2 = x_min + text_size[0] + 5
    bg_y2 = y_max + 5 + text_size[1] + 5
    cv2.rectangle(image, (bg_x1, bg_y1), (bg_x2, bg_y2), color, -1)
    cv2.putText(image, conf_text, (x_min + 2, y_max + 20), font, 
               font_scale, (0, 0, 0), font_thickness)


def draw_status(image: Any, status: str, roi_id: str = "") -> None:
    """
    Ve trang thai OK/NG len goc anh
    
    Args:
        image: Anh
        status: "OK" hoac "NG" hoac chi tiet khac
        roi_id: ID roi (tuong ung)
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.0
    font_thickness = 2
    
    # Chon mau theo status
    if "OK" in status:
        color_bg = (0, 255, 0)  # Green for OK
        color_text = (0, 0, 0)
    else:
        color_bg = (0, 0, 255)  # Red for NG
        color_text = (255, 255, 255)
    
    # Ve text status
    status_text = status if not roi_id else f"{roi_id}: {status}"
    text_size = cv2.getTextSize(status_text, font, font_scale, font_thickness)[0]
    
    x, y = 10, 40
    cv2.rectangle(image, (x, y - text_size[1] - 10), 
                 (x + text_size[0] + 10, y + 10), color_bg, -1)
    cv2.putText(image, status_text, (x + 5, y), font, font_scale, 
               color_text, font_thickness)


def draw_keypoints_and_angle(image: Any, detect_result: Dict[str, Any],
                              angle_info: Dict[str, Any],
                              color_kp: Tuple[int, int, int] = (0, 255, 255),
                              color_line: Tuple[int, int, int] = (255, 0, 255)) -> None:
    """
    Ve keypoints va goc len anh (toa do tuyet doi tren anh goc)
    
    Args:
        image: Anh (se ve truc tiep len anh nay)
        detect_result: Ket qua detect (co "keypoints")
        angle_info: Dict tu compare_angle():
            {"kp1": (x,y), "kp2": (x,y), "measured_angle": float, ...}
        color_kp: Mau cham keypoint (BGR) - Vang
        color_line: Mau duong noi + goc (BGR) - Tim
    """
    if not angle_info:
        return
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    # 1. Ve tat ca keypoints (cham nho + so thu tu)
    keypoints = detect_result.get("keypoints")
    if keypoints is None:
        detections = detect_result.get("detections")
        if detections:
            best_det = max(detections, key=lambda det: det.get("confidence", 0.0))
            keypoints = best_det.get("keypoints", [])
        else:
            keypoints = []
    if keypoints:
        for i, kp in enumerate(keypoints):
            kx, ky = int(kp[0]), int(kp[1])
            cv2.circle(image, (kx, ky), 4, color_kp, -1)
            cv2.putText(image, str(i), (kx + 6, ky - 6), font, 0.4, color_kp, 1)
    
    # 2. Ve 2 keypoints dung cho angle (cham to hon)
    kp1 = (int(angle_info["kp1"][0]), int(angle_info["kp1"][1]))
    kp2 = (int(angle_info["kp2"][0]), int(angle_info["kp2"][1]))
    cv2.circle(image, kp1, 5, color_line, -1)
    cv2.circle(image, kp2, 5, color_line, -1)
    
    # 3. Ve duong noi voi mui ten (kp1 -> kp2)
    cv2.arrowedLine(image, kp1, kp2, color_line, 2, tipLength=0.15)
    
    # 4. Ve thong tin goc o giua duong noi
    mid_x = (kp1[0] + kp2[0]) // 2
    mid_y = (kp1[1] + kp2[1]) // 2
    
    measured = angle_info["measured_angle"]
    expected = angle_info["expected_angle"]
    tolerance = angle_info["angle_tolerance"]
    
    angle_text = f"{measured:.1f} (exp:{expected:.0f}+/-{tolerance:.0f})"
    cv2.putText(image, angle_text, (mid_x - 50, mid_y - 40), font, 0.5, color_line, 2)


def visualize_detection_result(
    image: Any,
    roi_data: Dict[str, Any],
    detect_result: Dict[str, Any],
    passed: bool,
    output_path: str = None,
    angle_info: Dict[str, Any] = None
) -> Any:
    """
    Ve toan bo ket qua detection: detect_roi, compare_roi, bbox, status
    
    Args:
        image: Anh goc
        roi_data: Du lieu ROI chua:
            {
                "roi_id": str,
                "camera": str,
                "detect_image": anh (crop roi),
                "compare_roi": (x_min, y_min, x_max, y_max),
                "rule": dict
            }
        detect_result: Ket qua detection:
            {
                "found": bool,
                "detections": [
                    {"bbox": (x_min, y_min, x_max, y_max), "confidence": float, "keypoints": list|None}, ...
                ]
            }
        passed: True neu OK, False neu NG
        output_path: Duong dan luu anh (neu None thi chi tra ve anh)
        
    Returns:
        Anh da ve (numpy array)
    """
    # Copy anh de khong chinh sua anh goc
    viz_image = image.copy()
    
    rule = roi_data["rule"]
    roi_id = roi_data["roi_id"]
    detect_roi = rule["detect_roi"]
    compare_roi = roi_data["compare_roi"]
    
    # Ve detect_roi (xanh)
    draw_roi(viz_image, detect_roi, (0, 255, 0), thickness=2, label="detect")
    
    # Ve compare_roi (do)
    draw_roi(viz_image, compare_roi, (0, 0, 255), thickness=2, label="compare")
    
    # Ve bbox neu tim thay
    detections = detect_result.get("detections")
    if detections is None:
        if detect_result.get("found"):
            bbox = detect_result.get("bbox")
            confidence = detect_result.get("confidence", 0.0)
            if bbox:
                draw_bbox(viz_image, bbox, (0, 255, 255), thickness=2, confidence=confidence)
    else:
        for det in detections:
            bbox = det.get("bbox")
            confidence = det.get("confidence", 0.0)
            if bbox:
                draw_bbox(viz_image, bbox, (0, 255, 255), thickness=2, confidence=confidence)
    
    # Ve keypoints va goc (neu co)
    if angle_info:
        draw_keypoints_and_angle(viz_image, detect_result, angle_info)
    
    # Ve status OK/NG
    status_text = "OK" if passed else "NG"
    draw_status(viz_image, status_text, roi_id)
    
    # Luu file neu co duong dan
    if output_path:
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
        cv2.imwrite(output_path, viz_image)
        print(f"[OK] Saved visualization: {output_path}")
    
    return viz_image


# ============================================================================
# BATCH RESULT SAVE (Thay thế per-ROI save)
# ============================================================================

def _draw_dashed_rect(img: np.ndarray, pt1: Tuple[int, int], pt2: Tuple[int, int],
                      color: Tuple[int, int, int], thickness: int = 2) -> None:
    """Ve hinh chu nhat net dut"""
    x1, y1 = pt1
    x2, y2 = pt2
    dash = 10
    gap = 10
    
    # Canh tren
    for i in range(x1, x2, dash + gap):
        cv2.line(img, (i, y1), (min(i + dash, x2), y1), color, thickness)
    # Canh duoi
    for i in range(x1, x2, dash + gap):
        cv2.line(img, (i, y2), (min(i + dash, x2), y2), color, thickness)
    # Canh trai
    for i in range(y1, y2, dash + gap):
        cv2.line(img, (x1, i), (x1, min(i + dash, y2)), color, thickness)
    # Canh phai
    for i in range(y1, y2, dash + gap):
        cv2.line(img, (x2, i), (x2, min(i + dash, y2)), color, thickness)


def _draw_roi_crop_for_batch(item: Dict) -> np.ndarray:
    """
    Ve duoc 1 ROI crop cho batch display (CHAT LUONG CAO).
    
    item:
        crop_image, roi_id, camera, passed, reason,
        detect_result, detect_roi, compare_roi
    """
    COLOR_OK = (0, 200, 0)        # Xanh
    COLOR_NG = (0, 0, 220)        # Do
    COLOR_BBOX = (0, 255, 255)    # Vang
    COLOR_COMPARE = (255, 200, 0) # Cyan nhat
    
    crop = item.get("crop_image")
    if crop is None or crop.size == 0:
        result_img = np.zeros((300, 400, 3), dtype=np.uint8)
        cv2.putText(result_img, "NO IMAGE", (80, 150), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (100, 100, 100), 3)
        return result_img

    passed = item.get("passed", False)

    measurement_result = item.get("measurement_result")
    if measurement_result is not None and measurement_result.get("measurement") is not None:
        meas = measurement_result["measurement"]
        overlay_color = (0, 255, 0) if passed else (0, 0, 255)

        median_px = meas["median_px"]
        median_mm = measurement_result.get("median_width_mm")
        if median_mm is not None:
            label = f"median={median_mm:.3f}mm ({median_px:.2f}px)"
        else:
            label = f"median={median_px:.2f}px"

        crop = draw_measurement(crop, meas, label=label, color=overlay_color, alpha=0.35, max_circles=20)

    # TANG CHAT LUONG: Resize crop to higher quality first
    scale_factor = 1.5  # Phong to anh crop
    new_h = int(crop.shape[0] * scale_factor)
    new_w = int(crop.shape[1] * scale_factor)
    result_img = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    h, w = result_img.shape[:2]

    roi_id = item.get("roi_id", "?")
    camera = item.get("camera", "?")
    passed = item.get("passed", False)
    reason = item.get("reason", "")
    detect_result = item.get("detect_result", {})
    detect_roi = item.get("detect_roi", (0, 0, 0, 0))
    compare_roi = item.get("compare_roi", (0, 0, 0, 0))
    
    # Offset: detect_roi top-left la goc (0,0) cua anh crop (TANG SCALE)
    offset_x = detect_roi[0] * scale_factor
    offset_y = detect_roi[1] * scale_factor
    
    # Mau theo ket qua
    color = COLOR_OK if passed else COLOR_NG
    
    # 1. Ve border (vien xanh/do) - TANG DO DAY
    cv2.rectangle(result_img, (0, 0), (w - 1, h - 1), color, 4)
    
    # 2. Ve ROI ID + status (goc tren trai) - TANG FONT
    status_text = "OK" if passed else "NG"
    label = f"{roi_id} ({camera}): {status_text}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_size = cv2.getTextSize(label, font, 0.8, 2)[0]
    cv2.rectangle(result_img, (0, 0), (text_size[0] + 12, text_size[1] + 16), color, -1)
    cv2.putText(result_img, label, (6, text_size[1] + 9), font, 0.8, (0, 0, 0), 2)
    
    # 3. Ve compare ROI (net dut) - TANG SCALE
    cx1 = int((compare_roi[0] - detect_roi[0]) * scale_factor)
    cy1 = int((compare_roi[1] - detect_roi[1]) * scale_factor)
    cx2 = int((compare_roi[2] - detect_roi[0]) * scale_factor)
    cy2 = int((compare_roi[3] - detect_roi[1]) * scale_factor)
    _draw_dashed_rect(result_img, (cx1, cy1), (cx2, cy2), COLOR_COMPARE, 2)
    
    # 4. Ve bbox neu tim thay
    detections = detect_result.get("detections")
    if detections is None:
        if detect_result.get("found", False):
            bbox = detect_result.get("bbox")
            confidence = detect_result.get("confidence", 0.0)
            if bbox:
                detections = [{"bbox": bbox, "confidence": confidence}]
            else:
                detections = []
        else:
            detections = []
    
    for det in detections:
        bbox = det.get("bbox")
        confidence = det.get("confidence", 0.0)
        if not bbox:
            continue
        
        # Chuyen toa do tuyet doi → toa do crop (TANG SCALE)
        bx1 = int((bbox[0] - detect_roi[0]) * scale_factor)
        by1 = int((bbox[1] - detect_roi[1]) * scale_factor)
        bx2 = int((bbox[2] - detect_roi[0]) * scale_factor)
        by2 = int((bbox[3] - detect_roi[1]) * scale_factor)
        
        # Ve bbox - TANG DO DAY
        cv2.rectangle(result_img, (bx1, by1), (bx2, by2), COLOR_BBOX, 3)
        
        # Ve confidence (duoi bbox) - TANG FONT
        conf_text = f"Conf: {confidence:.2f}"
        cv2.putText(result_img, conf_text, (bx1, by2 + 25),
                    font, 0.7, COLOR_BBOX, 2)
        
        # Ve center point - TANG RADIUS
        cx_pt = (bx1 + bx2) // 2
        cy_pt = (by1 + by2) // 2
        cv2.circle(result_img, (cx_pt, cy_pt), 5, color, -1)
    
    # 5. Ve keypoints va goc (neu co) - TANG SCALE
    angle_info = item.get("angle_info")
    if angle_info and detect_result.get("found"):
        COLOR_KP = (0, 255, 255)      # Vang
        COLOR_ANGLE = (255, 0, 255)   # Tim

        # Ve tat ca keypoints (cham nho + so thu tu)
        keypoints = detect_result.get("keypoints")
        if keypoints is None:
            det_list = detect_result.get("detections")
            if det_list:
                best_det = max(det_list, key=lambda det: det.get("confidence", 0.0))
                keypoints = best_det.get("keypoints", [])
            else:
                keypoints = []

        for i, kp in enumerate(keypoints):
            kx = int((kp[0] - detect_roi[0]) * scale_factor)
            ky = int((kp[1] - detect_roi[1]) * scale_factor)
            cv2.circle(result_img, (kx, ky), 4, COLOR_KP, -1)
            cv2.putText(result_img, str(i), (kx + 6, ky - 6), font, 0.4, COLOR_KP, 1)

        # Ve 2 keypoints dung cho angle (cham to)
        akp1 = (
            int((angle_info["kp1"][0] - detect_roi[0]) * scale_factor),
            int((angle_info["kp1"][1] - detect_roi[1]) * scale_factor),
        )
        akp2 = (
            int((angle_info["kp2"][0] - detect_roi[0]) * scale_factor),
            int((angle_info["kp2"][1] - detect_roi[1]) * scale_factor),
        )
        cv2.circle(result_img, akp1, 7, COLOR_ANGLE, -1)
        cv2.circle(result_img, akp2, 7, COLOR_ANGLE, -1)

        # Ve duong noi voi mui ten
        cv2.arrowedLine(result_img, akp1, akp2, COLOR_ANGLE, 2, tipLength=0.2)

        # Ve goc
        mid_x = (akp1[0] + akp2[0]) // 2
        mid_y = (akp1[1] + akp2[1]) // 2
        angle_text = f"{angle_info['measured_angle']:.1f}"
        cv2.putText(result_img, angle_text, (mid_x, mid_y - 40),
                    font, 0.6, COLOR_ANGLE, 2)

    # 6. Ve reason (goc duoi) - TANG FONT
    if reason:
        cv2.putText(result_img, reason, (6, h - 10), 
                    font, 0.6, (200, 200, 200), 2)
    
    return result_img


def _arrange_batch_grid(roi_images: List[np.ndarray]) -> np.ndarray:
    """
    Xep danh sach anh ROI thanh luoi (grid) - HI EM CAO.
    Tương tự result_gui._arrange_grid() nhưng resolution cao hơn
    """
    n = len(roi_images)
    
    if n == 0:
        return np.zeros((400, 800, 3), dtype=np.uint8)
    
    # Tinh so hang x cot (layout vuong)
    cols = max(1, int(np.ceil(np.sqrt(n))))
    rows = max(1, int(np.ceil(n / cols)))
    
    # Kich thuoc moi cell: TANG RESOLUTION - 400x300 (thay vi 200x150)
    cell_w = 700
    cell_h = 700
    
    grid_w = cols * cell_w
    grid_h = rows * cell_h
    
    # Tao grid canvas
    grid = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)
    
    # Dat anh vao grid
    for idx, img in enumerate(roi_images):
        row = idx // cols
        col = idx % cols
        y_start = row * cell_h
        x_start = col * cell_w
        
        # Resize anh de fit cell (giu aspect ratio)
        h, w = img.shape[:2]
        scale = min(cell_w / w, cell_h / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        # Tang quality: INTER_CUBIC instead of default
        img_resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        
        # Dat giua cell
        y_offset = (cell_h - new_h) // 2
        x_offset = (cell_w - new_w) // 2
        y_end = y_start + y_offset + new_h
        x_end = x_start + x_offset + new_w
        
        grid[y_start + y_offset:y_end, x_start + x_offset:x_end] = img_resized
    
    return grid


def save_batch_result(roi_items: List[Dict], final_status: str, 
                      batch_info: Dict, output_path: str,
                      total_ok: int = 0, total_ng: int = 0) -> None:
    """
    Luu ket qua batch thanh anh (chat luong cao, khong header/footer).
    - Chi hien thi batch grid voi anh ROI crop
    - Kich thuoc cao (400x300/cell)
    - Loai bo header va footer
    
    Args:
        roi_items: Danh sach ROI items
        final_status: "OK" hoac "NG" (không dùng)
        batch_info: {"batch_num": int, ...} (không dùng)
        output_path: Duong dan luu anh
        total_ok, total_ng: Không dùng (giữ cho compatibility)
    """
    # --- Buoc 1: Ve tung ROI crop ---
    roi_images = []
    for item in roi_items:
        roi_img = _draw_roi_crop_for_batch(item)
        roi_images.append(roi_img)
    
    # --- Buoc 2: Xep luoi (resolution cao) ---
    grid = _arrange_batch_grid(roi_images)
    
    # --- Buoc 3: Luu file truc tiep (khong header/footer) ---
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    cv2.imwrite(output_path, grid)
    print(f"[OK] Saved batch result (HD): {output_path}")


# ============================================================================
# TEST
# ============================================================================

def create_sample_image(width: int = 640, height: int = 480) -> Any:
    """Tao anh mau de test"""
    # Tao anh trang
    image = cv2.cvtColor(
        cv2.create(height, width, cv2.CV_8UC3),
        cv2.COLOR_RGB2BGR
    )
    # Tao gradient
    for i in range(height):
        for j in range(width):
            image[i, j] = [50 + i % 50, 100 + j % 50, 150]
    return image

def main():
    """Test function"""
    import numpy as np
    
    print("[TEST] Result Visualizer\n")
    
    # Tao anh test
    print("[1] Creating test image...")
    image = np.ones((480, 640, 3), dtype=np.uint8) * 200
    
    # Tao du lieu ROI
    roi_data = {
        "roi_id": "roi_001",
        "camera": "CAM1",
        "detect_image": image[100:400, 150:550],
        "compare_roi": (100, 150, 300, 250),
        "rule": {
            "detect_roi": (100, 100, 400, 400),
            "compare_roi": (120, 120, 380, 380),
            "class_id": 0,
            "class_name": "mark"
        }
    }
    
    # Test case 1: Tim thay object
    print("[2] Test case 1: Found object (PASS)")
    detect_result = {
        "found": True,
        "detections": [
            {"bbox": (150, 150, 250, 250), "confidence": 0.85, "keypoints": None}
        ]
    }
    viz_image = visualize_detection_result(image, roi_data, detect_result, 
                                           passed=True, 
                                           output_path="output/test_ok.jpg")
    print("[OK] Visualization saved\n")
    
    # Test case 2: Khong tim thay
    print("[3] Test case 2: Object not found (FAIL)")
    detect_result = {
        "found": False,
        "detections": []
    }
    viz_image = visualize_detection_result(image, roi_data, detect_result, 
                                           passed=False, 
                                           output_path="output/test_ng.jpg")
    print("[OK] Visualization saved\n")
    
    # Test case 3: Tim thay nhung o ngoai roi
    print("[4] Test case 3: Found but outside compare ROI (FAIL)")
    detect_result = {
        "found": True,
        "detections": [
            {"bbox": (350, 350, 450, 450), "confidence": 0.75, "keypoints": None}
        ]
    }
    viz_image = visualize_detection_result(image, roi_data, detect_result, 
                                           passed=False, 
                                           output_path="output/test_out_of_roi.jpg")
    print("[OK] Visualization saved\n")
    
    print("[DONE] Tests completed!")


if __name__ == "__main__":
    main()
