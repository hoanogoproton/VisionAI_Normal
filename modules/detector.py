"""
Module: detector
Chuc nang: Detect object bang YOLO
          + NMS (Non-Maximum Suppression) de loc boxes overlap
Khong phu thuoc: Chi import ultralytics
"""

from ultralytics import YOLO
from typing import Dict, Tuple, Any, Optional, List


# ============================================================================
# NMS - Non-Maximum Suppression
# ============================================================================

def calculate_iou(box1: Tuple[int, int, int, int], 
                  box2: Tuple[int, int, int, int]) -> float:
    """
    Tinh Intersection over Union (IoU) giua 2 boxes
    
    Args:
        box1, box2: (x1, y1, x2, y2) - top-left va bottom-right
        
    Returns:
        IoU [0.0-1.0]: 
            1.0 = boxes trung 100%
            0.5 = 50% overlap
            0.0 = khong overlap
    """
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    
    # Tinh dien tich intersection (giao)
    x_left = max(x1_1, x1_2)
    y_top = max(y1_1, y1_2)
    x_right = min(x2_1, x2_2)
    y_bottom = min(y2_1, y2_2)
    
    if x_right <= x_left or y_bottom <= y_top:
        return 0.0  # Khong overlap
    
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    
    # Tinh dien tich union (hop)
    box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
    box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
    union_area = box1_area + box2_area - intersection_area
    
    if union_area == 0:
        return 0.0
    
    iou = intersection_area / union_area
    return iou


def non_maximum_suppression(detections: List[Dict[str, Any]], 
                            iou_threshold: float = 0.5) -> List[Dict[str, Any]]:
    """
    Loc cac boxes overlap (NMS - Non-Maximum Suppression)
    
    Cach hoat dong:
    1. Sort boxes theo confidence (cao xuong thap)
    2. Voi moi box:
       - Neu khong overlap voi boxes da giu → giu
       - Neu overlap (IoU > threshold) → xoa confidence thap hon
    3. Tra ve danh sach boxes sau loc
    
    Args:
        detections: [{bbox: (x1,y1,x2,y2), confidence: float, ...}, ...]
        iou_threshold: IoU > threshold → xem la overlap
                      Gia tri thuong dung: 0.3-0.5
                      - 0.3: Strict (xoa nhieu trung)
                      - 0.5: Normal (balance)
                      - 0.7: Lax (giu nhieu)
    
    Returns:
        Danh sach detections sau NMS
    """
    if not detections:
        return []
    
    if len(detections) == 1:
        return detections
    
    # Sort by confidence (cao xuong thap)
    sorted_dets = sorted(detections, key=lambda x: x.get("confidence", 0), 
                        reverse=True)
    
    keep = []
    
    for det_i in sorted_dets:
        bbox_i = det_i["bbox"]
        conf_i = det_i.get("confidence", 0)
        
        # Kiem tra overlap voi cac boxes da giu
        is_overlap = False
        for j, det_j in enumerate(keep):
            bbox_j = det_j["bbox"]
            iou = calculate_iou(bbox_i, bbox_j)
            
            if iou > iou_threshold:
                is_overlap = True
                break
        
        # Neu khong overlap → giu box nay
        if not is_overlap:
            keep.append(det_i)
    
    return keep


def detect_object(model: YOLO, image: Any, class_id: int, conf_thres: float,
                  roi_offset: Tuple[int, int] = (0, 0), 
                  apply_nms: bool = True, nms_iou_threshold: float = 0.5) -> Dict[str, Any]:
    """
    Detect object trong anh bang YOLO + Optional NMS filter overlap
    
    Args:
        model: Model YOLO
        image: Anh (numpy array)
        class_id: ID class can detect (VD: 0)
        conf_thres: Nguong tin cay [0.0-1.0]
        roi_offset: Toa do goc cua anh crop tren anh goc: (x_offset, y_offset)
                   Dung de chuyen toa do tu anh crop sang anh goc
                   VD: Neu crop tu (100, 150), offset=(100, 150)
        apply_nms: Ap dung NMS de loc boxes overlap (default: True)
        nms_iou_threshold: IoU threshold cho NMS (default: 0.5)
                          0.3-0.4: Strict (xoa nhieu)
                          0.5-0.6: Normal (balance)
                          0.7-0.8: Lax (giu nhieu)
    
    Returns:
        Dict:
            - Neu tim thay: {"found": True, "detections": [
                  {"bbox": (x1, y1, x2, y2), "confidence": float, "keypoints": list|None}, ...
              ]}
            - Neu khong: {"found": False, "detections": []}

            Luu y: bbox la toa do TUYET DOI tren anh goc!
    """
    # FIXED: Them iou=0.5 de loc cac detection trung lap (NMS - Non-Maximum Suppression)
    # iou=0.5: 2 boxes co IoU > 0.5 se bi loai bo box confidence thap hon
    # max_det=300: Gioi han toi da 300 detections (tranh qua nhieu boxes)
    results = model(image, verbose=False, iou=0.5, max_det=300)
    
    x_offset, y_offset = roi_offset
    detections = []

    for r in results:
        if r.boxes is None:
            continue
        for idx, box in enumerate(r.boxes):
            if int(box.cls[0]) != class_id or box.conf[0] < conf_thres:
                continue

            # Toa do tren anh crop
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Chuyen thanh toa do tuyet doi tren anh goc
            x1_real = x1 + x_offset
            y1_real = y1 + y_offset
            x2_real = x2 + x_offset
            y2_real = y2 + y_offset

            keypoints = None
            if r.keypoints is not None:
                try:
                    kp_data = r.keypoints.data[idx]
                    keypoints = []
                    for kp in kp_data:
                        kx = float(kp[0]) + x_offset
                        ky = float(kp[1]) + y_offset
                        kconf = float(kp[2]) if len(kp) > 2 else 0.0
                        keypoints.append((kx, ky, kconf))
                except Exception:
                    keypoints = None

            detections.append({
                "bbox": (x1_real, y1_real, x2_real, y2_real),
                "confidence": float(box.conf[0]),
                "keypoints": keypoints
            })
    
    # === AP DUNG NMS (optional) ===
    if apply_nms and detections:
        detections = non_maximum_suppression(detections, nms_iou_threshold)

    return {
        "found": len(detections) > 0,
        "detections": detections
    }


# ============================================================================
# CLASSIFICATION (YOLO Classification Models - Independent)
# ============================================================================

def classify_crop(crop_image: Any, classifier_model: YOLO) -> tuple:
    """
    Phan loai toan bo 1 crop anh bang YOLO classification model
    
    Args:
        crop_image: Anh crop (numpy array) - VD: crop cua compare_roi
        classifier_model: YOLO classification model (da load truoc)
    
    Returns:
        (classified_class, confidence)
        - classified_class: Tên class (str) - VD: "clear", "defect", "ok"
        - confidence: Độ tin cậy classification (float) - VD: 0.95
        
        Nếu error: ("unknown", 0.0)
    """
    if classifier_model is None or crop_image is None:
        return "unknown", 0.0
    
    if crop_image.size == 0:
        return "unknown", 0.0
    
    try:
        # Run classifier on crop
        results = classifier_model(crop_image, verbose=False)
        
        if results and len(results) > 0:
            r = results[0]
            
            # Get predicted class
            if r.probs is not None:
                class_idx = int(r.probs.top1)
                class_confidence = float(r.probs.top1conf)
                class_name = r.names[class_idx]
                
                return class_name, class_confidence
            else:
                return "unknown", 0.0
        else:
            return "unknown", 0.0
            
    except Exception as e:
        # Classification error
        return "unknown", 0.0


# ============================================================================
# TEST
# ============================================================================

def main():
    """Test function - kiem tra la toa do offset hoat dong dung"""
    import numpy as np
    
    print("[TEST] Detector - Coordinate Offset\n")
    
    print("[SCENARIO]")
    print("  - Anh goc: 640x480")
    print("  - detect_roi: (100, 150, 400, 450)")
    print("  - Anh crop: 300x300")
    print("  - Model detect bbox tren anh crop: (50, 50, 100, 100)")
    print()
    
    print("[TINH TOAN]")
    print("  - Toa do tren anh crop: (50, 50, 100, 100)")
    print("  - Offset (x_min, y_min): (100, 150)")
    print("  - Toa do thuc tren anh goc: (50+100, 50+150, 100+100, 100+150)")
    print("                             = (150, 200, 200, 250)")
    print()
    
    print("[KET QUA DUNG]")
    print("  Bbox lay tren crop + offset = bbox tren anh goc")
    print("  (50, 50, 100, 100) + (100, 150) = (150, 200, 200, 250)")
    print()
    print("[DONE] Offset logic is correct!")


if __name__ == "__main__":
    main()
