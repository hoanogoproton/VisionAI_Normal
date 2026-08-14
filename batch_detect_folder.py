"""
Script: Batch Detect Folder
Chức năng: Load tất cả ảnh từ thư mục → Detect → Lưu kết quả batch

Cách dùng:
    python batch_detect_folder.py
    
    Nhập:
    - Đường dẫn thư mục ảnh (VD: images/test_images)
    - Mã sản phẩm (VD: ABC123x)
    - Camera (VD: CAM1)
    
    Output:
    - Batch result image: output/results/{timestamp}_batch_RESULT.jpg
    - Log file: output/logs/batch_detect.log
"""

import os
import sys
import cv2
import glob
import time
import csv
import re
from datetime import datetime
from pathlib import Path

# Import modules
from modules.config_loader import load_config
from modules.csv_loader import load_product_csv, build_product_signal_plan
from modules.camera_config_loader import load_camera_config, get_camera_folder
from modules.roi_manager import prepare_roi_data, crop_roi
from modules.model_manager import get_model
from modules.detector import detect_object, classify_crop
from modules.comparator import compare_detection, compare_angle, is_center_inside_roi, compare_classification
from modules.result_visualizer import save_batch_result

# ============================================================================
# Logger
# ============================================================================

class BatchLogger:
    """Simple logger for batch processing"""
    
    def __init__(self, log_dir: str):
        self.log_file = os.path.join(log_dir, f"batch_detect_{datetime.now().strftime('%Y%m%d.log')}")
        os.makedirs(log_dir, exist_ok=True)
    
    def log(self, message: str) -> None:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_msg = f"[{timestamp}] {message}"
        print(log_msg)
        
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(log_msg + "\n")

# ============================================================================
# Main Batch Processing
# ============================================================================

def list_image_files(folder_path: str, max_images: int = 50) -> list:
    """
    Lấy danh sách tất cả ảnh trong thư mục
    
    Args:
        folder_path: Đường dẫn thư mục
        max_images: Số lượng ảnh tối đa (default: 50)
        
    Returns:
        Danh sách đường dẫn ảnh
    """
    if not os.path.isdir(folder_path):
        raise FileNotFoundError(f"Folder not found: {folder_path}")
    
    # Hỗ trợ các định dạng ảnh phổ biến
    extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.JPG", "*.JPEG", "*.PNG"]
    image_files = []
    
    for ext in extensions:
        pattern = os.path.join(folder_path, ext)
        image_files.extend(glob.glob(pattern))
    
    return sorted(image_files)[:max_images]  # Giới hạn số ảnh / batch (tránh memory)


def save_result_csv(results: list, output_dir: str, product_code: str) -> str:
    """
    Lưu kết quả chi tiết vào file CSV (dễ import Excel)
    
    Args:
        results: Danh sách {image_name, roi_id, camera, result, reason}
        output_dir: Thư mục output
        product_code: Mã sản phẩm (cho tên file)
        
    Returns:
        Đường dẫn file CSV
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(output_dir, f"{timestamp}_result_{product_code}.csv")
    
    os.makedirs(output_dir, exist_ok=True)
    
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, 
            fieldnames=["Image", "ROI_ID", "Camera", "Result", "Reason"],
            delimiter="|"
        )
        
        writer.writeheader()
        writer.writerows(results)
    
    return csv_path


def sanitize_filename(value: str, fallback: str = "item") -> str:
    """Make a filesystem-safe filename fragment."""
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value or "")
    cleaned = cleaned.strip("._-")
    return cleaned or fallback


def save_ng_roi_images(roi_items: list, output_dir: str, product_code: str) -> tuple[str, str, int]:
    """
    Lưu tất cả ROI NG ra 1 folder để dùng train thêm.

    Args:
        roi_items: Danh sách GUI ROI items đã xử lý
        output_dir: Thư mục output gốc
        product_code: Mã sản phẩm

    Returns:
        (ng_folder, index_csv, saved_count)
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    ng_folder = os.path.join(output_dir, f"{timestamp}_ng_rois_{product_code}")
    os.makedirs(ng_folder, exist_ok=True)

    index_rows = []
    saved_count = 0

    for item in roi_items:
        if item.get("passed", True):
            continue

        crop_image = item.get("crop_image")
        if crop_image is None or getattr(crop_image, "size", 0) == 0:
            continue

        source_image = sanitize_filename(item.get("source_image", "unknown"))
        roi_id = sanitize_filename(item.get("roi_id", "roi"))
        camera = sanitize_filename(item.get("camera", "camera"))
        reason = sanitize_filename(item.get("reason", "NG"))

        filename = f"{saved_count + 1:05d}_{source_image}__{roi_id}__{camera}__{reason}.png"
        image_path = os.path.join(ng_folder, filename)

        if not cv2.imwrite(image_path, crop_image):
            continue

        h, w = crop_image.shape[:2]
        index_rows.append({
            "SavedImage": filename,
            "SourceImage": item.get("source_image", ""),
            "ROI_ID": item.get("roi_id", ""),
            "Camera": item.get("camera", ""),
            "Result": "NG",
            "Reason": item.get("reason", ""),
            "Width": w,
            "Height": h,
        })
        saved_count += 1

    index_csv = os.path.join(ng_folder, "index.csv")
    with open(index_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["SavedImage", "SourceImage", "ROI_ID", "Camera", "Result", "Reason", "Width", "Height"],
            delimiter="|"
        )
        writer.writeheader()
        writer.writerows(index_rows)

    return ng_folder, index_csv, saved_count


def batch_detect_folder(input_folder: str, product_code: str, camera: str, max_images: int = 50):
    """
    Xử lý batch: Load tất cả ảnh → Detect → Lưu kết quả
    
    Args:
        input_folder: Thư mục chứa ảnh
        product_code: Mã sản phẩm (VD: ABC123x)
        camera: Tên camera (VD: CAM1)
        max_images: Số lượng ảnh tối đa
    """
    
    # ┌─── INIT ────────────────────────────────────────────┐
    
    cfg = load_config("config/config.yaml")
    logger = BatchLogger(cfg['paths']['log_dir'])
    
    logger.log("=" * 70)
    logger.log("BATCH DETECT START")
    logger.log(f"Folder: {input_folder}")
    logger.log(f"Product: {product_code}")
    logger.log(f"Camera: {camera}")
    logger.log(f"Max Images: {max_images}")
    logger.log("=" * 70)
    
    # Load product CSV
    try:
        csv_path = cfg['product']['csv_path'].format(code=product_code)
        roi_rules = load_product_csv(csv_path)
        logger.log(f"[LOAD] Product CSV: {len(roi_rules)} ROIs loaded")
        signal_plan = build_product_signal_plan(roi_rules)
        logger.log(f"[LOAD] Signal order: {len(signal_plan)} class groups")
        for entry in signal_plan:
            roi_ids = ", ".join(entry["roi_ids"])
            logger.log(
                f"[LOAD] {entry['ok_signal']} / {entry['ng_signal']} -> "
                f"model={entry['model_name']}, class_id={entry['class_id']}, class_name={entry['class_name']} | ROIs: {roi_ids}"
            )
    except FileNotFoundError as e:
        logger.log(f"[ERROR] Product CSV not found: {e}")
        return
    
    # Load camera config
    try:
        camera_config = load_camera_config(
            cfg['camera']['config_csv'],
            create_folders=False,
            verbose=False
        )
        logger.log(f"[LOAD] Camera config loaded")
    except Exception as e:
        logger.log(f"[ERROR] Camera config error: {e}")
        return
    
    # List image files
    try:
        image_files = list_image_files(input_folder, max_images=max_images)
        if not image_files:
            logger.log("[WARNING] No image files found")
            return
        logger.log(f"[LOAD] Found {len(image_files)} image files")
    except Exception as e:
        logger.log(f"[ERROR] Cannot list images: {e}")
        return
    
    # ┌─── PROCESSING ──────────────────────────────────────┐
    
    start_time = time.time()
    batch_num = 1
    roi_results = []  # Collect all results
    detail_results = []  # For CSV (with image name)
    gui_roi_items = []  # For batch image
    
    for idx, image_path in enumerate(image_files, 1):
        image_name = os.path.basename(image_path)
        logger.log(f"\n[{idx}/{len(image_files)}] Processing: {image_name}")
        
        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                logger.log(f"  [ERROR] Cannot load image")
                continue
            
            h, w = image.shape[:2]
            logger.log(f"  Image size: {w}x{h}")
            
            # Filter ROIs by camera
            cam_rules = [r for r in roi_rules if r["camera"] == camera]
            if not cam_rules:
                logger.log(f"  [SKIP] No ROIs for camera {camera}")
                continue
            
            logger.log(f"  Processing {len(cam_rules)} ROIs...")
            
            # Process each ROI
            for rule in cam_rules:
                try:
                    check_mode = rule.get("check_mode", "compare").lower()
                    angle_info = {}
                    detect_result = {"found": False, "detections": []}
                    display_crop_image = None
                    
                    # ===== MODE: CLASSIFY (Pure Classification - No Detection) =====
                    if check_mode == "classify":
                        expected_class = (rule.get("class_name") or "").strip()
                        class_threshold = float(rule.get("confidence", 0.5))
                        
                        if not expected_class:
                            logger.log(f"    [ERROR] {rule['roi_id']}: classify mode requires class_name as expected class")
                            reason = "CONFIG_ERROR(missing_class_name)"
                            passed = False
                        else:
                            try:
                                classifier = get_model(rule["model_name"])
                                compare_roi = rule["compare_roi"]
                                crop_image = crop_roi(image, compare_roi)
                                # For UI/batch image display, keep detect ROI as base and overlay compare ROI.
                                display_crop_image = crop_roi(image, rule["detect_roi"])
                                
                                if crop_image.size == 0:
                                    reason = "CROP_ERROR(empty_crop)"
                                    passed = False
                                else:
                                    classified_class, class_confidence = classify_crop(crop_image, classifier)
                                    passed, reason = compare_classification(
                                        classified_class,
                                        class_confidence,
                                        expected_class,
                                        class_threshold
                                    )
                                    
                            except FileNotFoundError as e:
                                reason = f"CLASSIFIER_ERROR({rule['model_name']})"
                                passed = False
                            except Exception as e:
                                reason = f"CLASSIFY_ERROR: {str(e)}"
                                passed = False
                    
                    # ===== OTHER MODES: Detection-based =====
                    else:
                        roi_data = prepare_roi_data(image, rule)
                        display_crop_image = roi_data["detect_image"]
                        model = get_model(rule["model_name"])
                        
                        detect_roi = rule["detect_roi"]
                        roi_offset = (detect_roi[0], detect_roi[1])
                        
                        # Lọc trùng lặp bằng NMS (Non-Maximum Suppression)
                        # IoU threshold = 0.4: Loại bỏ các box đè lên nhau > 40%
                        detect_result = detect_object(
                            model,
                            roi_data["detect_image"],
                            rule["class_id"],
                            rule["confidence"],
                            roi_offset=roi_offset,
                            apply_nms=True,
                            nms_iou_threshold=0.4
                        )
                        
                        if check_mode == "count":
                            detections = detect_result.get("detections", [])
                            compare_roi = rule["compare_roi"]
                            in_roi_detections = [
                                det for det in detections
                                if det.get("bbox") and is_center_inside_roi(det.get("bbox"), compare_roi)
                            ]

                            # Count mode: chi tinh object co tam nam trong compare_roi
                            detect_result["detections"] = in_roi_detections
                            detected_count = len(in_roi_detections)
                            min_count = rule.get("min_count")
                            max_count = rule.get("max_count")
                            required_min = min_count if min_count is not None else 1
                            
                            if max_count is None:
                                passed = detected_count >= required_min
                                reason = (
                                    f"COUNT_OK({detected_count}>={required_min},in_roi)"
                                    if passed else f"COUNT_NG({detected_count}<{required_min},in_roi)"
                                )
                            else:
                                passed = required_min <= detected_count <= max_count
                                reason = (
                                    f"COUNT_OK({detected_count}/{required_min}-{max_count},in_roi)"
                                    if passed else f"COUNT_NG({detected_count}/{required_min}-{max_count},in_roi)"
                                )
                        
                        elif check_mode == "forbidden":
                            detections = detect_result.get("detections", [])
                            detected_count = len(detections)
                            
                            if detected_count == 0:
                                passed = True
                                reason = "FORBIDDEN_OK(no_object)"
                            else:
                                passed = False
                                reason = f"FORBIDDEN_NG({detected_count}_found)"
                        
                        else:  # compare mode
                            passed, reason = compare_detection(
                                detect_result,
                                rule["compare_roi"]
                            )
                            
                            if rule.get("keypoint_idx_1") is not None:
                                if detect_result.get("found"):
                                    angle_passed, angle_reason, angle_info = compare_angle(
                                        detect_result,
                                        rule["keypoint_idx_1"],
                                        rule["keypoint_idx_2"],
                                        rule["expected_angle"],
                                        rule["angle_tolerance"]
                                    )
                                    if passed and not angle_passed:
                                        passed = False
                                        reason = angle_reason
                    
                    status_str = "OK" if passed else "NG"
                    logger.log(f"    {rule['roi_id']}: {status_str} ({reason})")
                    
                    roi_results.append({
                        "roi_id": rule["roi_id"],
                        "camera": camera,
                        "pass": passed,
                        "reason": reason
                    })
                    
                    # === THEM: Chi tiết cho CSV ===
                    detail_results.append({
                        "Image": image_name,
                        "ROI_ID": rule["roi_id"],
                        "Camera": camera,
                        "Result": status_str,
                        "Reason": reason
                    })

                    if display_crop_image is None:
                        display_crop_image = crop_roi(image, rule["detect_roi"])
                    
                    gui_roi_items.append({
                        "roi_id": rule["roi_id"],
                        "camera": camera,
                        "source_image": image_name,
                        "crop_image": display_crop_image,
                        "passed": passed,
                        "reason": reason,
                        "detect_result": detect_result,
                        "detect_roi": rule["detect_roi"],
                        "compare_roi": rule["compare_roi"],
                        "angle_info": angle_info,
                    })
                    
                except Exception as e:
                    logger.log(f"    [ERROR] {rule['roi_id']}: {str(e)}")
                    roi_results.append({
                        "roi_id": rule["roi_id"],
                        "camera": camera,
                        "pass": False,
                        "reason": f"ERROR: {str(e)}"
                    })
                    
                    # === THEM: Chi tiết cho CSV (error) ===
                    detail_results.append({
                        "Image": image_name,
                        "ROI_ID": rule["roi_id"],
                        "Camera": camera,
                        "Result": "NG",
                        "Reason": f"ERROR: {str(e)}"
                    })
        
        except Exception as e:
            logger.log(f"  [ERROR] Cannot process image: {str(e)}")
            continue
    
    # ┌─── SAVE BATCH RESULT ───────────────────────────────┐
    
    if not gui_roi_items:
        logger.log("[WARNING] No results to save (all failed)")
        return
    
    elapsed_time = time.time() - start_time
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"{cfg['paths']['output_dir']}/{timestamp}_batch_{batch_num:03d}_RESULT.jpg"
    
    # Calculate final status
    final_status = "NG"
    for roi_result in roi_results:
        if not roi_result["pass"]:
            final_status = "NG"
            break
        final_status = "OK"
    
    logger.log(f"\n[SAVE] Saving batch result...")
    save_batch_result(
        gui_roi_items,
        final_status,
        {
            "batch_num": batch_num,
            "product_code": product_code,
            "batch_time": elapsed_time,
        },
        output_path,
        total_ok=sum(1 for r in roi_results if r["pass"]),
        total_ng=sum(1 for r in roi_results if not r["pass"])
    )
    
    # === LUU FILE CSV CHI TIET (NEW) ===
    csv_path = save_result_csv(detail_results, cfg['paths']['output_dir'], product_code)
    logger.log(f"[SAVE] Result CSV: {csv_path}")

    ng_folder, ng_index_csv, ng_saved_count = save_ng_roi_images(
        gui_roi_items,
        cfg['paths']['output_dir'],
        product_code
    )
    logger.log(f"[SAVE] NG ROI folder: {ng_folder}")
    logger.log(f"[SAVE] NG ROI index: {ng_index_csv}")
    logger.log(f"[SAVE] NG ROI images: {ng_saved_count}")
    
    # ┌─── SUMMARY ─────────────────────────────────────────┐
    
    logger.log("\n" + "=" * 70)
    logger.log("BATCH RESULT SUMMARY")
    logger.log("=" * 70)
    logger.log(f"Final Status: {final_status}")
    logger.log(f"Total Processed: {len(image_files)}")
    logger.log(f"Total ROIs: {len(roi_results)}")
    logger.log(f"OK Count: {sum(1 for r in roi_results if r['pass'])}")
    logger.log(f"NG Count: {sum(1 for r in roi_results if not r['pass'])}")
    logger.log(f"Processing Time: {elapsed_time:.2f}s")
    logger.log(f"\n📊 Output Files:")
    logger.log(f"  Batch Image: {output_path}")
    logger.log(f"  Result CSV: {csv_path}")
    logger.log(f"  NG ROI Folder: {ng_folder}")
    logger.log("=" * 70)

# ============================================================================
# CLI Interface
# ============================================================================

def main():
    """Interactive CLI"""
    print("\n" + "=" * 70)
    print("BATCH DETECT FOLDER")
    print("=" * 70)
    
    # Nhập thư mục ảnh
    while True:
        input_folder = input("\n📁 Input folder (images path): ").strip()
        if input_folder and os.path.isdir(input_folder):
            break
        print(f"   ❌ Folder not found: {input_folder}")
    
    # Nhập product code
    product_code = input("📦 Product code (default: ABC123x): ").strip()
    if not product_code:
        product_code = "ABC123x"
    
    # Nhập camera
    camera = input("📷 Camera (default: CAM1): ").strip()
    if not camera:
        camera = "CAM1"
    
    # Nhập số lượng ảnh tối đa
    max_images_str = input("🔢 Max images (default: 50): ").strip()
    try:
        max_images = int(max_images_str) if max_images_str else 50
    except ValueError:
        print("   ⚠️ Invalid number, using default 50")
        max_images = 50
    
    print(f"\n✓ Configuration:")
    print(f"  Folder: {input_folder}")
    print(f"  Product: {product_code}")
    print(f"  Camera: {camera}")
    print(f"  Max Images: {max_images}")
    
    confirm = input("\n▶️ Start processing? (y/n): ").strip().lower()
    if confirm != "y":
        print("Cancelled.")
        return
    
    print("\n" + "=" * 70)
    
    batch_detect_folder(input_folder, product_code, camera, max_images=max_images)
    
    print("\n✅ Done! Check output files:")
    print("   📊 output/results/{timestamp}_batch_XXX_RESULT.jpg")
    print("   📋 output/results/{timestamp}_result_{product_code}.csv")
    print("=" * 70 + "\n")

if __name__ == "__main__":
    main()
