from modules.csv_loader import load_product_csv
from modules.camera_selector import get_used_cameras
from modules.image_watcher import ImageWatcher
from modules.roi_manager import prepare_roi_data
from modules.model_manager import get_model
from modules.detector import detect_object
from modules.comparator import compare_detection, compare_angle, is_center_inside_roi
from modules.result_manager import build_signal_payload, build_signal_summary
from modules.result_visualizer import save_batch_result
from modules.camera_config_loader import load_camera_config, get_camera_folder, print_camera_config_summary
from modules.result_gui import ResultGUI
from modules.com_output import COMOutput
from modules.com_input import COMProductReader
from modules.log_input import LogBarcodeReader
from modules.config_loader import load_config
from modules.csv_loader import build_product_signal_plan
from modules.wire_measure import detect_aruco_scale, measure_wire_from_crop, draw_measurement, draw_aruco_scale
import os
import time
import csv
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
import cv2

# ============================================================================
# Load Configuration
# ============================================================================
# Tai config tu file ngoai (config.yaml) thay vi hard-code
# - De doi product: chi can sua config.yaml, khong can sua code
# - De deploy nhieu may: moi may 1 file config rieng
# - Rollback: neu file loi, tu dong dung default config
cfg = load_config("config/config.yaml")

# Product Mode (manual, auto hoac log)
PRODUCT_MODE = cfg['product'].get('mode', 'manual')
PRODUCT_CODE_MANUAL = cfg['product'].get('code', 'ABC123x')
PRODUCT_DEFAULT_CODE = cfg['product'].get('default_code', 'ABC123x')

# Extract config values
OUTPUT_DIR = cfg['paths']['output_dir']
LOG_DIR = cfg['paths']['log_dir']

# COM Output Config
ENABLE_COM_OUTPUT = cfg['com_output']['enabled']
COM_PORT = cfg['com_output']['port']
COM_BAUDRATE = cfg['com_output']['baudrate']
COM_RETRY = cfg['com_output']['retry_count']

# Camera Config
CAMERA_CONFIG_CSV = cfg['camera']['config_csv']
CAMERA_CREATE_FOLDERS = cfg['camera']['create_folders']
CAMERA_POLL_INTERVAL = cfg['camera']['poll_interval']

# GUI Config
GUI_ENABLED = cfg['gui']['enabled']
GUI_WINDOW_NAME_TEMPLATE = cfg['gui']['window_name']  # Template, se format sau
GUI_MAX_HISTORY = cfg['gui']['max_history']

# Result CSV History
RESULT_SAVE_CSV_HISTORY = cfg.get('result', {}).get('save_csv_history', False)
RESULT_CSV_HISTORY_PATH = cfg.get('result', {}).get('csv_history_path', 'output/results/detect_history.csv')

# ============================================================================
# Product Code Reader
# ============================================================================
# Khoi tao Product Reader theo mode
product_reader = None
if PRODUCT_MODE == 'auto':
    product_reader = COMProductReader(
        port=cfg['product'].get('com_input_port', 'COM4'),
        baudrate=cfg['product'].get('com_input_baudrate', 9600),
        timeout=cfg['product'].get('com_input_timeout', 1.0),
        mode='latest',
        poll_interval=cfg['product'].get('com_input_poll_interval', 0.5)
    )
    print(f"[PRODUCT] Mode: AUTO (from COM {cfg['product'].get('com_input_port')})")
elif PRODUCT_MODE == 'log':
    log_root_dir = cfg['product'].get('log_root_dir', r'D:\anh')
    product_reader = LogBarcodeReader(
        base_dir=log_root_dir,
        log_filename=cfg['product'].get('log_filename', 'WH_AVIS_System.log'),
        date_folder_format=cfg['product'].get('log_date_folder_format', '%Y%m%d'),
        poll_interval=cfg['product'].get('log_poll_interval', 0.5),
    )
    print(f"[PRODUCT] Mode: LOG (from root folder {log_root_dir})")
else:
    print(f"[PRODUCT] Mode: MANUAL (code={PRODUCT_CODE_MANUAL})")

def get_current_product_code() -> str:
    """
    Lay product code hien tai (tu COM hoac config)
    
    Returns:
        Product code (str)
    """
    if product_reader:
        # Che do AUTO: doc tu COM
        code = product_reader.get_current()
        if code:
            return code
        else:
            # Chua nhan duoc → dung default
            return PRODUCT_DEFAULT_CODE
    else:
        # Che do MANUAL: dung code co dinh
        return PRODUCT_CODE_MANUAL

# ============================================================================

# Tao thu muc output neu chua ton tai
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
if RESULT_SAVE_CSV_HISTORY:
    os.makedirs(os.path.dirname(RESULT_CSV_HISTORY_PATH), exist_ok=True)


def log_message(message: str) -> None:
    """In va luu log"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_msg = f"[{timestamp}] {message}"
    print(log_msg)
    
    # Luu vao file log
    log_file = os.path.join(LOG_DIR, "processing.log")
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(log_msg + "\n")


def append_history_csv(csv_path: str, batch_info: dict, final_status: str, roi_results: list) -> None:
    file_exists = os.path.exists(csv_path)

    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([
                "timestamp",
                "batch_num",
                "product_code",
                "batch_time",
                "final_status",
                "camera",
                "roi_id",
                "pass",
                "reason"
            ])

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        for result in roi_results:
            writer.writerow([
                timestamp,
                batch_info.get("batch_num"),
                batch_info.get("product_code"),
                f"{batch_info.get('batch_time', 0):.3f}",
                final_status,
                result.get("camera"),
                result.get("roi_id"),
                "OK" if result.get("pass") else "NG",
                result.get("reason", "")
            ])


def log_signal_plan(roi_rules: list, csv_path: str) -> None:
    """Log the ordered model/class signal mapping for the loaded CSV."""
    signal_plan = build_product_signal_plan(roi_rules)
    log_message(f"[CSV] Loaded product CSV: {csv_path}")
    log_message(f"[CSV] Signal order: {len(signal_plan)} com_index groups")

    for entry in signal_plan:
        roi_ids = ", ".join(entry["roi_ids"])
        log_message(
            f"[CSV] {entry['ok_signal']} / {entry['ng_signal']} -> "
             f"model={entry['model_name']}, class_id={entry['class_id']}, class_name={entry['class_name']} "
            f"| ROIs: {roi_ids}"
        )


def poll_cameras_once(watchers: dict, used_cameras: list, images: dict, temp_paths: dict) -> tuple:
    """
    Kiem tra 1 luot tat ca camera (NON-BLOCKING).
    - Khong cho, chi kiem tra co anh moi khong
    - Neu co -> doc anh, them vao dict
    - Neu khong -> bo qua, tra ve ngay

    Args:
        watchers: {"CAM1": ImageWatcher, ...}
        used_cameras: Danh sach camera can cho
        images: Dict anh da thu thap duoc (se duoc cap nhat)
        temp_paths: Dict luu duong dan temp file (de cleanup sau)

    Returns:
        tuple: (images, temp_paths) da duoc cap nhat
    """
    cameras_to_check = [cam for cam in used_cameras if cam not in images]
    if not cameras_to_check:
        return images, temp_paths

    max_workers = min(len(cameras_to_check), 8)

    def poll_single_camera(cam: str):
        new_images = watchers[cam].get_new_images()
        if not new_images:
            return None

        image_path = new_images[0]['temp_path']
        image = cv2.imread(image_path)
        if image is None:
            return None

        return cam, image, image_path

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(poll_single_camera, cam) for cam in cameras_to_check]
        for future in futures:
            result = future.result()
            if not result:
                continue
            cam, image, image_path = result
            images[cam] = image
            temp_paths[cam] = image_path
            log_message(f"[GOT] Camera {cam} - image received {image.shape}")

    return images, temp_paths
    
def run_detection_for_cam(cam: str, image, roi_rules: list, batch_num: int) -> tuple:
    cam_roi_results = []
    cam_gui_items = []

    log_message(f"[BATCH {batch_num}] Processing {cam} - {image.shape}")
    cam_rules = [r for r in roi_rules if r["camera"] == cam]

    # Step 0: Detect ArUco scale once per camera image (for measure_width rules)
    aruco_info = None
    measure_rules = [r for r in cam_rules if r.get("check_mode") == "measure_width"]
    for mr in measure_rules:
        marker_size = mr.get("aruco_marker_size_mm")
        if marker_size is not None and marker_size > 0:
            aruco_info = detect_aruco_scale(
                image=image,
                marker_size_mm=marker_size,
                dict_name=mr.get("aruco_dict", "DICT_4X4_50"),
                marker_id=mr.get("aruco_id"),
            )
            if aruco_info is not None:
                log_message(f"  [ARUCO] Scale: {aruco_info['px_per_mm']:.4f} px/mm, ids={aruco_info['detected_ids']}")
                break
            else:
                msg = "  [ARUCO] Marker not detected."
                if mr.get("require_aruco"):
                    log_message(f"{msg} ABORTING (require_aruco=True)")
                    break
                log_message(f"{msg} mm conversion disabled.")
                aruco_info = None
            break

    for rule in cam_rules:
        try:
            roi_data = prepare_roi_data(image, rule)

            detect_roi = rule["detect_roi"]
            roi_offset = (detect_roi[0], detect_roi[1])

            check_mode = rule.get("check_mode", "compare")
            angle_info = {}
            detect_result = None

            if check_mode == "measure_width":
                seg_model = get_model(rule["model_name"])
                measurement_result = measure_wire_from_crop(
                    seg_model,
                    roi_data["detect_image"],
                    rule,
                    aruco_info,
                )

                if not measurement_result["found"]:
                    passed = False
                    reason = "NOT_FOUND"
                elif measurement_result["measurement"] is None:
                    passed = False
                    reason = "MEASUREMENT_FAILED"
                else:
                    median_mm = measurement_result["median_width_mm"]
                    if median_mm is not None:
                        min_mm = rule.get("width_min_mm")
                        max_mm = rule.get("width_max_mm")
                        if min_mm is not None and median_mm < min_mm:
                            passed = False
                            reason = f"WIDTH_NG({median_mm:.3f}mm<{min_mm:.3f}mm)"
                        elif max_mm is not None and median_mm > max_mm:
                            passed = False
                            reason = f"WIDTH_NG({median_mm:.3f}mm>{max_mm:.3f}mm)"
                        else:
                            passed = True
                            reason = f"WIDTH_OK({median_mm:.3f}mm)"
                    else:
                        median_px = measurement_result["measurement"]["median_px"]
                        passed = True
                        reason = f"WIDTH_OK({median_px:.2f}px)"
            else:
                model = get_model(rule["model_name"])

                detect_result = detect_object(
                    model,
                    roi_data["detect_image"],
                    rule["class_id"],
                    rule["confidence"],
                    roi_offset=roi_offset
                )

            if check_mode == "count":
                detections = detect_result.get("detections", [])

                # DEBUG: In ra toan bo detections
                log_message(f"  [COUNT DEBUG] {rule['roi_id']}: Total detections = {len(detections)}")
                for idx, det in enumerate(detections):
                    bbox = det["bbox"]
                    center_x = (bbox[0] + bbox[2]) / 2
                    center_y = (bbox[1] + bbox[3]) / 2
                    log_message(
                        f"    Detection {idx}: bbox={bbox}, center=({center_x:.1f}, {center_y:.1f}), conf={det['confidence']:.2f}"
                    )

                log_message(f"  [COUNT DEBUG] compare_roi={rule['compare_roi']}")

                # FIXED: Chi dem detections co trung tam nam trong compare_roi
                valid_detections = [
                    det for det in detections
                    if is_center_inside_roi(det["bbox"], rule["compare_roi"])
                ]
                detected_count = len(valid_detections)

                log_message(f"  [COUNT DEBUG] Valid detections in compare_roi = {detected_count}")

                min_count = rule.get("min_count")
                max_count = rule.get("max_count")
                required_min = min_count if min_count is not None else 1

                log_message(f"  [COUNT DEBUG] min_count={min_count}, max_count={max_count}, required_min={required_min}")

                if max_count is None:
                    passed = detected_count >= required_min
                    reason = (
                        f"COUNT_OK({detected_count}>={required_min})"
                        if passed else f"COUNT_NG({detected_count}<{required_min})"
                    )
                else:
                    passed = required_min <= detected_count <= max_count
                    reason = (
                        f"COUNT_OK({detected_count}/{required_min}-{max_count})"
                        if passed else f"COUNT_NG({detected_count}/{required_min}-{max_count})"
                    )

                log_message(f"  [COUNT DEBUG] Logic: {required_min} <= {detected_count} <= {max_count} = {passed}")
            elif check_mode == "forbidden":
                # Vung cam - KHONG DUOC CO object
                detections = detect_result.get("detections", [])
                # FIXED: Chi dem detections co trung tam nam trong compare_roi
                valid_detections = [
                    det for det in detections
                    if is_center_inside_roi(det["bbox"], rule["compare_roi"])
                ]
                detected_count = len(valid_detections)

                if detected_count == 0:
                    passed = True
                    reason = "FORBIDDEN_OK(no_object)"
                else:
                    passed = False
                    reason = f"FORBIDDEN_NG({detected_count}_found)"
            elif check_mode == "compare":
                passed, reason = compare_detection(
                    detect_result,
                    rule["compare_roi"]
                )

                # === Keypoint angle check (neu co config) ===
                if rule.get("keypoint_idx_1") is not None:
                    if detect_result.get("found"):
                        # Tinh goc (luon tinh de hien thi debug)
                        angle_passed, angle_reason, angle_info = compare_angle(
                            detect_result,
                            rule["keypoint_idx_1"],
                            rule["keypoint_idx_2"],
                            rule["expected_angle"],
                            rule["angle_tolerance"]
                        )
                        # Goc sai -> NG (chi khi vi tri da OK)
                        if passed and not angle_passed:
                            passed = False
                            reason = angle_reason

            status_str = "OK" if passed else "NG"
            log_message(f"  {rule['roi_id']}: {status_str} ({reason})")

            cam_roi_results.append({
                "roi_id": rule["roi_id"],
                "camera": cam,
                "model_name": rule["model_name"],
                "class_id": rule["class_id"],
                "class_name": rule["class_name"],
                "com_index": rule.get("com_index"),
                "pass": passed,
                "reason": reason
            })

            # === THEM: Gom data cho GUI ===
            gui_item = {
                "roi_id": rule["roi_id"],
                "camera": cam,
                "crop_image": roi_data["detect_image"],
                "passed": passed,
                "reason": reason,
                "detect_result": detect_result if detect_result is not None else {"found": False, "detections": []},
                "detect_roi": rule["detect_roi"],
                "compare_roi": rule["compare_roi"],
                "angle_info": angle_info,
                "check_mode": check_mode,
            }

            if check_mode == "measure_width":
                gui_item["measurement_result"] = measurement_result

            cam_gui_items.append(gui_item)

        except Exception as e:
            log_message(f"  [ERROR] {rule['roi_id']}: {str(e)}")
            cam_roi_results.append({
                "roi_id": rule["roi_id"],
                "camera": cam,
                "model_name": rule.get("model_name", ""),
                "class_id": rule.get("class_id", 0),
                "class_name": rule.get("class_name", ""),
                "com_index": rule.get("com_index"),
                "pass": False,
                "reason": f"ERROR: {str(e)}"
            })

    return cam_roi_results, cam_gui_items


def merge_cam_results(items_by_cam: dict, cameras: list) -> list:
    merged = []
    for cam in cameras:
        merged.extend(items_by_cam.get(cam, []))
    return merged


def merge_updates_for_ng_rois(previous_items: list, updated_items: list, ng_roi_ids: set) -> list:
    """
    Chi cap nhat lai cac ROI dang NG, giu nguyen ROI OK truoc do.
    """
    if not previous_items:
        return updated_items

    updated_by_roi = {
        item.get("roi_id"): item
        for item in updated_items
        if item.get("roi_id") in ng_roi_ids
    }

    merged_items = []
    existing_roi_ids = set()

    for item in previous_items:
        roi_id = item.get("roi_id")
        existing_roi_ids.add(roi_id)
        if roi_id in updated_by_roi:
            merged_items.append(updated_by_roi[roi_id])
        else:
            merged_items.append(item)

    # Truong hop edge: co ROI moi trong ket qua update nhung chua co trong previous
    for roi_id, item in updated_by_roi.items():
        if roi_id not in existing_roi_ids:
            merged_items.append(item)

    return merged_items


def get_ng_cameras(results_by_cam: dict) -> list:
    ng_cameras = []
    for cam, items in results_by_cam.items():
        for result in items:
            if not result.get("pass", True):
                ng_cameras.append(cam)
                break
    return ng_cameras


def emit_batch_result(roi_results: list, gui_roi_items: list, batch_num: int,
                      product_code: str, gui: ResultGUI, com_output: COMOutput,
                      batch_start_time: float) -> tuple:
    # Tinh ket qua batch
    signal_summary = build_signal_summary(roi_results)
    final_status = "OK" if signal_summary and all(item["passed"] for item in signal_summary) else "NG"
    signal_payload = build_signal_payload(roi_results)
    batch_time = time.time() - batch_start_time

    # === LUU BATCH RESULT (Thay the per-ROI save) ===
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    batch_output_path = f"{OUTPUT_DIR}/{timestamp}_batch_{batch_num:03d}_RESULT.jpg"
    save_batch_result(
        gui_roi_items,
        final_status,
        {
            "batch_num": batch_num,
            "product_code": product_code,
            "batch_time": batch_time,
        },
        batch_output_path,
        total_ok=gui.total_ok,
        total_ng=gui.total_ng
    )

    # === THEM: Gui tin hieu COM ===
    com_output.send_result(signal_payload)

    # === THEM: Update GUI ===
    gui.update(gui_roi_items, final_status, {
        "batch_num": batch_num,
        "product_code": product_code,
        "batch_time": batch_time,
    })

    # Print ket qua (GIU NGUYEN)
    print("\n" + "="*70)
    print(f"BATCH #{batch_num} - RESULT: {final_status} | COM: {signal_payload or '-'} | Time: {batch_time:.2f}s")
    print("="*70)
    print("  Class summary:")
    for item in signal_summary:
        print(
            f"    [{item['signal_code']}] model={item['model_name']} "
            f"class_id={item['class_id']} class_name={item['class_name']} | "
            f"ROIs={item['roi_count']}"
        )
    print("  ROI details:")
    for result in roi_results:
        status_icon = "[OK]" if result["pass"] else "[NG]"
        camera = result.get("camera", "?")
        roi_id = result.get("roi_id", "?")
        reason = result.get("reason", "")
        model_name = result.get("model_name", "?")
        class_id = result.get("class_id", "?")
        class_name = result.get("class_name", "?")
        print(f"    {status_icon} {roi_id} ({camera}) model={model_name} class_id={class_id} class_name={class_name}: {reason}")
    print("="*70)
    print(f"Batch result saved to: {os.path.abspath(batch_output_path)}\n")

    log_message(f"[BATCH {batch_num}] COMPLETED - RESULT: {final_status} - COM: {signal_payload or '-'}")
    log_message("")

    return final_status, batch_time, batch_output_path

def try_load_product_config(product_code: str):
    """
    Cố gắng load cấu hình product CSV.
    
    Args:
        product_code: Mã hàng
        
    Returns:
        tuple: (success, roi_rules, used_cameras) hoặc (False, None, None)
    """
    try:
        csv_path = cfg['product']['csv_path'].format(code=product_code)
        roi_rules = load_product_csv(csv_path)
        used_cameras = get_used_cameras(roi_rules)
        log_signal_plan(roi_rules, csv_path)
        return True, roi_rules, used_cameras
    except FileNotFoundError as e:
        log_message(f"[WARNING] Mã hàng '{product_code}' không được tìm thấy: {e}")
        print(f"\n⚠️  CẢNH BÁO: Mã hàng '{product_code}' không tồn tại!")
        print(f"    Vui lòng kiểm tra lại mã hàng.\n")
        return False, None, None
    except Exception as e:
        log_message(f"[ERROR] Lỗi khi load product CSV: {e}")
        print(f"\n❌ LỖI: {e}\n")
        return False, None, None

def main():
    """
    Chỉ xử lý khi tất cả camera có ảnh
    - Khởi tạo ImageWatcher cho mỗi camera
    - Chờ tất cả camera
    - Xử lý 1 batch
    - Print kết quả
    - Lặp lại chờ batch tiếp
    - Hỗ trợ đổi product code động (từ COM hoac config)
    """
    # Bien theo doi product code hien tai
    current_product_code = None
    last_product_event_id = None
    roi_rules = None
    used_cameras = None
    
    try:
        # Start COM Product Reader (neu mode=auto)
        if product_reader:
            product_reader.start()
        
        # Lay product code ban dau
        current_product_code = get_current_product_code()
        if PRODUCT_MODE == 'log' and product_reader:
            last_product_event_id = product_reader.get_event_id()
        
        # Load config
        log_message("="*70)
        log_message("SYSTEM STARTED - Waiting for camera images")
        log_message(f"Product mode: {PRODUCT_MODE}")
        log_message(f"Product code: {current_product_code}")
        log_message("="*70)
        
        # Load product CSV - with retry logic
        retry_count = 0
        max_retries = 3
        while True:
            success, roi_rules, used_cameras = try_load_product_config(current_product_code)
            if success:
                break
            retry_count += 1
            if retry_count >= max_retries:
                log_message(f"[WARNING] Không thể load product '{current_product_code}' sau {max_retries} lần thử")
                print(f"⏳  Đợi mã hàng hợp lệ...\n")
                # Wait for a new product code (wait a bit before retrying)
                for _ in range(50):  # Wait ~2.5 seconds (50 * 50ms)
                    time.sleep(0.05)
                    if product_reader:
                        new_code = product_reader.get_current()
                        if new_code and new_code != current_product_code:
                            log_message(f"[RETRY] Product code changed: {current_product_code} → {new_code}")
                            current_product_code = new_code
                            break
                retry_count = 0
            else:   
                time.sleep(0.5)  # Wait before retry
        
        # Load camera config
        log_message("[LOADING] Camera configuration...")
        camera_config = load_camera_config(
            csv_file=CAMERA_CONFIG_CSV,
            create_folders=CAMERA_CREATE_FOLDERS,
            verbose=False
        )
        print_camera_config_summary(camera_config)
        
        # Khởi tạo ImageWatcher cho mỗi camera
        watchers = {}
        for cam in used_cameras:
            # Lấy folder từ config
            input_folder = get_camera_folder(cam, camera_config, "input")
            temp_folder = get_camera_folder(cam, camera_config, "temp")
            
            # Kiểm tra camera có được cấu hình không
            if input_folder is None or temp_folder is None:
                log_message(f"[ERROR] Camera {cam} is not configured in camera_config.csv!")
                continue
            
            # Tạo folder nếu chưa có
            os.makedirs(input_folder, exist_ok=True)
            os.makedirs(temp_folder, exist_ok=True)
            
            # Khởi tạo watcher
            watchers[cam] = ImageWatcher(input_folder, temp_folder, poll_interval=CAMERA_POLL_INTERVAL)
            log_message(f"[INIT] ImageWatcher for {cam}: {input_folder}")
        
        batch_num = 0
        images = {}  # Thu thap anh dan dan (non-blocking)
        temp_paths = {}  # Track temp files de cleanup sau khi xu ly
        last_images_by_cam = {}
        last_roi_results_by_cam = {}
        last_gui_items_by_cam = {}
        
        # Khoi tao GUI (dynamic window name)
        gui_window_name = GUI_WINDOW_NAME_TEMPLATE.format(product_code=current_product_code)
        gui = ResultGUI(window_name=gui_window_name, max_history=GUI_MAX_HISTORY)
        gui.start()
        
        # Khoi tao COM Output
        com_output = COMOutput(
            port=COM_PORT,
            baudrate=COM_BAUDRATE,
            enabled=ENABLE_COM_OUTPUT,
            retry_count=COM_RETRY
        )
        
        log_message(f"[WAITING] Waiting for images from: {used_cameras}")
        
        # === VONG LAP CHINH (Non-blocking) ===
        # Moi vong: kiem tra product change → poll anh → refresh GUI → xu ly neu du anh
        while True:
            
            # --- Buoc 0: Kiem tra product code co thay doi khong ---
            new_product_code = get_current_product_code()
            is_new_event = False
            if PRODUCT_MODE == 'log' and product_reader:
                new_event_id = product_reader.get_event_id()
                if last_product_event_id is None:
                    last_product_event_id = new_event_id
                elif new_event_id != last_product_event_id:
                    is_new_event = True
                    last_product_event_id = new_event_id

            if new_product_code != current_product_code or is_new_event:
                log_message("="*70)
                if new_product_code != current_product_code:
                    log_message(f"[PRODUCT CHANGE] {current_product_code} → {new_product_code}")
                else:
                    log_message(f"[PRODUCT SCAN] {current_product_code} (repeat scan)")
                log_message("="*70)
                
                # Try to load new product configuration
                success, new_roi_rules, new_used_cameras = try_load_product_config(new_product_code)
                
                if success:
                    # Update to new product
                    current_product_code = new_product_code
                    roi_rules = new_roi_rules
                    used_cameras = new_used_cameras
                    
                    log_message(f"[RELOAD] Cameras: {used_cameras}")

                    # Khoi tao ImageWatcher cho camera moi (neu chua co)
                    for cam in used_cameras:
                        if cam in watchers:
                            continue

                        input_folder = get_camera_folder(cam, camera_config, "input")
                        temp_folder = get_camera_folder(cam, camera_config, "temp")

                        if input_folder is None or temp_folder is None:
                            log_message(f"[ERROR] Camera {cam} is not configured in camera_config.csv!")
                            continue

                        os.makedirs(input_folder, exist_ok=True)
                        os.makedirs(temp_folder, exist_ok=True)

                        watchers[cam] = ImageWatcher(input_folder, temp_folder, poll_interval=CAMERA_POLL_INTERVAL)
                        log_message(f"[INIT] ImageWatcher for {cam}: {input_folder}")
                    
                    # Reset images (cho batch moi)
                    images = {}
                    temp_paths = {}

                    # Reset cache re-detect khi doi product
                    last_images_by_cam = {}
                    last_roi_results_by_cam = {}
                    last_gui_items_by_cam = {}

                    # Reset GUI ve trang thai ban dau
                    gui.reset_display()

                    # Update GUI window title
                    gui.window_name = GUI_WINDOW_NAME_TEMPLATE.format(product_code=current_product_code)
                    
                else:
                    # Product code không hợp lệ
                    log_message(f"[WARNING] Product code '{new_product_code}' không hợp lệ")
                    log_message(f"[WAITING] Giữ nguyên product: {current_product_code}")
                    print(f"\n⏳  Đợi mã hàng hợp lệ (hiện tại: {current_product_code})...\n")
            
            # --- Buoc 1: Poll anh (KHONG block) ---
            images, temp_paths = poll_cameras_once(watchers, used_cameras, images, temp_paths)
            
            # --- Buoc 2: Refresh GUI + doc phim (BAT BUOC moi vong) ---
            action = gui.show(wait_time=30)
            if action == 'quit':
                log_message("[GUI] User pressed Quit")
                break
            
            # --- Buoc 3: Chua du anh → quay lai buoc 1 ---
            if len(images) < len(used_cameras):
                continue
            
            # --- Buoc 4: DU ANH → Xu ly batch ---
            # (Logic xu ly GIU NGUYEN 100%)
            batch_num += 1
            log_message(f"[READY] All cameras ready. Processing batch #{batch_num}...")
            
            roi_results_by_cam = {}
            gui_items_by_cam = {}
            batch_start_time = time.time()

            # Xu ly tung camera
            for cam in used_cameras:
                image = images[cam]
                cam_roi_results, cam_gui_items = run_detection_for_cam(
                    cam, image, roi_rules, batch_num
                )
                roi_results_by_cam[cam] = cam_roi_results
                gui_items_by_cam[cam] = cam_gui_items

            roi_results = merge_cam_results(roi_results_by_cam, used_cameras)
            gui_roi_items = merge_cam_results(gui_items_by_cam, used_cameras)

            final_status, batch_time, batch_output_path = emit_batch_result(
                roi_results,
                gui_roi_items,
                batch_num,
                current_product_code,
                gui,
                com_output,
                batch_start_time
            )

            if RESULT_SAVE_CSV_HISTORY:
                append_history_csv(
                    RESULT_CSV_HISTORY_PATH,
                    {
                        "batch_num": batch_num,
                        "product_code": current_product_code,
                        "batch_time": batch_time,
                    },
                    final_status,
                    roi_results
                )
            
            # === Cleanup temp files ===
            for cam, temp_path in temp_paths.items():
                try:
                    watchers[cam].cleanup_temp_file(temp_path)
                except Exception as e:
                    log_message(f"[CLEANUP] Error cleaning {cam} temp file: {e}")

            # Cache ket qua batch de re-detect
            last_images_by_cam = dict(images)
            last_roi_results_by_cam = dict(roi_results_by_cam)
            last_gui_items_by_cam = dict(gui_items_by_cam)

            # Auto re-detect (retry den khi OK cho moi CAM NG)
            # Auto re-detect (retry den khi OK, xu ly ngay khi CAM NG co anh moi)
            if final_status == "NG":
                ng_cameras = get_ng_cameras(last_roi_results_by_cam)
                if ng_cameras:
                    log_message(f"[REDETECT] NG cameras: {ng_cameras}")

                stop_requested = False
                abort_redetect = False

                pending_ng = set(ng_cameras)
                re_images = {}
                re_temp_paths = {}

                while pending_ng and not stop_requested and not abort_redetect:
                    log_message(f"[REDETECT] Waiting for new images from {sorted(pending_ng)}")

                    # Kiem tra scan event moi (repeat scan cung ma) trong luc re-detect
                    if PRODUCT_MODE == 'log' and product_reader:
                        new_event_id = product_reader.get_event_id()
                        if last_product_event_id is not None and new_event_id != last_product_event_id:
                            log_message(f"[REDETECT] New scan event (same product: {current_product_code}), aborting redetect")
                            abort_redetect = True
                            break

                    # Kiem tra product code thay doi trong luc re-detect
                    new_product_code = get_current_product_code()
                    if new_product_code != current_product_code:
                        log_message(f"[REDETECT] Product changed: {current_product_code} -> {new_product_code}")
                        abort_redetect = True
                        break

                    re_images, re_temp_paths = poll_cameras_once(
                        watchers, list(pending_ng), re_images, re_temp_paths
                    )

                    action = gui.show(wait_time=30)
                    if action == 'quit':
                        log_message("[GUI] User pressed Quit")
                        stop_requested = True
                        break

                    ready_cams = [cam for cam in list(pending_ng) if cam in re_images]
                    if not ready_cams:
                        continue

                    for cam in ready_cams:
                        batch_num += 1
                        batch_start_time = time.time()

                        image = re_images.pop(cam)
                        previous_cam_results = last_roi_results_by_cam.get(cam, [])
                        ng_roi_ids = {
                            result.get("roi_id")
                            for result in previous_cam_results
                            if not result.get("pass", True)
                        }

                        if not ng_roi_ids:
                            pending_ng.discard(cam)
                            log_message(f"[REDETECT] {cam} has no NG ROI left, skip update")
                            continue

                        cam_roi_results, cam_gui_items = run_detection_for_cam(
                            cam, image, roi_rules, batch_num
                        )
                        last_images_by_cam[cam] = image
                        last_roi_results_by_cam[cam] = merge_updates_for_ng_rois(
                            previous_cam_results,
                            cam_roi_results,
                            ng_roi_ids
                        )
                        last_gui_items_by_cam[cam] = merge_updates_for_ng_rois(
                            last_gui_items_by_cam.get(cam, []),
                            cam_gui_items,
                            ng_roi_ids
                        )

                        roi_results = merge_cam_results(last_roi_results_by_cam, used_cameras)
                        gui_roi_items = merge_cam_results(last_gui_items_by_cam, used_cameras)

                        final_status, batch_time, batch_output_path = emit_batch_result(
                            roi_results,
                            gui_roi_items,
                            batch_num,
                            current_product_code,
                            gui,
                            com_output,
                            batch_start_time
                        )

                        if RESULT_SAVE_CSV_HISTORY:
                            append_history_csv(
                                RESULT_CSV_HISTORY_PATH,
                                {
                                    "batch_num": batch_num,
                                    "product_code": current_product_code,
                                    "batch_time": batch_time,
                                },
                                final_status,
                                roi_results
                            )

                        temp_path = re_temp_paths.pop(cam, None)
                        if temp_path:
                            try:
                                watchers[cam].cleanup_temp_file(temp_path)
                            except Exception as e:
                                log_message(f"[CLEANUP] Error cleaning {cam} temp file: {e}")

                        merged_cam_results = last_roi_results_by_cam.get(cam, [])
                        cam_passed = all(result.get("pass", False) for result in merged_cam_results)
                        if cam_passed:
                            pending_ng.discard(cam)
                            log_message(f"[REDETECT] {cam} is OK, stop retry")
                        else:
                            log_message(f"[REDETECT] {cam} still NG, retry")

                if stop_requested:
                    break

            # Neu tat ca OK, dung cho den khi co lan quet tiep theo (LOG mode)
            if final_status == "OK" and PRODUCT_MODE == 'log' and product_reader:
                log_message("[WAIT] OK - waiting for next product scan...")
                stop_requested = False
                while True:
                    action = gui.show()
                    if action == 'quit':
                        log_message("[GUI] User pressed Quit")
                        stop_requested = True
                        break

                    new_event_id = product_reader.get_event_id()
                    if last_product_event_id is None:
                        last_product_event_id = new_event_id
                    elif new_event_id != last_product_event_id:
                        log_message("[WAIT] New product scan detected")
                        break

                if stop_requested:
                    break
            
            # Reset cho batch tiep theo
            images = {}
            temp_paths = {}
            log_message(f"[WAITING] Waiting for images from: {used_cameras}")
    
    except FileNotFoundError as e:
        log_message(f"[FATAL] {e}")
        print(f"\n[ERROR] File not found: {e}\n")
    except KeyboardInterrupt:
        log_message("="*70)
        log_message(f"SYSTEM STOPPED (User interrupted after {batch_num} batches)")
        log_message("="*70)
        print("\n[INFO] System stopped by user\n")
    except Exception as e:
        log_message(f"[FATAL] {str(e)}")
        print(f"\n[ERROR] {str(e)}\n")
    finally:
        # Dong GUI, COM output va COM input an toan
        try:
            gui.close()
        except Exception:
            pass
        try:
            com_output.close()
        except Exception:
            pass
        try:
            if product_reader:
                product_reader.stop()
        except Exception:
            pass


if __name__ == "__main__":
    main()