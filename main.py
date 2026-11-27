from modules.csv_loader import load_product_csv
from modules.camera_selector import get_used_cameras
from modules.image_watcher import ImageWatcher
from modules.roi_manager import prepare_roi_data
from modules.model_manager import get_model
from modules.detector import detect_object
from modules.comparator import (is_center_inside_roi, compare_count_and_angle)
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
from modules.ai_worker import decode_packets, detect_images
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
ACQUISITION_CFG = cfg.get('acquisition', {})
# Periodic capture statistics: default 60 seconds; 0 disables only this log.
CAPTURE_STATS_INTERVAL = float(ACQUISITION_CFG.get('stats_interval_seconds', 60.0))
# Separate from legacy camera.poll_interval=0.5. Tune against measured file lifetime.
CAMERA_POLL_INTERVAL = ACQUISITION_CFG.get('poll_interval', 0.02)
ACQUISITION_OPTIONS = {
    'queue_max_images': ACQUISITION_CFG.get('queue_max_images', 32),
    'queue_max_bytes': int(ACQUISITION_CFG.get('queue_max_mb', 128) * 1024 * 1024),
    'max_image_bytes': int(ACQUISITION_CFG.get('max_image_mb', 32) * 1024 * 1024),
    'retry_delays': ACQUISITION_CFG.get('retry_delays_ms', [5, 10, 20, 40]),
    'retry_timeout': ACQUISITION_CFG.get('retry_timeout_ms', 250) / 1000.0,
}
ACQUISITION_OPTIONS['retry_delays'] = [
    delay / 1000.0 for delay in ACQUISITION_OPTIONS['retry_delays']
]

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



def cleanup_partial_batch_temp_files(watchers: dict, temp_paths: dict) -> None:
    """Remove temporary copies belonging to a batch that is being discarded."""
    for cam, temp_path in temp_paths.items():
        watcher = watchers.get(cam)
        if watcher is None:
            continue

        try:
            watcher.cleanup_temp_file(temp_path)
        except Exception as e:
            log_message(f"[CLEANUP] Error cleaning discarded temp file for {cam}: {e}")


def reset_image_boundary_for_new_scan(
    watchers: dict,
    used_cameras: list,
    temp_paths: dict,
) -> tuple:
    """
    Start a fresh image batch after an accepted product scan.

    Images already present in each camera folder are treated as belonging to
    the previous batch. Only files that appear or are overwritten after this
    reset can be collected for the new product.
    """
    cleanup_partial_batch_temp_files(watchers, temp_paths)

    for cam in used_cameras:
        watcher = watchers.get(cam)
        if watcher is None:
            log_message(f"[RESET] Camera {cam} has no ImageWatcher; skipped image boundary")
            continue

        watcher.reset()
        log_message(f"[RESET] Camera {cam}: ignored images currently in folder")

    return {}, {}


def poll_cameras_once(watchers: dict, used_cameras: list,
                      images: dict, temp_paths: dict) -> tuple:
    """Compatibility API: dequeue RAM packets only; no scan/read/decode/wait.

    Values in images are ImagePacket, not OpenCV arrays. main() uses its
    asynchronous decode job before detection. temp_paths remains empty.
    """
    for cam in used_cameras:
        if cam not in images:
            packet = watchers[cam].get_packet_nowait()
            if packet is not None:
                images[cam] = packet
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
            count_angle_details = {}
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

            if check_mode in ("count", "compare"):
                passed, reason, count_angle_details = compare_count_and_angle(
                    detect_result,
                    rule["compare_roi"],
                    min_count=rule.get("min_count"),
                    max_count=rule.get("max_count"),
                    keypoint_idx_1=rule.get("keypoint_idx_1"),
                    keypoint_idx_2=rule.get("keypoint_idx_2"),
                    expected_angle=rule.get("expected_angle"),
                    angle_tolerance=rule.get("angle_tolerance"),
                )
                log_message(
                    f"  [{check_mode.upper()}] {rule['roi_id']}: "
                    f"in_compare_roi={count_angle_details['detected_count']}, "
                    f"min={rule.get('min_count')}, max={rule.get('max_count')}"
                )
                for angle_result in count_angle_details["angle_results"]:
                    log_message(
                        f"    [ANGLE] obj{angle_result['object_index']}: "
                        f"{angle_result['reason']}"
                    )
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

            if check_mode in ("count", "compare"):
                cam_roi_results[-1].update(count_angle_details)

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

            if check_mode in ("count", "compare"):
                gui_item.update(count_angle_details)

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
    """Main-thread GUI/COM coordinator; capture and AI never run on this thread."""
    exit_code = 0
    watchers = {}
    gui = None
    com_output = None
    executor = None
    future = None
    job = None
    job_items = {}
    epoch = 0
    batch_num = 0
    current_product_code = None
    last_product_event_id = None
    images = {}
    last_images = {}
    last_results = {}
    last_gui_items = {}
    pending_ng = set()
    mode = "collect"
    last_stats_at = time.monotonic()

    def service_capture():
        for cam, watcher in watchers.items():
            for code, message in watcher.drain_events():
                log_message(f"[CAPTURE {cam}] {code}: {message}")
            watcher.check_health()  # fail-stop on overload; never send artificial OK

    def configure_watchers(cameras):
        # Same directory cannot unambiguously supply two cameras in this project.
        if not cameras:
            raise ValueError("Product configuration has no active cameras")
        folders = {}
        for cam in cameras:
            folder = get_camera_folder(cam, camera_config, "input")
            if not folder:
                raise ValueError(f"Camera {cam} missing input folder")
            canonical = os.path.normcase(os.path.realpath(folder))
            # Never save our outputs/logs inside a producer-owned input tree.
            for label, path in (("output_dir", globals().get("OUTPUT_DIR")),
                                ("log_dir", globals().get("LOG_DIR"))):
                if path:
                    target = os.path.normcase(os.path.realpath(path))
                    try:
                        inside = os.path.commonpath([canonical, target]) == canonical
                    except ValueError:
                        inside = False  # different Windows drives
                    if inside:
                        raise ValueError(f"{label} must be outside input folder: {folder}")
            if canonical in folders:
                raise ValueError(
                    f"{cam} and {folders[canonical]} share one input folder. "
                    "Explicit filename-to-camera routing is required.")
            folders[canonical] = cam
        for cam in list(watchers):
            if cam not in cameras:
                watchers[cam].stop()
                watchers[cam].reset()
                for code, message in watchers[cam].drain_events():
                    log_message(f"[CAPTURE REMOVED {cam}] {code}: {message}")
                log_message(f"[CAPTURE REMOVED FINAL {cam}] {watchers[cam].stats()}")
                del watchers[cam]
        for cam in cameras:
            if cam not in watchers:
                input_folder = get_camera_folder(cam, camera_config, "input")
                # temp_folder is not required by the RAM reader.
                watcher = ImageWatcher(
                    input_folder,
                    poll_interval=CAMERA_POLL_INTERVAL,
                    **ACQUISITION_OPTIONS)
                watchers[cam] = watcher
                watcher.start()
                log_message(f"[INIT] Independent capture {cam}: {input_folder}; "
                            f"poll={CAMERA_POLL_INTERVAL}s")

    def notify_new_log_barcode(barcode):
        if not ENABLE_COM_OUTPUT:
            return
        if not com_output.enabled:
            log_message(f"[COM] WARNING: cannot send n for barcode={barcode}; output unavailable")
            return
        if com_output.send_new_code():
            log_message(f"[COM] New log barcode={barcode}; sent n\\r\\n")
        else:
            log_message(f"[COM] WARNING: failed to send n for barcode={barcode}")

    try:
        # Connect before starting the log reader, including during CSV startup retries.
        com_output = COMOutput(
            port=COM_PORT, baudrate=COM_BAUDRATE,
            enabled=ENABLE_COM_OUTPUT, retry_count=COM_RETRY)
        if PRODUCT_MODE == 'log' and product_reader:
            product_reader.on_new_barcode = notify_new_log_barcode
        if product_reader:
            product_reader.start()
        current_product_code = get_current_product_code()
        if PRODUCT_MODE == 'log' and product_reader:
            last_product_event_id = product_reader.get_event_id()

        log_message(f"[START] product={current_product_code}; mode={PRODUCT_MODE}")
        # Preserve retrying startup until a valid product configuration exists.
        while True:
            success, roi_rules, used_cameras = try_load_product_config(current_product_code)
            if success:
                break
            time.sleep(0.5)
            current_product_code = get_current_product_code()

        camera_config = load_camera_config(
            csv_file=CAMERA_CONFIG_CSV,
            create_folders=CAMERA_CREATE_FOLDERS,
            verbose=False)
        print_camera_config_summary(camera_config)

        gui = ResultGUI(
            window_name=GUI_WINDOW_NAME_TEMPLATE.format(product_code=current_product_code),
            max_history=GUI_MAX_HISTORY)
        if GUI_ENABLED:
            gui.start()
        executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="AI")
        configure_watchers(used_cameras)
        log_message(f"[WAITING] Cameras: {used_cameras}")

        while True:
            service_capture()
            if (CAPTURE_STATS_INTERVAL > 0
                    and time.monotonic() - last_stats_at >= CAPTURE_STATS_INTERVAL):
                for cam, watcher in watchers.items():
                    log_message(f"[CAPTURE STATS {cam}] {watcher.stats()}")
                last_stats_at = time.monotonic()

            # GUI remains on the main thread, including while inference runs.
            action = gui.show(wait_time=30) if GUI_ENABLED else None
            if not GUI_ENABLED:
                time.sleep(0.03)
            if action == "quit":
                break

            new_code = get_current_product_code()
            new_event_id = (product_reader.get_event_id()
                            if PRODUCT_MODE == 'log' and product_reader else None)
            is_new_event = (new_event_id is not None
                            and last_product_event_id is not None
                            and new_event_id != last_product_event_id)
            if new_code != current_product_code or is_new_event:
                success, new_rules, new_cameras = try_load_product_config(new_code)
                if new_event_id is not None:
                    last_product_event_id = new_event_id
                if success:
                    epoch += 1
                    # A running inference cannot be safely killed. Its result will
                    # be ignored, including COM output, via the epoch guard below.
                    configure_watchers(new_cameras)
                    for watcher in watchers.values():
                        watcher.reset()
                    images = {}
                    last_images = {}
                    last_results = {}
                    last_gui_items = {}
                    pending_ng = set()
                    mode = "collect"
                    current_product_code = new_code
                    roi_rules, used_cameras = new_rules, new_cameras
                    gui.reset_display()
                    gui.window_name = GUI_WINDOW_NAME_TEMPLATE.format(
                        product_code=current_product_code)
                    log_message(f"[BOUNDARY] product={new_code}; epoch={epoch}; "
                                "old queued/decoded images explicitly invalidated")

            # Never call result() until done(); no future waiting on GUI thread.
            if future is not None:
                if not future.done():
                    continue
                completed_job = job
                completed_items = job_items
                try:
                    result = future.result()
                except Exception:
                    for watcher, packet in completed_items.values():
                        watcher.discard(packet, "AI job failed")
                    raise
                finally:
                    future = None
                    job = None
                    job_items = {}

                if completed_job["epoch"] != epoch:
                    log_message("[AI] Obsolete product result suppressed; no COM sent")
                    continue

                if completed_job["kind"] == "decode":
                    images.update(result)
                    continue

                if completed_job["kind"] == "batch":
                    last_images = result["images"]
                    last_results = result["results"]
                    last_gui_items = result["gui_items"]
                    images = {}
                else:
                    cam = completed_job["camera"]
                    ng_ids = {
                        item.get("roi_id")
                        for item in last_results.get(cam, [])
                        if not item.get("pass", True)
                    }
                    last_images[cam] = result["images"][cam]
                    last_results[cam] = merge_updates_for_ng_rois(
                        last_results.get(cam, []), result["results"][cam], ng_ids)
                    last_gui_items[cam] = merge_updates_for_ng_rois(
                        last_gui_items.get(cam, []), result["gui_items"][cam], ng_ids)
                    if all(item.get("pass", False) for item in last_results[cam]):
                        pending_ng.discard(cam)

                merged_results = merge_cam_results(last_results, used_cameras)
                merged_gui = merge_cam_results(last_gui_items, used_cameras)

                # GUI and existing COM ordering remain main-thread-owned.
                # Saving the result/COM can still briefly block this thread,
                # but NEVER holds source files or stops acquisition.
                service_capture()
                final_status, batch_time, _ = emit_batch_result(
                    merged_results, merged_gui, completed_job["batch_num"],
                    completed_job["product_code"], gui, com_output,
                    result["started_at"])
                if RESULT_SAVE_CSV_HISTORY:
                    append_history_csv(
                        RESULT_CSV_HISTORY_PATH,
                        {"batch_num": completed_job["batch_num"],
                         "product_code": completed_job["product_code"],
                         "batch_time": batch_time},
                        final_status, merged_results)

                if final_status == "OK":
                    pending_ng = set()
                    mode = ("wait_scan" if PRODUCT_MODE == 'log' and product_reader
                            else "collect")
                    if mode == "wait_scan":
                        images = {}
                        log_message("[WAIT] OK; waiting for next scan boundary")
                else:
                    if completed_job["kind"] == "batch":
                        pending_ng = set(get_ng_cameras(last_results))
                    mode = "redetect" if pending_ng else "collect"
                    log_message(f"[REDETECT] pending={sorted(pending_ng)}")
                continue

            if mode == "wait_scan":
                continue

            cameras_needed = (used_cameras if mode == "collect"
                              else [cam for cam in used_cameras if cam in pending_ng])

            # Process ready images before dequeuing more: at most one decoded
            # image per camera is retained by the coordinator.
            if mode == "collect" and all(cam in images for cam in used_cameras):
                batch_num += 1
                job = {"kind": "batch", "epoch": epoch, "batch_num": batch_num,
                       "product_code": current_product_code}
                batch_images = {cam: images[cam] for cam in used_cameras}
                images = {}
                future = executor.submit(
                    detect_images, batch_images, list(roi_rules),
                    batch_num, run_detection_for_cam)
                continue

            if mode == "redetect":
                ready = [cam for cam in cameras_needed if cam in images]
                if ready:
                    cam = ready[0]
                    batch_num += 1
                    job = {"kind": "redetect", "epoch": epoch,
                           "batch_num": batch_num, "camera": cam,
                           "product_code": current_product_code}
                    future = executor.submit(
                        detect_images, {cam: images.pop(cam)}, list(roi_rules),
                        batch_num, run_detection_for_cam)
                    continue

            packets = {}
            for cam in cameras_needed:
                if cam in images:
                    continue
                packet = watchers[cam].get_packet_nowait()
                if packet is not None:
                    packets[cam] = (watchers[cam], packet)
            if packets:
                job = {"kind": "decode", "epoch": epoch}
                job_items = packets
                future = executor.submit(
                    decode_packets, packets,
                    ACQUISITION_CFG.get('require_complete_markers', True),
                    ACQUISITION_CFG.get('max_decoded_pixels', 40_000_000))

    except KeyboardInterrupt:
        exit_code = 130
        log_message(f"[STOP] Interrupted after {batch_num} detection jobs")
    except Exception as exc:
        exit_code = 2
        log_message(f"[FATAL] {exc}")
        print("Acquisition/AI stopped. The external producer is NOT stopped. "
              "Do not treat this as an OK inspection result.")
    finally:
        # Stop acquisition first; then drain/finish the single AI job safely.
        for cam, watcher in watchers.items():
            watcher.stop()
            for code, message in watcher.drain_events():
                log_message(f"[CAPTURE {cam}] {code}: {message}")
        if executor is not None:
            executor.shutdown(wait=True)
        for cam, watcher in watchers.items():
            # Explicitly account for packets still retained at shutdown.
            watcher.discard_pending("shutdown before processing")
            for code, message in watcher.drain_events():
                log_message(f"[CAPTURE SHUTDOWN {cam}] {code}: {message}")
            log_message(f"[CAPTURE FINAL {cam}] {watcher.stats()}")
        for resource, method in (
                (gui, "close"), (com_output, "close"), (product_reader, "stop")):
            if resource is not None:
                try:
                    getattr(resource, method)()
                except Exception as exc:
                    log_message(f"[SHUTDOWN] {method}: {exc}")

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
