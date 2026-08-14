"""
Module: result_gui
Chuc nang: Hien thi ket qua detect len cua so GUI (OpenCV)
    - Crop tung ROI, ve bbox + compare_roi + OK/NG
    - Xep thanh luoi (grid)
    - Header: batch info, product code, final result
    - Footer: thong ke OK/NG qua cac batch
    - Non-blocking: gui.show() goi cv2.waitKey() de GUI khong bi do

Hien thi ROI crop, bbox, compare_roi, OK/NG, Angle, Measure Width chat luong cao
    (giong nhu save_batch_result trong result_visualizer.py)
"""

import cv2
import numpy as np
from typing import Any, Dict, List, Tuple, Optional
from modules.wire_measure import draw_measurement


class ResultGUI:
    """
    GUI hien thi ket qua detect cua tung batch.
    
    Cach dung:
        gui = ResultGUI()
        gui.start()
        
        # Trong vong lap chinh:
        while True:
            action = gui.show(wait_time=30)   # Giu GUI song
            if action == 'quit':
                break
            
            # Khi co ket qua batch:
            gui.update(roi_items, final_status, batch_info)
        
        gui.close()
    """

    def __init__(self, window_name: str = "VisionAI - ROI Result", 
                 max_history: int = 10):
        """
        Args:
            window_name: Ten cua so hien thi
            max_history: So batch luu lich su (hien thi o footer)
        """
        self.window_name = window_name
        self.max_history = max_history
        
        # Trang thai
        self.is_running = False
        self.paused = False
        
        # Du lieu hien thi hien tai
        self._display_image = None    # Anh dang hien thi
        self._roi_items = []          # Danh sach ROI items cua batch hien tai
        self._final_status = "N/A"    # OK / NG
        self._batch_info = {}         # {"batch_num": 1, "product_code": "ABC123", ...}
        
        # Thong ke
        self.total_batches = 0
        self.total_ok = 0
        self.total_ng = 0
        
        # Mau sac (BGR)
        self.COLOR_OK = (0, 200, 0)       # Xanh la
        self.COLOR_NG = (0, 0, 220)       # Do
        self.COLOR_BBOX = (0, 255, 255)   # Vang
        self.COLOR_COMPARE = (255, 200, 0) # Cyan nhat
        self.COLOR_HEADER = (50, 50, 50)  # Xam dam
        self.COLOR_FOOTER = (40, 40, 40)  # Xam dam
        self.COLOR_WHITE = (255, 255, 255)
        self.COLOR_BLACK = (0, 0, 0)
    
    # ==================================================================
    # PUBLIC METHODS
    # ==================================================================
    
    def start(self) -> None:
        """Tao cua so OpenCV"""
        self.is_running = True
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, 1600, 900)
        print(f"[GUI] Window started: {self.window_name}")
        print(f"[GUI] Controls: Q = Quit | SPACE = Pause/Resume")
    
    def update(self, roi_items: List[Dict], final_status: str, 
               batch_info: Dict) -> None:
        """
        Cap nhat GUI voi ket qua batch moi.
        
        Args:
            roi_items: Danh sach ROI, moi item la dict:
                {
                    "roi_id": "ROI_01",
                    "camera": "CAM1",
                    "crop_image": numpy array (anh crop tu detect_roi),
                    "passed": True/False,
                    "reason": "OK" / "NOT_FOUND" / "OUT_OF_COMPARE_ROI",
                    "detect_result": {"found": bool, "detections": list},
                    "detect_roi": (x1, y1, x2, y2),
                    "compare_roi": (x1, y1, x2, y2),
                }
            final_status: "OK" hoac "NG"
            batch_info: {"batch_num": int, "product_code": str, "batch_time": float}
        """
        def camera_sort_key(item: Dict) -> Tuple[int, str]:
            camera = str(item.get("camera", ""))
            if camera.upper().startswith("CAM"):
                suffix = camera[3:]
                if suffix.isdigit():
                    return int(suffix), camera
            return 999, camera

        self._roi_items = sorted(roi_items, key=camera_sort_key)
        self._final_status = final_status
        self._batch_info = batch_info
        
        # Cap nhat thong ke
        self.total_batches += 1
        if final_status == "OK":
            self.total_ok += 1
        else:
            self.total_ng += 1
        
        # Tao anh hien thi
        self._display_image = self._create_display()
    
    def show(self, wait_time: int = 30) -> Optional[str]:
        """
        Hien thi GUI va doc phim nhan. GOI MOI VONG LAP.
        
        Args:
            wait_time: Thoi gian cho phim nhan (ms). 
                       30ms ~ 33 FPS, du muot.
        
        Returns:
            'quit'   - Nguoi dung nhan Q hoac ESC
            'pause'  - Nhan SPACE (dang chay → pause)
            'resume' - Nhan SPACE (dang pause → chay lai)
            None     - Khong co gi
        """
        if not self.is_running:
            return None
        
        # Tao anh hien thi
        if self._display_image is not None:
            display = self._display_image
        else:
            display = self._create_waiting_screen()
        
        # Hien thi
        cv2.imshow(self.window_name, display)
        
        # Doc phim nhan
        key = cv2.waitKey(wait_time) & 0xFF
        
        if key == ord('q') or key == ord('Q') or key == 27:  # Q hoac ESC
            return 'quit'
        elif key == ord(' '):  # SPACE
            self.paused = not self.paused
            state = "PAUSED" if self.paused else "RESUMED"
            print(f"[GUI] {state}")
            return 'pause' if self.paused else 'resume'
        
        return None
    
    def close(self) -> None:
        """Dong cua so"""
        self.is_running = False
        cv2.destroyAllWindows()
        print(f"[GUI] Window closed")

    def reset_display(self) -> None:
        """Reset GUI ve trang thai ban dau (man hinh cho)"""
        self._display_image = None
        self._roi_items = []
        self._final_status = None
        self._batch_info = None
        print(f"[GUI] Display reset to waiting screen")
    
    # ==================================================================
    # PRIVATE: Tao anh hien thi
    # ==================================================================
    
    def _create_waiting_screen(self) -> np.ndarray:
        """Tao anh cho khi chua co du lieu"""
        canvas = np.zeros((720, 1280, 3), dtype=np.uint8)
        
        # Header
        cv2.rectangle(canvas, (0, 0), (1280, 60), self.COLOR_HEADER, -1)
        cv2.putText(canvas, "VisionAI - SHWS", 
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, self.COLOR_WHITE, 2)
        
        # Waiting message
        cv2.putText(canvas, "Waiting for images...", 
                    (420, 380), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (150, 150, 150), 2)
        cv2.putText(canvas, "Q = Quit | SPACE = Pause", 
                    (440, 420), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 100), 1)
        
        return canvas
    
    def _create_display(self) -> np.ndarray:
        """Tao anh hien thi hoan chinh: header + grid ROI + footer"""
        
        if not self._roi_items:
            return self._create_waiting_screen()
        
        # --- Buoc 1: Ve tung ROI crop ---
        roi_images = []
        for item in self._roi_items:
            roi_img = self._draw_roi_crop(item)
            roi_images.append(roi_img)
        
        # --- Buoc 2: Xep luoi ---
        grid = self._arrange_grid(roi_images)
        grid_h, grid_w = grid.shape[:2]
        
        # --- Buoc 3: Ghep header + grid + footer ---
        header_h = 80
        footer_h = 60
        total_h = header_h + grid_h + footer_h
        total_w = max(grid_w, 800)  # Toi thieu 800px de header du cho
        
        canvas = np.zeros((total_h, total_w, 3), dtype=np.uint8)
        
        # Header
        self._draw_header(canvas, total_w, header_h)
        
        # Grid (dat giua neu grid nho hon canvas)
        x_offset = (total_w - grid_w) // 2
        canvas[header_h:header_h + grid_h, x_offset:x_offset + grid_w] = grid
        
        # Footer
        self._draw_footer(canvas, total_w, total_h, footer_h)
        
        # Pause indicator
        if self.paused:
            cv2.putText(canvas, "PAUSED", 
                        (total_w - 160, 40), cv2.FONT_HERSHEY_SIMPLEX,
                        1.0, (0, 255, 255), 2)
        
        return canvas
    
    def _draw_header(self, canvas: np.ndarray, width: int, height: int) -> None:
        """Ve header: batch info + final result"""
        cv2.rectangle(canvas, (0, 0), (width, height), self.COLOR_HEADER, -1)
        
        # Final result (ben trai)
        status = self._final_status
        color = self.COLOR_OK if status == "OK" else self.COLOR_NG
        
        # Background cho status
        cv2.rectangle(canvas, (10, 8), (140, 52), color, -1)
        cv2.putText(canvas, status, 
                    (30, 43), cv2.FONT_HERSHEY_SIMPLEX, 1.2, self.COLOR_BLACK, 3)
        
        # Batch info (ben phai)
        batch_num = self._batch_info.get("batch_num", "?")
        product = self._batch_info.get("product_code", "?")
        batch_time = self._batch_info.get("batch_time", 0)
        
        info_text = f"Batch #{batch_num} | Product: {product} | Time: {batch_time:.2f}s"
        cv2.putText(canvas, info_text, 
                    (160, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.COLOR_WHITE, 1)
    
    def _draw_footer(self, canvas: np.ndarray, width: int, 
                     total_h: int, footer_h: int) -> None:
        """Ve footer: thong ke"""
        y_start = total_h - footer_h
        cv2.rectangle(canvas, (0, y_start), (width, total_h), self.COLOR_FOOTER, -1)
        
        stats = (f"Total Batches: {self.total_batches} | "
                 f"OK: {self.total_ok} | NG: {self.total_ng}")
        cv2.putText(canvas, stats, 
                    (20, y_start + 32), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.6, self.COLOR_WHITE, 1)
        
        # Controls hint (ben phai)
        cv2.putText(canvas, "Q=Quit | SPACE=Pause", 
                    (width - 280, y_start + 32), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (120, 120, 120), 1)
    
    # ==================================================================
    # PRIVATE: Ve tung ROI crop
    # ==================================================================
    
    def _draw_roi_crop(self, item: Dict) -> np.ndarray:
        """
        Ve ket qua len anh crop cua 1 ROI (chat luong cao, tuong tu save).
        
        item:
            roi_id, camera, crop_image, passed, reason,
            detect_result, detect_roi, compare_roi
        """
        crop = item.get("crop_image")
        if crop is None or crop.size == 0:
            result_img = np.zeros((300, 400, 3), dtype=np.uint8)
            cv2.putText(result_img, "NO IMAGE", (80, 150), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (100, 100, 100), 3)
            return result_img

        passed = item.get("passed", False)

        # Measurement overlay (measure_width mode)
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

        # Scale factor for higher quality
        scale_factor = 1.5
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

        # Offset scaled
        offset_x = detect_roi[0] * scale_factor
        offset_y = detect_roi[1] * scale_factor

        color = self.COLOR_OK if passed else self.COLOR_NG

        # 1. Border (thicker)
        cv2.rectangle(result_img, (0, 0), (w - 1, h - 1), color, 4)

        # 2. ROI ID + status (larger font)
        status_text = "OK" if passed else "NG"
        label = f"{roi_id} ({camera}): {status_text}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_size = cv2.getTextSize(label, font, 0.8, 2)[0]
        cv2.rectangle(result_img, (0, 0), (text_size[0] + 12, text_size[1] + 16), color, -1)
        cv2.putText(result_img, label, (6, text_size[1] + 9), font, 0.8, self.COLOR_BLACK, 2)

        # 3. Compare ROI (dashed, scaled)
        cx1 = int((compare_roi[0] - detect_roi[0]) * scale_factor)
        cy1 = int((compare_roi[1] - detect_roi[1]) * scale_factor)
        cx2 = int((compare_roi[2] - detect_roi[0]) * scale_factor)
        cy2 = int((compare_roi[3] - detect_roi[1]) * scale_factor)
        self._draw_dashed_rect(result_img, (cx1, cy1), (cx2, cy2), self.COLOR_COMPARE, 2)

        # 4. Bounding boxes (scaled)
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

            bx1 = int((bbox[0] - detect_roi[0]) * scale_factor)
            by1 = int((bbox[1] - detect_roi[1]) * scale_factor)
            bx2 = int((bbox[2] - detect_roi[0]) * scale_factor)
            by2 = int((bbox[3] - detect_roi[1]) * scale_factor)

            cv2.rectangle(result_img, (bx1, by1), (bx2, by2), self.COLOR_BBOX, 3)

            conf_text = f"Conf: {confidence:.2f}"
            cv2.putText(result_img, conf_text, (bx1, by2 + 25),
                        font, 0.7, self.COLOR_BBOX, 2)

            cx_pt = (bx1 + bx2) // 2
            cy_pt = (by1 + by2) // 2
            cv2.circle(result_img, (cx_pt, cy_pt), 5, color, -1)

        # 5. Keypoints and angle (scaled)
        angle_info = item.get("angle_info")
        if angle_info and detect_result.get("found"):
            COLOR_KP = (0, 255, 255)
            COLOR_ANGLE = (255, 0, 255)

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
            cv2.arrowedLine(result_img, akp1, akp2, COLOR_ANGLE, 2, tipLength=0.2)

            mid_x = (akp1[0] + akp2[0]) // 2
            mid_y = (akp1[1] + akp2[1]) // 2
            angle_text = f"{angle_info['measured_angle']:.1f}"
            cv2.putText(result_img, angle_text, (mid_x, mid_y - 40),
                        font, 0.6, COLOR_ANGLE, 2)

        # 6. Reason (bottom, larger font)
        if reason:
            cv2.putText(result_img, reason, (6, h - 10), 
                        font, 0.6, (200, 200, 200), 2)

        return result_img
    
    def _draw_dashed_rect(self, img: np.ndarray, 
                          pt1: Tuple[int, int], pt2: Tuple[int, int],
                          color: Tuple[int, int, int], thickness: int) -> None:
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
    
    # ==================================================================
    # PRIVATE: Xep luoi
    # ==================================================================
    
    def _arrange_grid(self, roi_images: List[np.ndarray]) -> np.ndarray:
        """
        Xep danh sach anh ROI thanh luoi (grid) voi KICH THUOC DONG NHAT.
        Resize tat ca ROI ve cung kich thuoc cell (chat luong cao nhu batch result).
        """
        n = len(roi_images)
        
        if n == 0:
            return np.zeros((400, 800, 3), dtype=np.uint8)
        
        # Tinh so hang x cot
        cols = max(1, int(np.ceil(np.sqrt(n))))
        rows = max(1, int(np.ceil(n / cols)))
        
        # Kich thuoc cell (chat luong cao)
        cell_w = 700
        cell_h = 700
        
        padding = 8
        
        canvas_h = rows * cell_h + (rows - 1) * padding
        canvas_w = cols * cell_w + (cols - 1) * padding
        
        canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
        
        for idx, roi_img in enumerate(roi_images):
            row = idx // cols
            col = idx % cols
            
            y_start = row * (cell_h + padding)
            x_start = col * (cell_w + padding)
            
            # RESIZE ROI de fit vao cell (giu aspect ratio)
            h, w = roi_img.shape[:2]
            scale = min(cell_w / w, cell_h / h)
            new_w = int(w * scale)
            new_h = int(h * scale)
            
            # Resize bang INTER_CUBIC de chat luong tot
            img_resized = cv2.resize(roi_img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
            
            # Dat giua cell
            y_offset = (cell_h - new_h) // 2
            x_offset = (cell_w - new_w) // 2
            y_end = y_start + y_offset + new_h
            x_end = x_start + x_offset + new_w
            
            # Dam bao khong vuot khoi canvas
            if (y_end <= canvas_h) and (x_end <= canvas_w):
                canvas[y_start + y_offset:y_end, x_start + x_offset:x_end] = img_resized
        
        return canvas


# ============================================================================
# TEST
# ============================================================================

def main():
    """Test nhanh GUI voi du lieu gia"""
    print("[TEST] ResultGUI\n")
    
    # Tao anh gia
    fake_crop_1 = np.ones((200, 300, 3), dtype=np.uint8) * 80
    cv2.rectangle(fake_crop_1, (50, 50), (150, 150), (0, 255, 0), 2)
    
    fake_crop_2 = np.ones((200, 300, 3), dtype=np.uint8) * 60
    cv2.rectangle(fake_crop_2, (30, 30), (180, 180), (0, 0, 255), 2)
    
    fake_crop_3 = np.ones((200, 300, 3), dtype=np.uint8) * 100
    
    # Tao ROI items
    roi_items = [
        {
            "roi_id": "ROI_01", "camera": "CAM1",
            "crop_image": fake_crop_1,
            "passed": True, "reason": "OK",
            "detect_result": {
                "found": True,
                "detections": [{"bbox": (150, 150, 250, 250), "confidence": 0.92, "keypoints": None}]
            },
            "detect_roi": (100, 100, 400, 300),
            "compare_roi": (120, 120, 380, 280),
        },
        {
            "roi_id": "ROI_02", "camera": "CAM2",
            "crop_image": fake_crop_2,
            "passed": False, "reason": "NOT_FOUND",
            "detect_result": {"found": False, "detections": []},
            "detect_roi": (50, 50, 350, 250),
            "compare_roi": (70, 70, 330, 230),
        },
        {
            "roi_id": "ROI_03", "camera": "CAM3",
            "crop_image": fake_crop_3,
            "passed": True, "reason": "OK",
            "detect_result": {
                "found": True,
                "detections": [{"bbox": (160, 160, 260, 260), "confidence": 0.88, "keypoints": None}]
            },
            "detect_roi": (100, 100, 400, 300),
            "compare_roi": (110, 110, 390, 290),
        },
    ]
    
    # Test GUI
    gui = ResultGUI()
    gui.start()
    
    # Simulate batch 1
    gui.update(roi_items, "NG", {
        "batch_num": 1,
        "product_code": "ABC123x",
        "batch_time": 1.23,
    })
    
    print("[INFO] Nhan Q de thoat, SPACE de pause/resume")
    
    while True:
        action = gui.show(wait_time=30)
        if action == 'quit':
            break
    
    gui.close()
    print("[DONE] Test completed!")


if __name__ == "__main__":
    main()
