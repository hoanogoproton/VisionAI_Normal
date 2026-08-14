"""
Module: log_input
Chuc nang: Doc barcode moi nhat tu file log he thong.

Ho tro format log nhu:
2026/04/11 00:01:56.241	... Read Barcode : EGMP8VT3V30A.

Reader nay duoc thiet ke de dung cung interface voi COM reader:
- start()
- stop()
- get_current()
"""

import os
import re
import time
import threading
from datetime import datetime
from typing import Optional


class LogBarcodeReader:
    """Doc barcode moi nhat tu file log theo thu muc ngay."""

    def __init__(
        self,
        base_dir: str,
        log_filename: str = "WH_AVIS_System.log",
        date_folder_format: str = "%Y%m%d",
        poll_interval: float = 0.5,
        tail_bytes: int = 65536,
    ):
        self.base_dir = base_dir
        self.log_filename = log_filename
        self.date_folder_format = date_folder_format
        self.poll_interval = max(0.1, float(poll_interval))
        self.tail_bytes = max(4096, int(tail_bytes))

        self.current_barcode = None
        self.current_log_path = None
        self.file_offset = 0
        self.last_poll_time = 0.0
        self.is_running = False
        self.event_id = 0

        self.total_received = 0
        self.total_errors = 0

        self.barcode_pattern = re.compile(r"Read Barcode\s*:\s*([^\s]+)", re.IGNORECASE)
        self.lock = threading.Lock()

    def _build_log_path(self) -> str:
        date_folder = datetime.now().strftime(self.date_folder_format)
        return os.path.join(self.base_dir, date_folder, self.log_filename)

    def _decode_bytes(self, data: bytes) -> str:
        for enc in ("utf-8-sig", "utf-8", "cp932", "shift_jis"):
            try:
                return data.decode(enc)
            except UnicodeDecodeError:
                continue
        return data.decode("latin-1", errors="ignore")

    def _normalize_barcode(self, barcode: str) -> str:
        cleaned = barcode.strip().rstrip(".,;:!?。！？")
        return cleaned[:5]

    def _extract_latest_barcode(self, text: str) -> Optional[str]:
        matches = self.barcode_pattern.findall(text)
        if not matches:
            return None
        barcode = self._normalize_barcode(matches[-1])
        return barcode or None

    def _bootstrap_from_tail(self, log_path: str) -> None:
        if not os.path.isfile(log_path):
            return

        try:
            size = os.path.getsize(log_path)
            with open(log_path, "rb") as f:
                if size > self.tail_bytes:
                    f.seek(size - self.tail_bytes)
                data = f.read()

            text = self._decode_bytes(data)
            barcode = self._extract_latest_barcode(text)

            with self.lock:
                self.file_offset = size
                if barcode:
                    self.current_barcode = barcode
                    self.event_id += 1
                    self.total_received += 1

            if barcode:
                print(f"[LOG_INPUT] Latest barcode from tail: '{barcode}'")

        except Exception as e:
            self.total_errors += 1
            print(f"[LOG_INPUT] WARNING: bootstrap error: {e}")

    def _poll_once(self) -> None:
        now = time.time()
        if (now - self.last_poll_time) < self.poll_interval:
            return
        self.last_poll_time = now

        log_path = self._build_log_path()

        if log_path != self.current_log_path:
            self.current_log_path = log_path
            self.file_offset = 0
            print(f"[LOG_INPUT] Switched log file: {self.current_log_path}")
            self._bootstrap_from_tail(log_path)
            return

        if not os.path.isfile(log_path):
            return

        try:
            file_size = os.path.getsize(log_path)
            if file_size < self.file_offset:
                self.file_offset = 0

            if file_size == self.file_offset:
                return

            with open(log_path, "rb") as f:
                f.seek(self.file_offset)
                data = f.read()
                self.file_offset = f.tell()

            if not data:
                return

            text = self._decode_bytes(data)
            barcode = self._extract_latest_barcode(text)
            if not barcode:
                return

            with self.lock:
                self.current_barcode = barcode
                self.event_id += 1
                self.total_received += 1
                print(f"[LOG_INPUT] Received: '{barcode}' (Total: {self.total_received})")

        except Exception as e:
            self.total_errors += 1
            print(f"[LOG_INPUT] WARNING: read error: {e}")

    def start(self) -> None:
        if self.is_running:
            print("[LOG_INPUT] Already running")
            return
        self.is_running = True
        self.current_log_path = self._build_log_path()
        print(f"[LOG_INPUT] Reader started (file={self.current_log_path})")
        self._bootstrap_from_tail(self.current_log_path)

    def stop(self) -> None:
        self.is_running = False
        print(f"[LOG_INPUT] Stopped. Stats: Received={self.total_received}, Errors={self.total_errors}")

    def get_current(self) -> Optional[str]:
        if self.is_running:
            self._poll_once()
        with self.lock:
            return self.current_barcode

    def get_event_id(self) -> int:
        if self.is_running:
            self._poll_once()
        with self.lock:
            return self.event_id
