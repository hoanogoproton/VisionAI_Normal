"""Offline tests + real Windows handle-sharing tests when run on Windows."""
import ast
import os
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch
from concurrent.futures import ThreadPoolExecutor
import cv2
import numpy as np

from modules.image_watcher import ImageWatcher, AcquisitionOverload
from modules.shared_image_reader import (read_shared_bytes, open_shared_fd,
    ImageTooLarge, file_identity, _matches_expected)
from modules.ai_worker import decode_packets, detect_images

def image_bytes(value=80, extension=".png"):
    ok, encoded = cv2.imencode(extension, np.full((12, 16, 3), value, np.uint8))
    assert ok
    return encoded.tobytes()

def wait_for(callback, timeout=2.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        result = callback()
        if result:
            return result
        time.sleep(0.002)
    raise AssertionError("Timed out")

class CaptureTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.folder = Path(self.tmp.name) / "input"
        self.folder.mkdir()
        self.watchers = []

    def tearDown(self):
        for watcher in self.watchers:
            watcher.stop()
        self.tmp.cleanup()

    def watcher(self, start=True, **kw):
        options = dict(poll_interval=0.005, retry_timeout=0.5)
        options.update(kw)
        w = ImageWatcher(self.folder, **options)
        self.watchers.append(w)
        if start:
            w.start()
        return w

    def put(self, name="a.png", value=80):
        path = self.folder / name
        path.write_bytes(image_bytes(value, ".jpg" if name.endswith(".jpg") else ".png"))
        return path

    def test_reads_bytes_and_source_can_be_deleted_before_decode(self):
        w = self.watcher()
        path = self.put()
        packet = wait_for(w.get_packet_nowait)
        path.unlink()
        decoded = decode_packets({"CAM1": (w, packet)})
        self.assertEqual(decoded["CAM1"].shape, (12, 16, 3))
        self.assertEqual(w.stats()["accepted"], 1)

    def test_all_three_images_are_fifo_and_not_discarded(self):
        w = self.watcher(start=False)
        for i in range(3):
            self.put(f"{i}.png", 10 + i)
        w.start()
        wait_for(lambda: w.stats()["captured"] == 3)
        values = []
        for _ in range(3):
            packet = w.get_packet_nowait()
            values.append(int(decode_packets({"C": (w, packet)})["C"][0, 0, 0]))
        self.assertEqual(values, [10, 11, 12])
        self.assertEqual(w.stats()["queue_items"], 0)

    def test_decode_failure_is_not_success_and_retries(self):
        w = self.watcher(retry_delays=(0.01, 0.01))
        (self.folder / "bad.jpg").write_bytes(b"\xff\xd8incomplete")
        p = wait_for(w.get_packet_nowait)
        self.assertEqual(decode_packets({"C": (w, p)}), {})
        self.assertNotIn("bad.jpg", w.processed_files)
        p2 = wait_for(w.get_packet_nowait)
        self.assertNotEqual(p.token, p2.token)
        decode_packets({"C": (w, p2)})
        p3 = wait_for(w.get_packet_nowait)
        decode_packets({"C": (w, p3)})
        wait_for(lambda: w.stats()["exhausted"] == 1)
        scans = w.stats()["scans"]
        time.sleep(0.04)
        self.assertEqual(w.stats()["captured"], 3)
        self.assertGreater(w.stats()["scans"], scans)

    def test_partial_write_then_complete_is_recovered(self):
        w = self.watcher()
        path = self.folder / "a.jpg"
        raw = image_bytes(95, ".jpg")
        path.write_bytes(raw[:len(raw)//2])
        p = wait_for(w.get_packet_nowait)
        decode_packets({"C": (w, p)})
        path.write_bytes(raw)
        p2 = wait_for(w.get_packet_nowait)
        decoded = decode_packets({"C": (w, p2)})
        self.assertEqual(int(decoded["C"][0, 0, 0]), 95)

    def test_same_filename_two_versions_preserved(self):
        w = self.watcher()
        path = self.put(value=10)
        wait_for(lambda: w.stats()["captured"] == 1)
        time.sleep(0.01)
        path.write_bytes(image_bytes(140))
        wait_for(lambda: w.stats()["captured"] == 2)
        vals = []
        for _ in range(2):
            p = w.get_packet_nowait()
            vals.append(int(decode_packets({"C": (w, p)})["C"][0, 0, 0]))
        self.assertEqual(vals, [10, 140])

    def test_reset_rejects_old_lease_and_ignores_existing(self):
        w = self.watcher()
        self.put()
        p = wait_for(w.get_packet_nowait)
        w.reset()
        self.assertEqual(decode_packets({"C": (w, p)}), {})
        time.sleep(0.03)
        self.assertIsNone(w.get_packet_nowait())
        self.put("new.png")
        p2 = wait_for(w.get_packet_nowait)
        self.assertNotEqual(p.generation, p2.generation)

    def test_reset_during_read_cannot_enqueue_stale_packet(self):
        w = self.watcher(start=False)
        self.put()
        started, release = threading.Event(), threading.Event()
        real_reader = read_shared_bytes
        def delayed(*args, **kwargs):
            started.set()
            release.wait(1)
            return real_reader(*args, **kwargs)
        with patch("modules.image_watcher.read_shared_bytes", delayed):
            w.start()
            self.assertTrue(started.wait(1))
            w.reset()
            release.set()
            time.sleep(0.04)
        self.assertEqual(w.stats()["captured"], 0)
        self.assertIsNone(w.get_packet_nowait())

    def test_overload_is_latched_no_drop_oldest(self):
        w = self.watcher(start=False, queue_max_images=1)
        self.put("a.png", 10)
        self.put("b.png", 20)
        w.start()
        wait_for(lambda: w.stats()["overloads"] == 1)
        with self.assertRaises(AcquisitionOverload):
            w.check_health()
        self.assertEqual(w.stats()["captured"], 1)
        self.assertEqual(w.stats()["queue_items"], 1)
        self.assertEqual(w.stats()["boundary_discarded"], 0)

    def test_byte_budget_overload(self):
        raw = image_bytes()
        w = self.watcher(start=False, queue_max_bytes=len(raw) + 10,
                         max_image_bytes=len(raw))
        self.put("a.png")
        self.put("b.png")
        w.start()
        wait_for(lambda: w.stats()["overloads"] == 1)
        self.assertLessEqual(w.stats()["queue_bytes"], len(raw) + 10)

    def test_real_poll_schedule_and_one_enumeration_per_scan(self):
        w = self.watcher(start=False, poll_interval=0.03)
        original = w._snapshot
        calls = []
        def wrapped():
            calls.append(time.monotonic())
            return original()
        with patch.object(w, "_snapshot", wrapped):
            w.start()
            time.sleep(0.16)
            w.stop()
        self.assertGreaterEqual(len(calls), 3)
        self.assertLessEqual(len(calls), 8)
        self.assertEqual(len(calls), w.stats()["scans"])
        self.assertTrue(all(b-a >= 0.015 for a,b in zip(calls,calls[1:])))

    def test_slow_ai_does_not_stop_capture(self):
        w = self.watcher()
        with ThreadPoolExecutor(max_workers=1) as ex:
            future = ex.submit(time.sleep, 0.2)
            for i in range(6):
                self.put(f"{i}.png", i)
                time.sleep(0.012)
            wait_for(lambda: w.stats()["captured"] == 6)
            self.assertFalse(future.done())
        self.assertEqual(w.stats()["ready_items"], 6)

    def test_missing_after_decode_error_is_not_retried_forever(self):
        w = self.watcher()
        path = self.folder / "bad.jpg"
        path.write_bytes(b"invalid")
        p = wait_for(w.get_packet_nowait)
        path.unlink()
        decode_packets({"C": (w, p)})
        wait_for(lambda: w.stats()["missing"] >= 1)
        time.sleep(0.04)
        self.assertEqual(w.stats()["captured"], 1)

    def test_initial_files_ignored_and_no_temp_created(self):
        self.put()
        temp = Path(self.tmp.name) / "temp-not-created"
        w = self.watcher(temp_folder=temp)
        time.sleep(0.03)
        self.assertEqual(w.stats()["captured"], 0)
        self.assertFalse(temp.exists())
        self.assertTrue((self.folder / "a.png").exists())

    def test_reader_limit_and_fd_closed_on_error(self):
        path = self.put()
        with self.assertRaises(ImageTooLarge):
            read_shared_bytes(path, max_bytes=1)
        path.unlink()  # On Windows, also verifies no disallow-delete leaked handle.

    def test_windows_scan_handle_comparison_uses_common_native_fields(self):
        actual = (123, 456, 1710000000123456700, 101, 250)
        directory = (900, 456, 1710000000123456700, 999, 250)
        self.assertTrue(_matches_expected(actual, directory, windows=True))
        self.assertFalse(_matches_expected(actual, directory, windows=False))
        changed = (900, 456, 1710000000123456700, 999, 251)
        self.assertFalse(_matches_expected(actual, changed, windows=True))

    def test_shared_read_preserves_scanner_version_token(self):
        path = self.put()
        token = file_identity(path.stat())
        raw, returned = read_shared_bytes(path, expected_identity=token)
        self.assertEqual(returned, token)
        self.assertEqual(raw, path.read_bytes())

    def test_transient_sharing_violation_retries_then_succeeds(self):
        w = self.watcher(start=False)
        path = self.put()
        real = read_shared_bytes
        calls = []
        def transient(*args, **kwargs):
            calls.append(1)
            if len(calls) <= 2:
                exc = PermissionError("simulated sharing conflict")
                exc.winerror = 32
                raise exc
            return real(*args, **kwargs)
        with patch("modules.image_watcher.read_shared_bytes", transient):
            w.start()
            p = wait_for(w.get_packet_nowait)
        self.assertEqual(len(calls), 3)
        self.assertEqual(w.stats()["read_errors"], 2)
        self.assertTrue(decode_packets({"C": (w, p)}))
        self.assertTrue(path.exists())

    def test_oversized_input_is_error_not_queue_overload(self):
        w = self.watcher(start=False, max_image_bytes=64, queue_max_bytes=128)
        (self.folder / "huge.png").write_bytes(b"x" * 200)
        w.start()
        wait_for(lambda: w.stats()["read_errors"] == 1)
        w.check_health()
        self.assertEqual(w.stats()["overloads"], 0)
        self.assertEqual(w.stats()["captured"], 0)
        self.assertTrue(any(code == "IMAGE_TOO_LARGE" for code, _ in w.drain_events()))

    def test_disappearance_between_scan_and_open_is_bounded(self):
        w = self.watcher(start=False)
        path = self.put()
        def disappears(*args, **kwargs):
            path.unlink()
            return read_shared_bytes(*args, **kwargs)
        with patch("modules.image_watcher.read_shared_bytes", disappears):
            w.start()
            wait_for(lambda: w.stats()["missing"] >= 1)
            time.sleep(0.03)
        self.assertEqual(w.stats()["read_errors"], 1)
        self.assertEqual(w.stats()["captured"], 0)
        w.check_health()

    def test_metadata_change_during_read_rejected_and_fd_closed(self):
        path = self.put()
        before = file_identity(path.stat())
        after = list(before)
        after[2] += 1000000
        with patch("modules.shared_image_reader._handle_identity",
                   side_effect=[before, tuple(after)]):
            from modules.shared_image_reader import ImageNotReady
            with self.assertRaises(ImageNotReady):
                read_shared_bytes(path, expected_identity=before)
        path.unlink()

class WindowsSharingTests(unittest.TestCase):
    @unittest.skipUnless(os.name == "nt", "Real Windows sharing test")
    def test_delete_while_reader_handle_is_open(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "image.png"
            raw = image_bytes()
            path.write_bytes(raw)
            fd = open_shared_fd(path)
            try:
                os.unlink(path)
                self.assertEqual(os.read(fd, len(raw)), raw)
            finally:
                os.close(fd)

    @unittest.skipUnless(os.name == "nt", "Real Windows sharing test")
    def test_exclusive_writer_is_not_bypassed(self):
        import ctypes
        from modules.shared_image_reader import _k32, _INVALID
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "image.png"
            path.write_bytes(image_bytes())
            handle = _k32.CreateFileW(str(path), 0x40000000, 0, None, 3, 0x80, None)
            self.assertNotEqual(handle, _INVALID)
            try:
                with self.assertRaises(OSError) as err:
                    read_shared_bytes(path)
                self.assertEqual(err.exception.winerror, 32)
            finally:
                _k32.CloseHandle(handle)

    @unittest.skipUnless(os.name == "nt", "Real Windows metadata test")
    def test_scanned_token_matches_native_windows_handle(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "image.png"
            raw = image_bytes()
            path.write_bytes(raw)
            expected = file_identity(path.stat())
            data, token = read_shared_bytes(path, expected_identity=expected)
            self.assertEqual(data, raw)
            self.assertEqual(token, expected)

    @unittest.skipUnless(os.name == "nt", "Real Windows rename sharing test")
    def test_rename_while_reader_handle_is_open(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "image.png"
            dest = Path(tmp) / "renamed.png"
            raw = image_bytes()
            path.write_bytes(raw)
            fd = open_shared_fd(path)
            try:
                os.rename(path, dest)
                self.assertEqual(os.read(fd, len(raw)), raw)
            finally:
                os.close(fd)

class MainCoordinatorTests(unittest.TestCase):
    def test_real_main_control_flow_fifo_and_gui_during_ai(self):
        # Execute unchanged AST of main() with hardware/model boundaries stubbed.
        source = (Path(__file__).resolve().parents[1] / "main.py").read_text(encoding="utf8")
        tree = ast.parse(source)
        selected = [
            node for node in tree.body if isinstance(node, ast.FunctionDef)
            and node.name in {"main", "merge_cam_results",
                              "merge_updates_for_ng_rois", "get_ng_cameras"}
        ]
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp) / "input"
            folder.mkdir()
            emitted, detected, gui_calls = [], [], []
            watchers = []
            class W(ImageWatcher):
                def __init__(self, *args, **kw):
                    super().__init__(*args, **kw)
                    watchers.append(self)
            class GUI:
                total_ok = 0
                total_ng = 0
                def __init__(self, **kw):
                    self.window_name = kw["window_name"]
                    self.tick = 0
                def start(self): pass
                def reset_display(self): pass
                def close(self): pass
                def show(self, **kw):
                    self.tick += 1
                    gui_calls.append(time.monotonic())
                    if self.tick in (1, 8, 16):
                        n = (1, 8, 16).index(self.tick)
                        (folder / f"{n}.png").write_bytes(image_bytes(30+n))
                    time.sleep(0.002)
                    if len(emitted) >= 3 or self.tick > 800:
                        return "quit"
            class COM:
                def __init__(self, **kw): pass
                def close(self): pass
            def detector(cam, image, rules, number):
                detected.append(int(image[0,0,0]))
                time.sleep(0.06)
                return ([{"camera": cam, "roi_id":"R", "pass":True}], [])
            def emit(results, items, number, product, gui, com, start):
                emitted.append(number)
                return "OK", time.time()-start, ""
            namespace = dict(
                os=os, time=time, ImageWatcher=W,
                ThreadPoolExecutor=ThreadPoolExecutor,
                decode_packets=decode_packets, detect_images=detect_images,
                ResultGUI=GUI, COMOutput=COM, log_message=lambda message: None,
                product_reader=None, PRODUCT_MODE="manual",
                GUI_ENABLED=True, GUI_WINDOW_NAME_TEMPLATE="{product_code}",
                GUI_MAX_HISTORY=10, COM_PORT="X", COM_BAUDRATE=9600,
                ENABLE_COM_OUTPUT=False, COM_RETRY=0,
                RESULT_SAVE_CSV_HISTORY=False, CAMERA_CONFIG_CSV="unused",
                CAMERA_CREATE_FOLDERS=False, CAMERA_POLL_INTERVAL=0.005,
                ACQUISITION_OPTIONS={}, ACQUISITION_CFG={},
                get_current_product_code=lambda: "P",
                try_load_product_config=lambda code: (True, [], ["CAM1"]),
                load_camera_config=lambda **kw: {},
                print_camera_config_summary=lambda cfg: None,
                get_camera_folder=lambda cam,cfg,kind: str(folder),
                run_detection_for_cam=detector, emit_batch_result=emit,
            )
            exec(compile(ast.Module(body=selected, type_ignores=[]),
                         "main.py", "exec"), namespace)
            namespace["main"]()
            self.assertEqual(detected, [30,31,32])
            self.assertEqual(emitted, [1,2,3])
            self.assertGreater(len(gui_calls), 50)
            self.assertEqual(watchers[0].stats()["accepted"], 3)

if __name__ == "__main__":
    unittest.main(verbosity=2)
