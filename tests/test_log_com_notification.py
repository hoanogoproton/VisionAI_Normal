"""Offline regression tests for new-log-barcode COM notifications."""
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

from modules.log_input import LogBarcodeReader
from modules.com_output import COMOutput
from test_main_state_machine import Harness


class LogCOMTests(unittest.TestCase):
    def make_reader(self, directory, callback):
        reader = LogBarcodeReader(directory, poll_interval=0.1,
                                  on_new_barcode=callback)
        path = Path(reader._build_log_path())
        path.parent.mkdir(parents=True, exist_ok=True)
        return reader, path

    def poll(self, reader):
        reader.last_poll_time = 0
        reader.get_current()

    def test_startup_history_is_not_notified(self):
        with tempfile.TemporaryDirectory() as tmp:
            received = []
            reader, path = self.make_reader(tmp, received.append)
            path.write_bytes(b"Read Barcode : ABCDE123.\r\n")
            reader.start()
            self.assertEqual(reader.get_current(), "ABCDE")
            self.assertEqual(received, [])
            reader.stop()

    def test_new_and_repeat_lines_each_notify_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            received = []
            reader, path = self.make_reader(tmp, received.append)
            path.write_bytes(b"")
            reader.start()
            with path.open("ab") as f:
                f.write(b"Read Barcode : ABCDE123.\r\n"
                        b"Read Barcode : ABCDE123.\r\n"
                        b"Read Barcode : FGHIJ456.\r\n")
            self.poll(reader)
            self.poll(reader)
            self.assertEqual(received, ["ABCDE", "ABCDE", "FGHIJ"])
            self.assertEqual(reader.get_event_id(), 3)
            reader.stop()

    def test_partial_line_waits_for_lf(self):
        with tempfile.TemporaryDirectory() as tmp:
            received = []
            reader, path = self.make_reader(tmp, received.append)
            path.write_bytes(b"")
            reader.start()
            with path.open("ab") as f:
                f.write(b"Read Barcode : AB")
            self.poll(reader)
            self.assertEqual(received, [])
            with path.open("ab") as f:
                f.write(b"CDE123.\r\n")
            self.poll(reader)
            self.assertEqual(received, ["ABCDE"])
            reader.stop()

    def test_pending_startup_line_is_not_lost(self):
        with tempfile.TemporaryDirectory() as tmp:
            received = []
            reader, path = self.make_reader(tmp, received.append)
            path.write_bytes(b"Read Barcode : AB")
            reader.start()
            with path.open("ab") as f:
                f.write(b"CDE123.\r\n")
            self.poll(reader)
            self.assertEqual(received, ["ABCDE"])
            reader.stop()

    def test_missing_file_created_after_start_is_notified(self):
        with tempfile.TemporaryDirectory() as tmp:
            received = []
            reader, path = self.make_reader(tmp, received.append)
            reader.start()
            path.write_bytes(b"Read Barcode : ABCDE123.\r\n")
            self.poll(reader)
            self.assertEqual(received, ["ABCDE"])
            reader.stop()

    def test_runtime_file_switch_notifies_all_scans(self):
        with tempfile.TemporaryDirectory() as tmp:
            received = []
            reader, path = self.make_reader(tmp, received.append)
            path.write_bytes(b"Read Barcode : OLD00123.\r\n")
            reader.start()
            next_path = path.parent / "next_day.log"
            next_path.write_bytes(b"Read Barcode : ABCDE123.\r\n"
                                  b"Read Barcode : FGHIJ456.\r\n")
            reader._build_log_path = lambda: str(next_path)
            self.poll(reader)
            self.assertEqual(received, ["ABCDE", "FGHIJ"])
            reader.stop()

    def test_callback_error_does_not_crash_reader(self):
        with tempfile.TemporaryDirectory() as tmp:
            def fail(code):
                raise RuntimeError("offline")
            reader, path = self.make_reader(tmp, fail)
            path.write_bytes(b"")
            reader.start()
            path.write_bytes(b"Read Barcode : ABCDE123.\r\n")
            self.poll(reader)
            self.assertEqual(reader.get_current(), "ABCDE")
            self.assertEqual(reader.total_errors, 1)
            reader.stop()

    def test_exact_serial_bytes_and_existing_results(self):
        sent = []
        class SerialPort:
            is_open = True
            def __init__(self, **kwargs): pass
            def write(self, data): sent.append(data); return len(data)
            def flush(self): pass
            def close(self): self.is_open = False
        fake = types.SimpleNamespace(Serial=SerialPort, SerialException=OSError)
        with patch.dict(sys.modules, {"serial": fake}):
            output = COMOutput()
            self.assertTrue(output.send_new_code())
            self.assertTrue(output.send_result("o1,o2"))
            self.assertEqual(sent, [b"n\r\n", b"o1,o2\r\n"])
            output.close()

    def test_main_wiring_repeat_scans_and_disabled_output(self):
        for enabled in (True, False):
            with self.subTest(enabled=enabled), tempfile.TemporaryDirectory() as tmp:
                h = Harness(tmp, mode="log")
                received = []
                reader, path = self.make_reader(tmp, None)
                path.write_bytes(b"Read Barcode : OLD00123.\r\n")
                class Output:
                    def __init__(self, **kw): self.enabled = kw["enabled"]
                    def send_new_code(self): received.append(b"n\r\n"); return True
                    def close(self): pass
                h.namespace.update(
                    product_reader=reader, COMOutput=Output,
                    ENABLE_COM_OUTPUT=enabled,
                    get_current_product_code=lambda: reader.get_current() or "DEFAULT")
                def ui(h):
                    if h.tick == 1:
                        with path.open("ab") as f:
                            f.write(b"Read Barcode : ABCDE123.\r\n"
                                    b"Read Barcode : ABCDE123.\r\n")
                        reader.last_poll_time = 0
                    if h.tick == 3:
                        return "quit"
                h.callback = ui
                h.run()
                self.assertEqual(received, [b"n\r\n"] * (2 if enabled else 0))
                self.assertFalse(any("[FATAL]" in entry for entry in h.logs))

    def test_main_notifies_while_waiting_for_valid_csv(self):
        with tempfile.TemporaryDirectory() as tmp:
            h = Harness(tmp, mode="log")
            received = []
            reader, path = self.make_reader(tmp, None)
            path.write_bytes(b"Read Barcode : OLD00123.\r\n")
            class Output:
                def __init__(self, **kw): self.enabled = True
                def send_new_code(self): received.append(b"n\r\n"); return True
                def close(self): pass
            attempts = []
            def load(code):
                attempts.append(code)
                if len(attempts) == 1:
                    with path.open("ab") as f:
                        f.write(b"Read Barcode : ABCDE123.\r\n")
                    reader.last_poll_time = 0
                    return False, None, None
                return True, [{"product": code}], ["CAM1"]
            h.namespace.update(
                product_reader=reader, COMOutput=Output, ENABLE_COM_OUTPUT=True,
                get_current_product_code=lambda: reader.get_current() or "DEFAULT",
                try_load_product_config=load)
            h.callback = lambda h: "quit"
            h.run()
            self.assertEqual(received, [b"n\r\n"])
            self.assertEqual(attempts, ["OLD00", "ABCDE"])
            self.assertFalse(any("[FATAL]" in entry for entry in h.logs))


if __name__ == "__main__":
    unittest.main(verbosity=2)
