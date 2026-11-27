"""Compare all-object count/angle regression and main integration tests."""
import ast
from pathlib import Path
import unittest
import numpy as np
from modules.comparator import compare_count_and_angle, compare_detection
from modules.roi_manager import prepare_roi_data
from modules.csv_loader import load_product_csv
from test_count_keypoint import detection, ROI

class CompareDetectionTests(unittest.TestCase):
    def test_all_objects_counted(self):
        ds = [detection(), detection(center=(60,60))]
        self.assertFalse(compare_detection({"detections": ds}, ROI, 1, 1)[0])
        self.assertTrue(compare_detection({"detections": ds}, ROI, 2, 2)[0])

    def test_outside_high_confidence_ignored(self):
        self.assertTrue(compare_detection({"detections": [
            detection(center=(150,150), confidence=.99),
            detection(confidence=.55)]}, ROI)[0])

    def test_legacy_single_and_empty(self):
        self.assertTrue(compare_detection(dict(found=True, **detection()), ROI)[0])
        self.assertFalse(compare_detection({"found": False}, ROI)[0])
        self.assertTrue(compare_detection({"found": False}, ROI, 0, 0)[0])

    def test_csv(self):
        rules = load_product_csv(str(Path(__file__).resolve().parents[1] /
                                    "examples/compare_count_keypoint.csv"))
        self.assertEqual(rules[0]["check_mode"], "compare")
        self.assertEqual(rules[0]["min_count"], 1)
        self.assertEqual(rules[0]["max_count"], 1)

class MainCompareIntegrationTests(unittest.TestCase):
    def run_detection(self, ds, **overrides):
        source=(Path(__file__).resolve().parents[1]/"main.py").read_text(encoding="utf8")
        fn=next(n for n in ast.parse(source).body
                if isinstance(n,ast.FunctionDef) and n.name=="run_detection_for_cam")
        logs=[]
        ns=dict(log_message=logs.append,prepare_roi_data=prepare_roi_data,
                get_model=lambda name:object(),
                detect_object=lambda *args,**kwargs:dict(found=bool(ds),detections=ds),
                compare_count_and_angle=compare_count_and_angle)
        exec(compile(ast.Module(body=[fn],type_ignores=[]),"main.py","exec"),ns)
        rule=dict(camera="CAM1",roi_id="R1",model_name="pose",class_id=0,
                  class_name="object",confidence=.5,detect_roi=ROI,compare_roi=ROI,
                  check_mode="compare",min_count=1,max_count=1,keypoint_idx_1=0,
                  keypoint_idx_2=1,expected_angle=0,angle_tolerance=10)
        rule.update(overrides)
        return ns["run_detection_for_cam"]("CAM1",np.zeros((100,100,3),np.uint8),
                                           [rule],1)
    def test_main_emits_count_and_angle_details(self):
        results,items=self.run_detection([detection()])
        self.assertTrue(results[0]["pass"])
        self.assertEqual(results[0]["detected_count"],1)
        self.assertEqual(len(items[0]["angle_results"]),1)
    def test_main_excess_two_good_ng(self):
        results,items=self.run_detection([detection(),detection(center=(60,60))])
        self.assertFalse(results[0]["pass"])
        self.assertEqual(len(items[0]["angle_results"]),2)
    def test_main_bad_angle_ng(self):
        results,items=self.run_detection([detection(90)])
        self.assertFalse(results[0]["pass"])
        self.assertIn("WRONG_ANGLE",results[0]["reason"])
        self.assertFalse(items[0]["angle_results"][0]["passed"])


    def test_outside_high_confidence_does_not_override_inside(self):
        results, items = self.run_detection(
            [detection(90, (150,150), .99), detection(confidence=.55)])
        self.assertTrue(results[0]["pass"])
        self.assertEqual(results[0]["detected_count"], 1)

    def test_all_objects_angle_checked_not_only_highest_confidence(self):
        results, items = self.run_detection(
            [detection(confidence=.99), detection(90, (60,60), .55)],
            min_count=2, max_count=2)
        self.assertFalse(results[0]["pass"])
        self.assertEqual(len(items[0]["angle_results"]), 2)
        self.assertIn("obj2", results[0]["reason"])

    def test_exact_two_ok(self):
        results, items = self.run_detection(
            [detection(), detection(center=(60,60))], min_count=2, max_count=2)
        self.assertTrue(results[0]["pass"])
        self.assertEqual(results[0]["detected_count"], 2)

    def test_count_only_and_defaults(self):
        results, items = self.run_detection(
            [detection(), detection()], min_count=None, max_count=None,
            keypoint_idx_1=None, keypoint_idx_2=None,
            expected_angle=None, angle_tolerance=None)
        self.assertTrue(results[0]["pass"])
        self.assertEqual(results[0]["detected_count"], 2)
        self.assertEqual(items[0]["angle_results"], [])

    def test_zero_allowed(self):
        results, items = self.run_detection([], min_count=0, max_count=0)
        self.assertTrue(results[0]["pass"])

    def test_no_objects_default_ng(self):
        results, items = self.run_detection([])
        self.assertFalse(results[0]["pass"])
        self.assertEqual(results[0]["detected_count"], 0)

    def test_partial_config_ng(self):
        results, items = self.run_detection([detection()], keypoint_idx_2=None)
        self.assertFalse(results[0]["pass"])
        self.assertEqual(results[0]["reason"], "INVALID_ANGLE_CONFIG")


class BatchCompareIntegrationTests(unittest.TestCase):
    def run_batch(self, ds, minimum=1, maximum=1):
        import os
        import time
        from datetime import datetime
        from types import SimpleNamespace
        captured = {}
        logs = []
        rule = dict(camera="CAM1", roi_id="R1", model_name="pose",
                    class_id=0, class_name="object", confidence=.5,
                    detect_roi=ROI, compare_roi=ROI, check_mode="compare",
                    min_count=minimum, max_count=maximum, keypoint_idx_1=0,
                    keypoint_idx_2=1, expected_angle=0, angle_tolerance=10)
        source = (Path(__file__).resolve().parents[1] /
                  "batch_detect_folder.py").read_text(encoding="utf8")
        fn = next(n for n in ast.parse(source).body
                  if isinstance(n, ast.FunctionDef) and n.name == "batch_detect_folder")
        def save(items, status, *args, **kwargs):
            captured.update(items=items, status=status)
        ns = dict(os=os, time=time, datetime=datetime,
            load_config=lambda path: dict(paths=dict(log_dir="unused", output_dir="unused"),
                product=dict(csv_path="{code}.csv"), camera=dict(config_csv="unused")),
            BatchLogger=lambda path: SimpleNamespace(log=logs.append),
            load_product_csv=lambda path: [rule],
            build_product_signal_plan=lambda rules: [],
            load_camera_config=lambda *a, **k: {},
            list_image_files=lambda *a, **k: ["stub.jpg"],
            cv2=SimpleNamespace(imread=lambda path: np.zeros((100,100,3), np.uint8)),
            prepare_roi_data=prepare_roi_data,
            get_model=lambda name: object(),
            detect_object=lambda *a, **k: dict(found=bool(ds), detections=ds),
            compare_count_and_angle=compare_count_and_angle,
            save_batch_result=save,
            save_result_csv=lambda *a: "unused",
            save_ng_roi_images=lambda *a: ("unused", "unused", 0))
        exec(compile(ast.Module(body=[fn], type_ignores=[]),
                     "batch_detect_folder.py", "exec"), ns)
        ns["batch_detect_folder"]("unused", "P1", "CAM1")
        self.assertIn("items", captured, logs)
        return captured

    def test_batch_exact_two_and_overlay_details(self):
        result = self.run_batch([detection(), detection(center=(60,60))], 2, 2)
        self.assertEqual(result["status"], "OK")
        self.assertEqual(result["items"][0]["detected_count"], 2)
        self.assertEqual(len(result["items"][0]["angle_results"]), 2)

    def test_batch_lower_confidence_bad_angle_ng(self):
        result = self.run_batch(
            [detection(confidence=.99), detection(90,(60,60),.55)], 2, 2)
        self.assertEqual(result["status"], "NG")
        self.assertIn("obj2", result["items"][0]["reason"])

    def test_batch_excess_count_ng(self):
        result = self.run_batch([detection(), detection()])
        self.assertEqual(result["status"], "NG")
        self.assertTrue(result["items"][0]["reason"].startswith("COUNT_NG"))


if __name__ == "__main__":
    unittest.main()
