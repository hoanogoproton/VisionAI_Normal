"""Count+angle unit and real run_detection_for_cam integration tests."""
import ast
import copy
import math
from pathlib import Path
import unittest
import numpy as np
from modules.comparator import compare_count_and_angle, compare_angle
from modules.csv_loader import load_product_csv
from modules.roi_manager import prepare_roi_data

ROI = (0, 0, 100, 100)

def detection(angle=0, center=(20, 20), confidence=0.8):
    x, y = center
    return {
        "bbox": (x-5, y-5, x+5, y+5), "confidence": confidence,
        "keypoints": [(x, y, 1), (x+10*math.cos(math.radians(angle)),
                                      y+10*math.sin(math.radians(angle)), 1)]}

def check(dets, **kwargs):
    config = dict(min_count=1, max_count=1, keypoint_idx_1=0,
                  keypoint_idx_2=1, expected_angle=0, angle_tolerance=10)
    config.update(kwargs)
    return compare_count_and_angle({"found": bool(dets), "detections": dets},
                                   ROI, **config)

class CountAngleTests(unittest.TestCase):
    def test_one_good(self):
        self.assertTrue(check([detection()])[0])
    def test_one_bad(self):
        passed, reason, details = check([detection(90)])
        self.assertFalse(passed)
        self.assertIn("WRONG_ANGLE", reason)
        self.assertEqual(len(details["angle_results"]), 1)
    def test_two_good_but_excess_count(self):
        passed, reason, details = check([detection(), detection(center=(60, 60))])
        self.assertFalse(passed)
        self.assertTrue(reason.startswith("COUNT_NG"))
        self.assertEqual(len(details["angle_results"]), 2)
    def test_two_good_exact_two(self):
        self.assertTrue(check([detection(), detection(center=(60, 60))],
                              min_count=2, max_count=2)[0])
    def test_lower_confidence_bad_not_ignored(self):
        passed, reason, details = check(
            [detection(confidence=.99), detection(90, (60, 60), .55)],
            min_count=2, max_count=2)
        self.assertFalse(passed)
        self.assertIn("obj2", reason)
        self.assertTrue(details["angle_results"][0]["passed"])
        self.assertFalse(details["angle_results"][1]["passed"])
    def test_bad_angle_not_removed_from_count(self):
        passed, reason, details = check(
            [detection(), detection(90, (60, 60))])
        self.assertFalse(passed)
        self.assertEqual(details["detected_count"], 2)
        self.assertTrue(reason.startswith("COUNT_NG"))
    def test_outside_high_confidence_ignored(self):
        passed, _, details = check(
            [detection(90, (150, 150), .99), detection(confidence=.6)])
        self.assertTrue(passed)
        self.assertEqual(details["detected_count"], 1)
        self.assertEqual(len(details["angle_results"]), 1)
    def test_empty_default_ng(self):
        self.assertFalse(check([])[0])
    def test_allowed_empty(self):
        self.assertTrue(check([], min_count=0, max_count=0)[0])
    def test_empty_invalid_config(self):
        self.assertEqual(check([], min_count=0, keypoint_idx_1=-1)[1],
                         "INVALID_ANGLE_CONFIG")
    def test_count_only_backwards_compatible(self):
        self.assertTrue(check([{"bbox": (1,1,3,3)}],
            keypoint_idx_1=None, keypoint_idx_2=None,
            expected_angle=None, angle_tolerance=None)[0])
    def test_unbounded_max_and_default_min(self):
        self.assertTrue(check([detection(),detection()], min_count=None,
                              max_count=None)[0])
    def test_partial_angle_config(self):
        self.assertEqual(check([detection()], keypoint_idx_2=None)[1],
                         "INVALID_ANGLE_CONFIG")
    def test_missing_keypoints(self):
        d = detection(); d["keypoints"] = None
        self.assertIn("NO_KEYPOINTS", check([d])[1])
    def test_out_of_range_index(self):
        self.assertIn("KEYPOINT_INDEX_OUT_OF_RANGE",
                      check([detection()], keypoint_idx_2=3)[1])
    def test_negative_and_same_indices(self):
        for k1, k2 in [(-1,1),(0,-1),(0,0)]:
            self.assertEqual(check([detection()], keypoint_idx_1=k1,
                                   keypoint_idx_2=k2)[1], "INVALID_ANGLE_CONFIG")
    def test_degenerate_points(self):
        d=detection(); d["keypoints"][1]=d["keypoints"][0]
        self.assertIn("DEGENERATE_KEYPOINTS",check([d])[1])
    def test_nonfinite_points(self):
        d=detection();d["keypoints"][1]=(float("nan"),20,1)
        self.assertIn("INVALID_KEYPOINTS",check([d])[1])
    def test_wrap_360(self):
        self.assertTrue(check([detection(359)],expected_angle=1,
                              angle_tolerance=2.01)[0])
    def test_normalize_expected_angle(self):
        self.assertTrue(check([detection()],expected_angle=720)[0])
    def test_invalid_tolerance(self):
        for t in [-1,181,float("nan"),float("inf")]:
            self.assertEqual(check([detection()],angle_tolerance=t)[1],
                             "INVALID_ANGLE_CONFIG")
    def test_invalid_counts(self):
        for config in [dict(min_count=-1),dict(min_count=2,max_count=1),
                       dict(max_count=-1),dict(min_count=1.5)]:
            self.assertEqual(check([detection()],**config)[1],
                             "INVALID_COUNT_CONFIG")
    def test_center_on_boundary(self):
        self.assertTrue(check([detection(center=(100,100))])[0])
    def test_single_detection_legacy_format(self):
        d=detection();d["found"]=True
        self.assertTrue(compare_count_and_angle(
            d,ROI,1,1,0,1,0,10)[0])
    def test_input_not_modified(self):
        ds=[detection()];original=copy.deepcopy(ds)
        check(ds)
        self.assertEqual(ds, original)
    def test_malformed_bbox(self):
        self.assertEqual(check([{"bbox":None}])[1],"INVALID_BBOX")
    def test_csv_example_with_bom(self):
        rules=load_product_csv(str(Path(__file__).resolve().parents[1] /
                                  "examples/count_keypoint.csv"))
        self.assertEqual(len(rules),1)
        self.assertEqual(rules[0]["min_count"],1)
        self.assertEqual(rules[0]["keypoint_idx_2"],3)

class MainCountIntegrationTests(unittest.TestCase):
    def run_detection(self, ds):
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
                  check_mode="count",min_count=1,max_count=1,keypoint_idx_1=0,
                  keypoint_idx_2=1,expected_angle=0,angle_tolerance=10)
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

if __name__=="__main__":
    unittest.main()
