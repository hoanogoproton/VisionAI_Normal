"""Actual main() state-machine tests with GUI, COM, models and readers stubbed."""
import ast
import os
from pathlib import Path
import tempfile
import time
import unittest
from concurrent.futures import ThreadPoolExecutor

from modules.image_watcher import ImageWatcher
from modules.ai_worker import decode_packets, detect_images
from test_async_acquisition import image_bytes

class Harness:
    def __init__(self, folder, mode="manual", cameras=("CAM1",), queue_size=32):
        self.folder = Path(folder)
        self.mode = mode
        self.cameras = list(cameras)
        self.product = "P1"
        self.event = 0
        self.tick = 0
        self.logs, self.emitted, self.detected, self.watchers = [], [], [], []
        self.callback = lambda h: None
        self.detection = lambda cam, image, rules, number: (
            [{"camera":cam, "roi_id":"R", "pass":True}], [])
        harness = self
        class Reader:
            def start(self): pass
            def stop(self): pass
            def get_event_id(self): return harness.event
            def get_current(self): return harness.product
        class GUI:
            total_ok = 0
            total_ng = 0
            def __init__(self, **kw): self.window_name = kw["window_name"]
            def start(self): pass
            def reset_display(self): pass
            def close(self): pass
            def show(self, **kw):
                harness.tick += 1
                action = harness.callback(harness)
                time.sleep(0.002)
                return action or ("quit" if harness.tick > 1000 else None)
        class COM:
            def __init__(self, **kw): pass
            def close(self): pass
        class W(ImageWatcher):
            def __init__(self, *args, **kw):
                super().__init__(*args, **kw)
                harness.watchers.append(self)
        def detector(cam, image, rules, number):
            harness.detected.append((cam, int(image[0,0,0]), rules[0]["product"]))
            return harness.detection(cam, image, rules, number)
        def emit(results, items, number, product, gui, com, start):
            status = "OK" if all(item["pass"] for item in results) else "NG"
            harness.emitted.append((product, status, results))
            return status, time.time()-start, ""
        self.namespace = dict(
            os=os, time=time, ImageWatcher=W, ThreadPoolExecutor=ThreadPoolExecutor,
            decode_packets=decode_packets, detect_images=detect_images,
            ResultGUI=GUI, COMOutput=COM, log_message=self.logs.append,
            product_reader=Reader() if mode == "log" else None,
            PRODUCT_MODE=mode, GUI_ENABLED=True, GUI_WINDOW_NAME_TEMPLATE="{product_code}",
            GUI_MAX_HISTORY=10, COM_PORT="X", COM_BAUDRATE=9600,
            ENABLE_COM_OUTPUT=False, COM_RETRY=0, RESULT_SAVE_CSV_HISTORY=False,
            CAMERA_CONFIG_CSV="unused", CAMERA_CREATE_FOLDERS=False,
            CAMERA_POLL_INTERVAL=0.005, ACQUISITION_CFG={},
            ACQUISITION_OPTIONS={"queue_max_images":queue_size},
            get_current_product_code=lambda: harness.product,
            try_load_product_config=lambda code: (
                True, [{"product":code}], list(harness.cameras)),
            load_camera_config=lambda **kw: {},
            print_camera_config_summary=lambda cfg: None,
            get_camera_folder=lambda cam,cfg,kind: str(self.folder / cam),
            run_detection_for_cam=detector, emit_batch_result=emit,
        )
        source = (Path(__file__).resolve().parents[1] / "main.py").read_text(encoding="utf8")
        nodes = [node for node in ast.parse(source).body
                 if isinstance(node, ast.FunctionDef) and node.name in {
                     "main","merge_cam_results","merge_updates_for_ng_rois","get_ng_cameras"}]
        exec(compile(ast.Module(body=nodes,type_ignores=[]), "main.py","exec"),
             self.namespace)

    def put(self, name, value, cam="CAM1"):
        (self.folder / cam / name).write_bytes(image_bytes(value))

    def run(self):
        self.namespace["main"]()

class StateMachineTests(unittest.TestCase):
    def test_product_change_during_inference_suppresses_old_com_result(self):
        with tempfile.TemporaryDirectory() as tmp:
            h = Harness(tmp,mode="log")
            state = {}
            def detector(cam,image,rules,number):
                time.sleep(0.07)
                return [{"camera":cam,"roi_id":"R","pass":True}], []
            h.detection = detector
            def ui(h):
                if h.tick == 1:
                    h.put("old.png",10)
                if h.detected and "switch_tick" not in state:
                    h.product="P2"
                    h.event+=1
                    state["switch_tick"]=h.tick
                if "switch_tick" in state and h.tick==state["switch_tick"]+3:
                    h.put("new.png",20)
                if h.emitted:
                    return "quit"
            h.callback=ui
            h.run()
            self.assertEqual([entry[0] for entry in h.emitted],["P2"])
            self.assertEqual([entry[1] for entry in h.detected],[10,20])
            self.assertTrue(any("Obsolete product" in message for message in h.logs))
            self.assertFalse(any("[FATAL]" in message for message in h.logs))

    def test_repeat_scan_same_product_invalidates_old_job(self):
        with tempfile.TemporaryDirectory() as tmp:
            h=Harness(tmp,mode="log")
            state={}
            def detector(cam,image,rules,number):
                time.sleep(0.07)
                return [{"camera":cam,"roi_id":"R","pass":True}], []
            h.detection=detector
            def ui(h):
                if h.tick==1: h.put("old.png",10)
                if h.detected and "switch_tick" not in state:
                    h.event+=1
                    state["switch_tick"]=h.tick
                if "switch_tick" in state and h.tick==state["switch_tick"]+3:
                    h.put("new.png",20)
                if h.emitted: return "quit"
            h.callback=ui
            h.run()
            self.assertEqual(len(h.emitted),1)
            self.assertEqual([entry[1] for entry in h.detected],[10,20])
            self.assertTrue(any("Obsolete product" in message for message in h.logs))

    def test_redetect_only_updates_previously_ng_rois(self):
        with tempfile.TemporaryDirectory() as tmp:
            h=Harness(tmp)
            state={}
            def detector(cam,image,rules,number):
                retry=int(image[0,0,0])==20
                return [
                    {"camera":cam,"roi_id":"R_OK","pass":not retry},
                    {"camera":cam,"roi_id":"R_NG","pass":retry},
                ], []
            h.detection=detector
            def ui(h):
                if h.tick==1: h.put("first.png",10)
                if len(h.emitted)==1 and not state:
                    h.put("retry.png",20)
                    state["retry"]=True
                if len(h.emitted)==2:return "quit"
            h.callback=ui
            h.run()
            self.assertEqual([entry[1] for entry in h.emitted],["NG","OK"])
            self.assertTrue(all(item["pass"] for item in h.emitted[-1][2]))
            self.assertEqual([entry[1] for entry in h.detected],[10,20])

    def test_two_camera_three_batches_fifo(self):
        with tempfile.TemporaryDirectory() as tmp:
            h=Harness(tmp,cameras=("CAM1","CAM2"))
            def ui(h):
                if h.tick in (1,8,16):
                    value=(1,8,16).index(h.tick)
                    h.put(f"{value}.png",10+value,"CAM1")
                    h.put(f"{value}.png",20+value,"CAM2")
                if len(h.emitted)==3:return "quit"
            h.callback=ui
            h.run()
            self.assertEqual([entry[1] for entry in h.detected],[10,20,11,21,12,22])
            self.assertEqual(len(h.emitted),3)
            self.assertFalse(any("[FATAL]" in message for message in h.logs))

    def test_overload_aborts_before_pending_result_emission(self):
        with tempfile.TemporaryDirectory() as tmp:
            h=Harness(tmp,queue_size=1)
            state={}
            def detector(cam,image,rules,number):
                time.sleep(0.1)
                return [{"camera":cam,"roi_id":"R","pass":True}], []
            h.detection=detector
            def ui(h):
                if h.tick==1:h.put("first.png",10)
                if h.detected and not state:
                    h.put("extra1.png",20)
                    h.put("extra2.png",30)
                    state["filled"]=True
            h.callback=ui
            h.run()
            self.assertEqual(h.emitted,[])
            self.assertTrue(any("[FATAL]" in message for message in h.logs))
            self.assertEqual(h.watchers[0].stats()["overloads"],1)

if __name__=="__main__":
    unittest.main(verbosity=2)
