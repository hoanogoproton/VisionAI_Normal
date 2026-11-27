"""Verify count-angle overlays using synthetic detections; no files generated."""

import numpy as np
from unittest.mock import patch
from modules.comparator import compare_count_and_angle
from modules.result_visualizer import draw_count_angle_results, _draw_roi_crop_for_batch
from modules.result_gui import ResultGUI
ds=[
 {"bbox":(120,120,140,140),"confidence":.9,
  "keypoints":[(125,125,1),(135,125,1)]},
 {"bbox":(160,160,180,180),"confidence":.8,
  "keypoints":[(165,165,1),(165,175,1)]}]
passed,reason,details=compare_count_and_angle(
 {"found":True,"detections":ds},(100,100,200,200),2,2,0,1,0,10)
image=np.zeros((150,150,3),np.uint8)
with patch("modules.result_visualizer.cv2.arrowedLine") as arrows:
 draw_count_angle_results(image,details["angle_results"],(100,100,200,200),1.5)
 assert arrows.call_count==2
 assert arrows.call_args_list[0].args[1:3]==((37,37),(52,37))
 assert arrows.call_args_list[1].args[1:3]==((97,97),(97,112))
item=dict(crop_image=np.zeros((100,100,3),np.uint8),roi_id="R",camera="CAM1",
          passed=passed,reason=reason,detect_result={"found":True,"detections":ds},
          detect_roi=(100,100,200,200),compare_roi=(100,100,200,200),**details)
with patch("modules.result_visualizer.draw_count_angle_results") as draw:
 _draw_roi_crop_for_batch(item)
 assert len(draw.call_args.args[1])==2
with patch("modules.result_gui.draw_count_angle_results") as draw:
 gui=ResultGUI()
 gui._draw_roi_crop(item)
 assert len(draw.call_args.args[1])==2
print("Overlay helper coordinate transform + saved batch + GUI: PASS")
