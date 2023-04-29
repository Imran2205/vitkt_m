import os
import sys
import cv2
import vot
from vot_path import base_path
if base_path not in sys.path:
    sys.path.append(base_path)
    sys.path.append(os.path.join(base_path, 'utils'))
from vitkt_m import vitTrack_Tracker
import numpy as np

class p_config(object):
    score_thrs=0.7
    update_score=0.8
    ctdis=0.55
# test DiMPMU
p = p_config()

handle = vot.VOT("rectangle")

selection = handle.region()
imagefile = handle.frame()

image = cv2.cvtColor(cv2.imread(imagefile), cv2.COLOR_BGR2RGB)
tracker = vitTrack_Tracker(image, selection, p=p)

while True:
    imagefile = handle.frame()
    if not imagefile:
        break
    image = cv2.cvtColor(cv2.imread(imagefile), cv2.COLOR_BGR2RGB)
    region, score,ap_dis,lof_dis= tracker.tracking(image)
    handle.report(vot.Rectangle(float(region[0]), float(region[1]), float(region[2]),
                float(region[3])),score)
