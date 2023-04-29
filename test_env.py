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

imagefile=os.path.join(base_path,'data_test/00000001.jpg')
image = cv2.cvtColor(cv2.imread(imagefile), cv2.COLOR_BGR2RGB)
tracker = vitTrack_Tracker(image, vot.Rectangle(601.0,318.0,213.0,95.0), p=p)
_=tracker.tracking(image)
print('Done!')
