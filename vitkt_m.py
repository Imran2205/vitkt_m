import cv2
import os
import torch
import numpy as np
import sys
from vot_path import base_path
sys.path.append(os.path.join(base_path,'meta_updater'))
sys.path.append(os.path.join(base_path,'motion'))
sys.path.append(os.path.join(base_path,'utils/metric_net'))
from metric_model import ft_net
import ltr.admin.loading as ltr_loading
from torch.autograd import Variable
from me_sample_generator import *

env_path = os.path.join(os.path.dirname(__file__))
if env_path not in sys.path:
    sys.path.append(env_path)
from lib.test.evaluation.tracker import Tracker_stark
from pytracking.evaluation import Tracker

from PIL import Image
Image.MAX_IMAGE_PIXELS = 1000000000
from tracking_utils import compute_iou, show_res, process_regions

from sklearn.neighbors import LocalOutlierFactor


import random

seed = 2022
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

def lof_fit(data,k=5,method='l2'):
    clf = LocalOutlierFactor(n_neighbors=k + 1, algorithm='auto', metric=method, contamination=0.1)
    clf.fit(data)
    return clf
def lof(predict, clf,k=5, method='l2',thresh=2):
    # 计算 LOF 离群因子
    predict = -clf._score_samples(predict)
    # predict=predict[200:]
    # 根据阈值划分离群点与正常点
    result=predict<=thresh
    return predict,result
def convert_bbox(offset,h,w):
    #[x1,y1,x2,y2] to [x1,y1,w,h]
    bbox=offset.copy()
    bbox[:,0]=offset[:,0]* w
    bbox[:, 1] = offset[:, 1] * h
    bbox[:, 2] = offset[:,2]* w
    bbox[:, 3] = offset[:,3]* h
    bbox[:, 2]-=bbox[:, 0]
    bbox[:, 3] -= bbox[:, 1]
    return bbox

class vitTrack_Tracker(object):
    def __init__(self, image, region, p=None, groundtruth=None):

        self.p = p
        self.i = 0
        self.fail_num=0
        self.globalmode = True
        if groundtruth is not None:
            self.groundtruth = groundtruth
        else:
            self.groundtruth=None

        init_gt1 = [region.x, region.y, region.width, region.height]
        init_gt = [init_gt1[1], init_gt1[0], init_gt1[1]+init_gt1[3], init_gt1[0]+init_gt1[2]] # ymin xmin ymax xmax

        self.last_gt = init_gt

        self.local_init(image, init_gt1)
        self.keepTrack_init(image, init_gt1)
        self.metric_init(image, np.array(init_gt1))
        self.motion_init(image, init_gt1)

        self.cy=(self.last_gt[0] + self.last_gt[2] - 1) / 2
        self.cx=(self.last_gt[1] + self.last_gt[3] - 1) / 2
        self.width = self.last_gt[3] - self.last_gt[1]
        self.height = self.last_gt[2] - self.last_gt[0]

    def keepTrack_init(self, image, init_bbox):
        local_tracker = Tracker('keep_track', 'default')
        params = local_tracker.get_parameters()

        debug_ = getattr(params, 'debug', 0)
        params.debug = debug_

        params.tracker_name = local_tracker.name
        params.param_name = local_tracker.parameter_name

        self.keepTracker = local_tracker.tracker_class(params)
        init_box = dict()
        init_box['init_bbox'] = init_bbox
        self.keepTracker.initialize(image, init_box)
    def keepTrack_eval(self, image):
        state, score = self.keepTracker.track(image)
        local_state = np.array(state).reshape((1, 4))
        ap_dis, lof_dis,_ = self.metric_eval(image, local_state)
        return state, score,ap_dis, lof_dis
    def motion_init(self,img,bbox):
        path = os.path.join(base_path,'checkpoints/motion_v1_ep0046.pth.tar')
        self.motion_model, _ = ltr_loading.load_network(path)
        self.motion_model = self.motion_model.cuda()
        self.motion_model.eval()

        tmp = np.random.rand(1, 30, 4)
        tmp = (Variable(torch.Tensor(tmp))).type(torch.FloatTensor).cuda()
        self.motion_model(tmp)

        self.h=img.shape[0]
        self.w=img.shape[1]
        self.motion_input=[]
        self.motion_input.append([bbox[0]/self.w,bbox[1]/self.h,(bbox[0]+bbox[2])/self.w,(bbox[1]+bbox[3])/self.h])
    def motion_eval(self,image):
        motion_input=np.array(self.motion_input)
        motion_input = motion_input[np.newaxis, :]
        motion_input = torch.tensor(motion_input, dtype=torch.float, device='cuda')
        offset = self.motion_model(motion_input)
        offset = offset.detach().cpu().numpy().reshape(-1, 4)
        bbox = convert_bbox(offset, self.h, self.w)
        bbox=bbox.squeeze()
        if bbox[0]<=0 or bbox[0]>=self.w or bbox[1]<=0 or bbox[1]>=self.h or \
                bbox[2]<=3 or (bbox[2]+bbox[0])>=self.w or bbox[3]<=3 or (bbox[3]+bbox[1])>=self.h:
            return bbox,100,100
        ap_dis, lof_dis,_ = self.metric_eval(image, bbox.reshape((1, 4)))
        return bbox,ap_dis,lof_dis

    def metric_init(self, im, init_box):
        self.metric_model = ft_net(class_num=1120)
        path = os.path.join(base_path,'checkpoints/metric_model_zj_57470.pt')
        self.metric_model.eval()
        self.metric_model = self.metric_model.cuda()
        self.metric_model.load_state_dict(torch.load(path))
        tmp = np.random.rand(1, 3, 107, 107)
        tmp = (Variable(torch.Tensor(tmp))).type(torch.FloatTensor).cuda()
        # get target feature
        self.metric_model(tmp)
        init_box = init_box.reshape((1, 4))
        with torch.no_grad():
            self.anchor_feature=self.get_metric_feature(im,init_box)

        pos_generator = SampleGenerator('gaussian', np.array([im.shape[1], im.shape[0]]), 0.1, 1.3)
        gt_pos_examples = pos_generator(init_box[0].astype(np.float32), 20, [0.7, 1])
        gt_iou = 0.7
        print(gt_pos_examples.shape[0])
        while gt_pos_examples.shape[0] < 10:
            gt_iou = gt_iou - 0.05
            gt_pos_examples = pos_generator(init_box[0].astype(np.float32), 20, [gt_iou, 1])
            print(gt_pos_examples.shape[0])
        with torch.no_grad():
            self.gt_pos_features=self.get_metric_feature(im,gt_pos_examples).cpu().detach().numpy()

        self.clf = lof_fit(self.gt_pos_features, k=5)
        self.lof_thresh = 2.5#2.1
    def get_metric_feature(self,im,box):
        anchor_region = me_extract_regions(im, box)
        anchor_region = process_regions(anchor_region)
        anchor_region = torch.Tensor(anchor_region)
        anchor_region = (Variable(anchor_region)).type(torch.FloatTensor).cuda()
        anchor_feature, _ = self.metric_model(anchor_region)
        return anchor_feature

    def metric_eval(self, im, boxes,single=True):
        box_features = self.get_metric_feature(np.array(im), boxes)
        ap_dist = torch.norm(self.anchor_feature - box_features, 2, dim=1).view(-1)
        lof_score, _ = lof(box_features.cpu().detach().numpy(), self.clf, k=5, thresh=self.lof_thresh)
        ap_dist=ap_dist.data.cpu().numpy()
        if single:
            return ap_dist[0],lof_score[0],box_features
        else:
            return ap_dist, lof_score,box_features

    def local_init(self, image, init_bbox):
        local_tracker = Tracker_stark('vittrack_baseline_score', 'baseline',"vot22")
        params = local_tracker.get_parameters()

        params.visualization = False
        params.debug = False

        self.local_Tracker = local_tracker.create_tracker(params)
        init_box = dict()
        init_box['init_bbox'] = init_bbox
        self.local_Tracker.initialize(image, init_box)

    def local_track(self, image):
        outputs = self.local_Tracker.track(image)
        pred_bbox = outputs['target_bbox']
        score=outputs['conf_score']
        self.last_gt = [pred_bbox[1], pred_bbox[0], pred_bbox[1]+pred_bbox[3],pred_bbox[0]+pred_bbox[2]]
        local_state = np.array(pred_bbox).reshape((1, 4))
        ap_dis,lof_dis,metric_feature = self.metric_eval(image, local_state)

        return pred_bbox, score,ap_dis,lof_dis,metric_feature

    def tracking(self, image):
        self.i += 1
        local_state1, score,ap_dis,lof_dis,metric_feature = self.local_track(image)
        kt_state, kt_score, kt_ap_dis, kt_lof_dis = self.keepTrack_eval(image)
        use_kt=0
        cx=local_state1[0]+local_state1[2]/2
        cy = local_state1[1] + local_state1[3] / 2
        ct_dis=np.sqrt(np.power((cx-self.cx)/self.width,2)+np.power((cy-self.cy)/self.height,2))

        if score>self.p.update_score and kt_score>self.p.update_score:
            self.anchor_feature=metric_feature
        if score<self.p.score_thrs and (kt_score>self.p.score_thrs or (kt_ap_dis<ap_dis or kt_lof_dis<lof_dis)):
            use_kt=1
            self.last_gt = [kt_state[1], kt_state[0], kt_state[1] + kt_state[3],
                                     kt_state[2] + kt_state[0]]
            self.local_Tracker.state = kt_state
            score=kt_score
            ap_dis = kt_ap_dis
            lof_dis = kt_lof_dis
            cx = kt_state[0] + kt_state[2] / 2
            cy = kt_state[1] + kt_state[3] / 2
            ct_dis=np.sqrt(np.power((cx-self.cx)/self.width,2)+np.power((cy-self.cy)/self.height,2))
        elif self.i>=30 and (score<self.p.score_thrs and ct_dis>self.p.ctdis):#####<0.5(0.26);<0.3(0.31)
            #fail, trigger motion_track
            motion_bbox,motion_ap_dis,motion_lof_dis=self.motion_eval(image)
            if (motion_ap_dis<ap_dis or motion_lof_dis<lof_dis):#####change to and
                self.last_gt=[motion_bbox[1], motion_bbox[0], motion_bbox[1]+motion_bbox[3], motion_bbox[0]+motion_bbox[2]]
                self.local_Tracker.state=list(motion_bbox)
                ap_dis=motion_ap_dis
                lof_dis=motion_lof_dis
                cx = motion_bbox[0] + motion_bbox[2] / 2
                cy = motion_bbox[1] + motion_bbox[3] / 2
        # print(self.i,score,ct_dis)
        if not use_kt:
            self.keepTracker.pos = torch.FloatTensor(
                [(self.last_gt[0] + self.last_gt[2] - 1) / 2, (self.last_gt[1] + self.last_gt[3] - 1) / 2])
            self.keepTracker.target_sz = torch.FloatTensor(
                [(self.last_gt[2] - self.last_gt[0]), (self.last_gt[3] - self.last_gt[1])])

        if self.i>=30:
            self.motion_input.pop(0)
        self.motion_input.append([self.last_gt[1]/self.w,self.last_gt[0]/self.h,self.last_gt[3]/self.w,self.last_gt[2]/self.h])


        self.width = self.last_gt[3] - self.last_gt[1]
        self.height = self.last_gt[2] - self.last_gt[0]
        self.cy=cy
        self.cx=cx
        return [float(self.last_gt[1]), float(self.last_gt[0]), float(self.width),
                float(self.height)], score,ap_dis,lof_dis

