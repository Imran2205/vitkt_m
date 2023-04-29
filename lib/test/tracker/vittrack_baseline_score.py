from lib.test.tracker.basetracker import BaseTracker
import torch
from lib.test.tracker.vittrack_utils import sample_target
# for debug
import cv2
import os
from lib.utils.merge import merge_template_search
from lib.models.vittrack import build_vittrack_baseline_score
from lib.test.tracker.vittrack_utils import Preprocessor
from lib.utils.box_ops import clip_box


class VITTRACK_BASELINE_SCORE(BaseTracker):
    def __init__(self, params, dataset_name):
        super(VITTRACK_BASELINE_SCORE, self).__init__(params)
        network = build_vittrack_baseline_score(params.cfg)
        network.load_state_dict(torch.load(self.params.checkpoint, map_location='cpu')['net'], strict=True)
        self.cfg = params.cfg
        self.network = network.cuda()
        self.network.eval()
        self.preprocessor = Preprocessor()
        self.state = None
        # for debug
        # self.debug = False
        self.debug = params.debug
        self.frame_id = 0
        if self.debug == 2:
            self.save_dir = "debug"
            if not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir)
        # for save boxes from all queries
        self.save_all_boxes = params.save_all_boxes
        self.z_dict1 = {}

    def initialize(self, image, info: dict):
        # forward the template once
        # info['init_bbox']: list [x0,y0,w,h] example: [367.0, 101.0, 41.0, 16.0]
        z_patch_arr, _ = sample_target(image, info['init_bbox'], self.params.template_factor,
                                       output_sz=self.params.template_size)

        self.template = self.preprocessor.process(z_patch_arr)
        self.template0=self.template.clone()
        # with torch.no_grad():
        #     self.z_dict1 = self.network.forward_backbone(template)
        # save states
        self.state = info['init_bbox']
        self.frame_id = 0
        if self.save_all_boxes:
            '''save all predicted boxes'''
            all_boxes_save = info['init_bbox'] * self.cfg.MODEL.NUM_OBJECT_QUERIES
            return {"all_boxes": all_boxes_save}

    def updateTemplate(self,image,update_interval,update):
        if self.frame_id % update_interval == 0 and update:
            z_patch_arr, _ = sample_target(image, self.state, self.params.template_factor,
                                           output_sz=self.params.template_size)
            self.template = self.preprocessor.process(z_patch_arr)
    def updateTemplate2(self,image,update_interval,update):
        if self.frame_id % update_interval == 0 and update:
            z_patch_arr, _ = sample_target(image, self.state, self.params.template_factor,
                                           output_sz=self.params.template_size)
            template = self.preprocessor.process(z_patch_arr)
            self.template=0.5*self.template0+0.5*template
    def updateTemplate3(self,image,update_interval,update):
        if self.frame_id % update_interval == 0 and update:
            z_patch_arr, _ = sample_target(image, self.state, self.params.template_factor,
                                           output_sz=self.params.template_size)
            template = self.preprocessor.process(z_patch_arr)
            self.template=0.5*self.template0+0.25*template+0.25*self.template

    def track(self, image, info: dict = None):
        H, W, _ = image.shape
        self.frame_id += 1
        x_patch_arr, resize_factor = sample_target(image, self.state, self.params.search_factor,
                                                                output_sz=self.params.search_size)  # (x1, y1, w, h)
        search = self.preprocessor.process(x_patch_arr)
        with torch.no_grad():
            xz = self.network.forward_backbone(search, self.template)
            # run the transformer
            out_dict, _, _ = self.network.forward_transformer(xz=xz,run_box_head=True, run_cls_head=True)

        pred_boxes = out_dict['pred_boxes'].view(-1, 4)
        # Baseline: Take the mean of all pred boxes as the final result
        pred_box = (pred_boxes.mean(dim=0) * self.params.search_size / resize_factor).tolist()  # (cx, cy, w, h) [0,1]
        # get the final box result
        self.state = clip_box(self.map_box_back(pred_box, resize_factor), H, W, margin=10)
        # self.state: list [x0,y0,w,h,] example: [365.4537048339844, 102.24719142913818, 47.13159942626953, 15.523386001586914]
        #conf_score = out_dict["pred_logits"].view(-1).item()
        conf_score = out_dict["pred_logits"].view(-1).sigmoid().item()
        # for debug
        if self.debug == 2:
            x1, y1, w, h = self.state
            image_BGR = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.rectangle(image_BGR, (int(x1),int(y1)), (int(x1+w),int(y1+h)), color=(0,0,255), thickness=2)
            save_path = os.path.join(self.save_dir, "%04d.jpg" % self.frame_id)
            cv2.imwrite(save_path, image_BGR)
        elif self.debug == 1:
            x1, y1, w, h = self.state
            image_BGR = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.rectangle(image_BGR, (int(x1),int(y1)), (int(x1+w),int(y1+h)), color=(0,0,255), thickness=2)
            cv2.imshow('vis', image_BGR)
            cv2.waitKey(1)
            # if cv2.waitKey() & 0xFF == ord('q'):
            #     pass
        if self.save_all_boxes:
            '''save all predictions'''
            all_boxes = self.map_box_back_batch(pred_boxes * self.params.search_size / resize_factor, resize_factor)
            all_boxes_save = all_boxes.view(-1).tolist()  # (4N, )
            return {"target_bbox": self.state,
                    "all_boxes": all_boxes_save,
                    "conf_score": conf_score}
        else:
            return {"target_bbox": self.state,
                    "conf_score": conf_score}

    def map_box_back(self, pred_box: list, resize_factor: float):
        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box
        half_side = 0.5 * self.params.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return [cx_real - 0.5 * w, cy_real - 0.5 * h, w, h]

    def map_box_back_batch(self, pred_box: torch.Tensor, resize_factor: float):
        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box.unbind(-1) # (N,4) --> (N,)
        half_side = 0.5 * self.params.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return torch.stack([cx_real - 0.5 * w, cy_real - 0.5 * h, w, h], dim=-1)


def get_tracker_class():
    return VITTRACK_BASELINE_SCORE
