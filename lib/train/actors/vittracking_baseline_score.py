from . import BaseActor
from lib.utils.misc import NestedTensor
from lib.utils.box_ops import box_cxcywh_to_xyxy, box_xywh_to_xyxy
import torch
from lib.utils.merge import merge_template_search
import numpy as np

class VitTrackingBaselineScoreActor(BaseActor):
    """ Actor for training the STARK-S and STARK-ST(Stage1)"""
    def __init__(self, net, objective, loss_weight, settings):
        super().__init__(net, objective)
        self.loss_weight = loss_weight
        self.settings = settings
        self.bs = self.settings.batchsize  # batch size

    def __call__(self, data):
        """
        args:
            data - The input data, should contain the fields 'template', 'search', 'gt_bbox'.
            template_images: (N_t, batch, 3, H, W)
            search_images: (N_s, batch, 3, H, W)
        returns:
            loss    - the training loss
            status  -  dict containing detailed losses
        """
        # forward pass
        out_dict = self.forward_pass(data, run_box_head=True, run_cls_head=True)

        # process the groundtruth
        labels = data['label'].view(-1)  # (batch, ) 0 or 1
        # compute losses
        loss, status = self.compute_losses(out_dict, labels)

        return loss, status

    def forward_pass(self, data, run_box_head, run_cls_head):
        # process the templates
        template_img = data['template_images'][-1].view(-1, *data['template_images'].shape[2:])  # (batch, 3, 192, 192)

        # process the search regions (t-th frame)
        search_img = data['search_images'].view(-1, *data['search_images'].shape[2:])  # (batch, 3, 384, 384)

        feature_xz = self.net(im_x=search_img, im_z=template_img, mode='backbone')

        out_dict, _, _ = self.net(xz=feature_xz, mode="transformer", run_box_head=run_box_head, run_cls_head=run_cls_head)
        # out_dict: (B, N, C), outputs_coord: (1, B, N, C), target_query: (1, B, N, C)
        return out_dict

    def compute_losses(self, pred_dict, labels, return_status=True):
        loss = self.loss_weight["cls"] * self.objective['cls'](pred_dict["pred_logits"].view(-1), labels)
        if return_status:
            # status for log
            status = {
                "cls_loss": loss.item()}
            return loss, status
        else:
            return loss
