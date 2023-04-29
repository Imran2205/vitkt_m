from . import BaseActor
from lib.utils.misc import NestedTensor
from lib.utils.box_ops import box_cxcywh_to_xyxy, box_xywh_to_xyxy
import torch
from lib.utils.merge import merge_template_search
import numpy as np
def label_function(x):
    a=x-0.5
    return 1/(1+np.exp(-10*a))
def cxcy2xyxy(bboxes):
    bbox_new=bboxes.copy()
    bbox_new[:,0]-=bboxes[:,2]/2
    bbox_new[:, 1] -= bboxes[:, 3] / 2
    bbox_new[:, 2]=bboxes[:, 0]+bboxes[:,2]/2
    bbox_new[:, 3] = bboxes[:, 1] + bboxes[:, 3] / 2
    return bbox_new
def bboxes_iou(bboxes1, bboxes2):
    # 计算两个box的交集：交集左上角的点取两个box的max，交集右下角的点取两个box的min
    int_ymin = np.maximum(bboxes1[:, 0], bboxes2[:, 0])
    int_xmin = np.maximum(bboxes1[:, 1], bboxes2[:, 1])
    int_ymax = np.minimum(bboxes1[:, 2], bboxes2[:, 2])
    int_xmax = np.minimum(bboxes1[:, 3], bboxes2[:, 3])

    # 计算两个box交集的wh：如果两个box没有交集，那么wh为0(按照计算方式wh为负数，跟0比较取最大值)
    int_h = np.maximum(int_ymax - int_ymin, 0.)
    int_w = np.maximum(int_xmax - int_xmin, 0.)

    # 计算IOU
    int_vol = np.multiply(int_h, int_w)  # 交集面积
    vol1 = np.multiply(bboxes1[:, 2] - bboxes1[:, 0], bboxes1[:, 3] - bboxes1[:, 1])  # bboxes1面积
    vol2 = np.multiply(bboxes2[:, 2] - bboxes2[:, 0], bboxes2[:, 3] - bboxes2[:, 1])  # bboxes2面积
    iou = (int_vol + 1e-8) / (vol1 + vol2 - int_vol + 1e-8)  # IOU=交集/并集
    label = label_function(iou)
    # iou = np.expand_dims(iou, axis=2)
    return label,iou

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
        gt_bboxes=data['search_anno'].squeeze().data.cpu().numpy()
        pred_bboxes = out_dict['pred_boxes'].squeeze().data.cpu().numpy()
        gt_bboxes[:, 2:] += gt_bboxes[:, :2]
        pred_bboxes = cxcy2xyxy(pred_bboxes)
        cls_label, iou = bboxes_iou(pred_bboxes, gt_bboxes)
        cls_label = torch.tensor(cls_label, dtype=torch.float, device='cuda')
        # compute losses
        loss, status = self.compute_losses(out_dict, cls_label,iou)

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

    def compute_losses(self, pred_dict, labels, iou, return_status=True):
        loss = self.loss_weight["cls"] * self.objective['cls'](pred_dict["pred_logits"].view(-1), labels)
        label = labels.data.cpu().numpy()
        if return_status:
            # status for log
            status = {
                "cls_loss": loss.item(),
                "iou_min": iou.min(),
                "iou_mean": iou.mean(),
                "iou_max": iou.max(),
                "label_min": label.min(),
                "label_mean": label.mean(),
                "label_max": label.max()}
            return loss, status
        else:
            return loss
