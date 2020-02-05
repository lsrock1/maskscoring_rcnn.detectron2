# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import fvcore.nn.weight_init as weight_init
import torch
from torch import nn
from torch.nn import functional as F
import pycocotools.mask as mask_utils

from detectron2.layers import Conv2d, ShapeSpec, cat, get_norm
from detectron2.utils.events import get_event_storage
from detectron2.utils.registry import Registry

ROI_MASKIOU_HEAD_REGISTRY = Registry("ROI_MASKIOU_HEAD")
ROI_MASKIOU_HEAD_REGISTRY.__doc__ = """
Registry for maskiou heads, which predicts predicted mask iou.

The registered object will be called with `obj(cfg, input_shape)`.
"""


def mask_iou_loss(labels, pred_maskiou, gt_maskiou, loss_weight):
    """
    Compute the maskiou loss.

    Args:
        labels (Tensor): Given mask labels
        pred_maskiou: Predicted maskiou
        gt_maskiou: Ground Truth IOU generated in mask head
    """
    def l2_loss(input, target):
        """
        very similar to the smooth_l1_loss from pytorch, but with
        the extra beta parameter
        """
        pos_inds = torch.nonzero(target > 0.0).squeeze(1)
        if pos_inds.shape[0] > 0:
            cond = torch.abs(input[pos_inds] - target[pos_inds])
            loss = 0.5 * cond**2 / pos_inds.shape[0]
        else:
            loss = input * 0.0
        return loss.sum()
    bg_label = pred_maskiou.shape[1]
    positive_inds = torch.nonzero(labels != bg_label).squeeze(1)
    labels_pos = labels[positive_inds]

    if labels_pos.numel() == 0:
        return pred_maskiou.sum() * 0
        
    maskiou_loss = l2_loss(pred_maskiou[positive_inds, labels_pos], gt_maskiou)
    maskiou_loss = loss_weight * maskiou_loss
    
    return maskiou_loss


def mask_iou_inference(pred_instances, pred_maskiou):
    labels = cat([i.pred_classes for i in pred_instances])
    num_masks = pred_maskiou.shape[0]
    index = torch.arange(num_masks, device=labels.device)
    maskious = pred_maskiou[index, labels]
    # maskious = [maskious]
    for maskiou, box in zip(maskious, pred_instances):
        box.scores = box.scores * maskiou


@ROI_MASKIOU_HEAD_REGISTRY.register()
class MaskIoUHead(nn.Module):
    def __init__(self, cfg):
        super(MaskIoUHead, self).__init__()
        num_classes = cfg.MODEL.ROI_HEADS.NUM_CLASSES
        input_channels = 257 

        self.maskiou_fcn1 = Conv2d(input_channels, 256, 3, 1, 1) 
        self.maskiou_fcn2 = Conv2d(256, 256, 3, 1, 1) 
        self.maskiou_fcn3 = Conv2d(256, 256, 3, 1, 1) 
        self.maskiou_fcn4 = Conv2d(256, 256, 3, 2, 1) 
        self.maskiou_fc1 = nn.Linear(256*7*7, 1024)
        self.maskiou_fc2 = nn.Linear(1024, 1024)
        self.maskiou = nn.Linear(1024, num_classes)

        for l in [self.maskiou_fcn1, self.maskiou_fcn2, self.maskiou_fcn3, self.maskiou_fcn4]:
            nn.init.kaiming_normal_(l.weight, mode="fan_out", nonlinearity="relu")
            nn.init.constant_(l.bias, 0)

        for l in [self.maskiou_fc1, self.maskiou_fc2]:
            nn.init.kaiming_uniform_(l.weight, a=1)
            nn.init.constant_(l.bias, 0)

        nn.init.normal_(self.maskiou.weight, mean=0, std=0.01)
        nn.init.constant_(self.maskiou.bias, 0)

    def forward(self, x, mask):
        mask_pool = F.max_pool2d(mask, kernel_size=2, stride=2)
        x = torch.cat((x, mask_pool), 1)
        x = F.relu(self.maskiou_fcn1(x))
        x = F.relu(self.maskiou_fcn2(x))
        x = F.relu(self.maskiou_fcn3(x))
        x = F.relu(self.maskiou_fcn4(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.maskiou_fc1(x))
        x = F.relu(self.maskiou_fc2(x))
        x = self.maskiou(x)
        return x


def build_maskiou_head(cfg):
    """
    Build a mask iou head defined by `cfg.MODEL.ROI_MASKIOU_HEAD.NAME`.
    """
    name = cfg.MODEL.ROI_MASKIOU_HEAD.NAME
    return ROI_MASKIOU_HEAD_REGISTRY.get(name)(cfg)
