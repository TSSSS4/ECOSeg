import torch
import torch.nn as nn
import numpy as np

from seg.structures.backbone import build_resnet_backbone
from seg.structures.roi_head import build_roi_heads
from .utils.image_list import to_image_list
from .utils.bounding_box import BoxList
# from eco.config import config


class ECOSeg(nn.Module):
    """
    Main class for Generalized R-CNN. Currently supports boxes and masks.
    It consists of three main parts:
    - backbone
    - rpn
    - heads: takes the features + the proposals from the RPN and computes
        detections / masks from it.
    """

    def __init__(self, config):
        super(ECOSeg, self).__init__()

        self.backbone = build_resnet_backbone()
        self.roi_heads = build_roi_heads(self.backbone.out_channels)
        # Data Argument
        self.min_scale = config.min_scale
        self.max_scale = config.max_scale
        self.scale_var = config.max_scale - config.min_scale
        self.scale_num = config.scale_num
        self.pos_var = config.pos_var
        self.pos_num = config.pos_num

    def forward(self, images, targets=None, proposals=None):
        """
        Arguments:
            images (list[Tensor] or ImageList): images to be processed
            targets (list[BoxList]): ground-truth boxes present in the image (optional)

        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).

        """
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")
        images = to_image_list(images)
        features = self.backbone(images.tensors)

        # proposals = self.proposals_cal(targets)
        proposals, targets = self.proposals_targets_cal(targets)

        x, result, detector_losses = self.roi_heads(features, proposals, targets)   # ROI HEAD, output mask

        if self.training:
            losses = {}
            losses.update(detector_losses)
            return losses

        return result

    def proposals_targets_cal(self, targets):
        proposals = []
        for idx, target in enumerate(targets):
            # if self.training:
            #     target.bbox[0] = self.argument(target.bbox[0], target.size)

            bboxes, target = self.argument(target)
            boxlist = BoxList(bboxes, target.size, mode=target.mode).to(target.bbox.device)
            proposals.append(boxlist)
            targets[idx] = target
        return proposals, targets

    def argument(self, target):
        target_bboxes = target.bbox
        image_size = target.size
        bboxes = []
        target_idxes = []
        idx = 0
        for target_bbox in target_bboxes:
            x0, y0, x1, y1 = target_bbox
            max_x, max_y = image_size
            cx = (x0 + x1) / 2
            cy = (y0 + y1) / 2
            w = x1 - x0
            h = y1 - y0
            scales = np.random.rand(self.scale_num) * self.scale_var + self.min_scale
            poses = np.random.rand(self.pos_var) * self.pos_var
            for scale in scales:
                for pos in poses:
                    cx1 = cx + np.random.randint(-1, 2) * pos
                    cy1 = cy + np.random.randint(-1, 2) * pos
                    w1 = w * scale
                    h1 = h * scale
                    bbox = []
                    bbox.append(max(0, cx1 - w1 / 2))
                    bbox.append(max(0, cy1 - h1 / 2))
                    bbox.append(min(max_x, cx1 + w1 / 2))
                    bbox.append(min(max_y, cy1 + h1 / 2))
                    bboxes.append(bbox)
                    target_idxes.append(idx)
            idx += 1
        return torch.tensor(bboxes), target[target_idxes]



