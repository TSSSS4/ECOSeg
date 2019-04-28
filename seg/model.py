import torch.nn as nn
import numpy as np

from seg.structures.backbone import build_resnet_backbone
from seg.structures.roi_head import build_roi_heads
from .utils.image_list import to_image_list
from .utils.bounding_box import BoxList


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

        proposals = self.proposals_cal(targets)

        x, result, detector_losses = self.roi_heads(features, proposals, targets)   # ROI HEAD, output mask

        if self.training:
            losses = {}
            losses.update(detector_losses)
            return losses

        return result

    def proposals_cal(self, targets):
        proposals = []
        for target in targets:
            # if self.training:
            #     target.bbox[0] = self.argument(target.bbox[0], target.size)
            # boxlist = BoxList(target.bbox, target.size, mode=target.mode)
            # proposals.append(boxlist)
            proposals.append(target.get_field('proposal'))
            target.remove_field('proposal')
        return proposals

    # def argument(self, bbox, image_size):
    #     scale = np.random.rand() * self.scale_var + self.min_scale
    #     x0, y0, x1, y1 = bbox
    #     max_x, max_y = image_size
    #     cx = (x0 + x1) / 2
    #     cy = (y0 + y1) / 2
    #     w = x1 - x0
    #     h = y1 - y0
    #     # pos argument
    #     pos_var = np.random.rand() * self.pos_var       # pos_var: [0, self.pos_var=2]
    #     cx = cx + np.random.randint(-1, 2) * pos_var    # cx_offset: {-1, 0, 1} * pos_var
    #     cy = cy + np.random.randint(-1, 2) * pos_var
    #     # scale argument
    #     w = w * scale
    #     h = h * scale
    #     bbox[0] = max(0, cx - w / 2)
    #     bbox[1] = max(0, cy - h / 2)
    #     bbox[2] = min(max_x, cx + w / 2)
    #     bbox[3] = min(max_y, cy + h / 2)
    #     return bbox



