import torch.nn as nn

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

    def __init__(self):
        super(ECOSeg, self).__init__()

        self.backbone = build_resnet_backbone()
        self.roi_heads = build_roi_heads(self.backbone.out_channels)

    def forward(self, images, targets=None, rects=None):
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

        # proposals, proposal_losses = self.rpn(images, features, targets)
        # proposals = self.proposals_init(rects, features[0].device, images.image_sizes)
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
            boxlist = BoxList(target.bbox, target.size, mode=target.mode)
            proposals.append(boxlist)
        return proposals


