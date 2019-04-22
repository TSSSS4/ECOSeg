import torch
import torch.nn as nn
import time

from eco.config import config
from .box_head import ROIBoxHead
from .mask_head import ROIMaskHead


class CombinedROIHeads(nn.ModuleDict):
    """
    Combines a set of individual heads (for box prediction or masks) into a single
    head.
    """

    def __init__(self, heads):
        super(CombinedROIHeads, self).__init__(heads)

    def forward(self, features, proposals, targets=None, metric=None):
        losses = {}
        # TODO rename x to roi_box_features, if it doesn't increase memory consumption

        mask_features = features
        # optimization: during training, if we share the feature extractor between
        # the box and the mask heads, then we can reuse the features already computed

        # During training, self.box() will return the unaltered proposals as "detections"
        # this makes the API consistent during training and testing
        # x: roi mask features [n,256,14,14], detection: cls and reg results and masks, loss box:empty for inference
        if metric:
            start = time.time()
        x, detections, loss_mask = self.mask(mask_features, proposals, targets)
        if metric:
            metric.mask['head_mask'] += time.time() - start
        losses.update(loss_mask)

        return x, detections, losses


def build_roi_heads(in_channels):
    # individually create the heads, that will be combined together
    # afterwards
    roi_heads = []
    roi_heads.append(("mask", ROIMaskHead(in_channels)))

    # combine individual heads in a single module
    if roi_heads:
        roi_heads = CombinedROIHeads(roi_heads)

    return roi_heads

