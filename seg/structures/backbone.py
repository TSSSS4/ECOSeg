from collections import OrderedDict

import torch.nn as nn

from .resnet import ResNet
from .fpn import FPN, LastLevelMaxPool
from .layer import conv_with_kaiming_uniform


def build_resnet_fpn_backbone():
    body = ResNet()
    in_channels_stage2 = 256
    out_channels = 256
    fpn = FPN(
        in_channels_list=[
            in_channels_stage2,
            in_channels_stage2 * 2,
            in_channels_stage2 * 4,
            in_channels_stage2 * 8,
        ],
        out_channels=out_channels,
        conv_block=conv_with_kaiming_uniform(
            False, False
        ),
        top_blocks=LastLevelMaxPool(),
    )
    model = nn.Sequential(OrderedDict([("body", body), ("fpn", fpn)]))
    model.out_channels = out_channels
    return model


def build_resnet_backbone():
    body = ResNet()
    model = nn.Sequential(OrderedDict([("body", body)]))
    model.out_channels = 1024
    return model

