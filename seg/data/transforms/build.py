# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from . import transforms as T


def build_transforms(cfg, is_train=True):
    if is_train:
        min_size = cfg.min_size_train
        max_size = cfg.max_size_train
        flip_prob = 0.5  # cfg.INPUT.FLIP_PROB_TRAIN
        data_argument = cfg.data_argument
    else:
        min_size = cfg.min_size_test
        max_size = cfg.max_size_test
        flip_prob = 0
        data_argument = None

    to_bgr255 = cfg.to_bgr255
    normalize_transform = T.Normalize(
        mean=cfg.pixel_mean, std=cfg.pixel_std, to_bgr255=to_bgr255
    )

    transform = T.Compose(
        [
            T.Crop(cfg.search_area_scale, cfg.total_stride, cfg.pixel_mean, data_argument),
            T.Resize(min_size, max_size, cfg.total_stride),
            T.RandomHorizontalFlip(flip_prob),
            T.ToTensor(),
            normalize_transform,
        ]
    )
    return transform
