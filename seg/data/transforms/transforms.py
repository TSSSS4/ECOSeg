# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import random

import torch
import torchvision
from torchvision.transforms import functional as F


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string


class Crop(object):
    def __init__(self, search_scale, stride):
        assert search_scale >= 1
        self.search_scale = search_scale
        self.stride = stride

    # modified from torchvision to add support for max size
    def get_box(self, box):
        target_w = box[2] - box[0]
        target_h = box[3] - box[1]
        sample_sz = ((target_h * target_w) ** 0.5) * self.search_scale
        feat_shape = sample_sz // self.stride
        desired_sz = feat_shape + 1 + feat_shape % 2
        sample_sz = desired_sz * self.stride

        target_cx = box[0] + target_w / 2
        target_cy = box[1] + target_h / 2
        sample_sz_half = sample_sz / 2
        x0 = target_cx - sample_sz_half
        y0 = target_cy - sample_sz_half
        x1 = target_cx + sample_sz_half
        y1 = target_cy + sample_sz_half

        return x0, y0, x1, y1

    def __call__(self, image, target):
        x0, y0, x1, y1 = self.get_box(target.bbox[0].numpy())
        image = F.crop(image, y0, x0, y1-y0, x1-x0)
        target = target.crop([x0, y0, x1, y1])
        return image, target


class Resize(object):
    def __init__(self, min_size, max_size, stride):
        self.min_size = min_size
        self.max_size = max_size
        self.stride = stride

    # modified from torchvision to add support for max size
    def get_size(self, image_size):
        w, h = image_size
        assert w == h
        size = w

        while size < self.min_size:
            size += 2 * self.stride
        while size > self.max_size:
            size -= 2 * self.stride

        return (size, size)

    def __call__(self, image, target):
        size = self.get_size(image.size)
        image = F.resize(image, size)
        target = target.resize(image.size)
        return image, target


class RandomHorizontalFlip(object):
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            image = F.hflip(image)
            target = target.transpose(0)
        return image, target


class ToTensor(object):
    def __call__(self, image, target):
        return F.to_tensor(image), target


class Normalize(object):
    def __init__(self, mean, std, to_bgr255=True):
        self.mean = mean
        self.std = std
        self.to_bgr255 = to_bgr255

    def __call__(self, image, target):
        if self.to_bgr255:
            image = image[[2, 1, 0]] * 255
        image = F.normalize(image, mean=self.mean, std=self.std)
        return image, target
