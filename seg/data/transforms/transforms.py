# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import random

import torch
import torchvision
from torchvision.transforms import functional as F
from PIL import Image
import numpy as np

from seg.utils.bounding_box import BoxList


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
    def __init__(self, search_scale, stride, pixel_mean, data_argument=None):
        assert search_scale >= 1
        self.search_scale = search_scale
        self.stride = stride
        self.pixel_mean = tuple(map(int, map(round, pixel_mean[::-1])))
        self.mode = 'RGB'
        # data argumentation
        if data_argument:
            self.training = True
            self.max_scale = data_argument.max_scale
            self.min_scale = data_argument.min_scale
            self.scale_var = self.max_scale - self.min_scale
            self.pos_var = data_argument.pos_var
        else:
            self.training = False

    # modified from torchvision to add support for max size
    def get_box(self, box):
        target_w = box[2] - box[0]
        target_h = box[3] - box[1]
        sample_sz = ((target_h * target_w) ** 0.5) * self.search_scale
        target_cx = box[0] + target_w / 2
        target_cy = box[1] + target_h / 2

        sample_sz_half = sample_sz / 2
        x0 = target_cx - sample_sz_half
        y0 = target_cy - sample_sz_half
        x1 = target_cx + sample_sz_half
        y1 = target_cy + sample_sz_half

        return x0, y0, x1, y1

    def crop(self, image, y0, x0, h, w):
        left = int(round(-x0))
        up = int(round(-y0))
        new_image = Image.new(mode=self.mode, size=(int(round(w)), int(round(h))), color=tuple(self.pixel_mean))
        new_image.paste(image, (left, up))
        return new_image

    def argument(self, bbox):
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        cx = bbox[0] + w / 2
        cy = bbox[1] + h / 2
        # Data Argument
        pos_var = np.random.rand() * self.pos_var
        cx = cx + np.random.randint(-1, 2) * pos_var  # pos_var: [0, self.pos_var=2]
        cy = cy + np.random.randint(-1, 2) * pos_var  # cx_offset: {-1, 0, 1} * pos_var
        w = w * (np.random.rand() * self.scale_var + self.min_scale)
        h = h * (np.random.rand() * self.scale_var + self.min_scale)
        return [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]

    def __call__(self, image, target):
        bbox = target.bbox[0].numpy()
        if self.training:
            bbox_argument = self.argument(bbox)
            x0, y0, x1, y1 = self.get_box(bbox_argument)
            proposal = BoxList(torch.tensor([bbox_argument]), target.size, mode=target.mode)
            target.add_field('proposal', proposal)
        else:
            x0, y0, x1, y1 = self.get_box(bbox)
        image = self.crop(image, y0, x0, y1 - y0 + 1, x1 - x0 + 1)
        # This crop of BoxList will operate iteratively, thus there is no need to crop proposal again.
        # Code of Resize and Transpose doesn't need to modify too.
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

        feat_shape = size // self.stride
        desired_sz = feat_shape + 1 + feat_shape % 2
        size = desired_sz * self.stride

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
