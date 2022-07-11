#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2022 Milan Ondrašovič <milan.ondrasovic@gmail.com>
#
# MIT License
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE

import abc
import functools

import torch
from torchvision import transforms as T
from torchvision.transforms import functional as TF


class TransformImgSegm(abc.ABC):
    @abc.abstractmethod
    def __call__(self, img, segm):
        pass

    def _should_apply(self, prob):
        return torch.rand(1) < prob


class RandomHVFlipTransform(TransformImgSegm):
    def __init__(self, hflip_prob=0.5, vflip_prob=0.5):
        super().__init__()

        self.hlip_prob = hflip_prob
        self.vflip_prob = vflip_prob

    def __call__(self, img, segm):
        img, segm = self._prob_apply(img, segm, self.hlip_prob, TF.hflip)
        img, segm = self._prob_apply(img, segm, self.vflip_prob, TF.vflip)

        return img, segm

    def _prob_apply(self, img, segm, prob, func):
        if self._should_apply(prob):
            img = func(img)
            segm = func(segm)

        return img, segm


class ColorJitterTransform(TransformImgSegm):
    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        self.color_jitter = T.ColorJitter(brightness, contrast, saturation, hue)

    def __call__(self, img, segm):
        img = self.color_jitter(img)
        return img, segm


class RandomPerspectiveTransform(TransformImgSegm):
    def __init__(self, distortion_scale=0.5, prob=0.5):
        super().__init__()

        self.distortion_scale = distortion_scale
        self.prob = prob
        self.perspective = functools.partial(
            TF.perspective, interpolation=T.InterpolationMode.BICUBIC, fill=0
        )

    def __call__(self, img, segm):
        if self._should_apply(self.prob):
            width, height = TF._get_image_size(img)
            startpoints, endpoints = T.RandomPerspective.get_params(
                width, height, self.distortion_scale
            )

            img = self.perspective(img, startpoints, endpoints)
            segm = self.perspective(segm, startpoints, endpoints)

        return img, segm


class NormalizeTransform(TransformImgSegm):
    def __init__(self, mean, std):
        super().__init__()

        self.normalize = T.Normalize(mean, std)

    def __call__(self, img, segm):
        img = self.normalize(img)

        return img, segm


class GaussianBlurTransform(TransformImgSegm):
    def __init__(self, kernel_size=5, sigma=(0.1, 2.0), prob=0.5):
        super().__init__()

        self.kernel_size = kernel_size
        self.sigma = sigma
        self.prob = prob

    def __call__(self, img, segm):
        if self._should_apply(self.prob):
            img = TF.gaussian_blur(img, self.kernel_size, self.sigma)

        return img, segm


class ResizeTransform(TransformImgSegm):
    def __init__(self, size):
        super().__init__()

        self.resize = T.Resize(size, interpolation=T.InterpolationMode.BICUBIC)

    def __call__(self, img, segm):
        img = self.resize(img)
        segm = self.resize(segm)

        return img, segm


class ToTensorTransform(TransformImgSegm):
    def __init__(self, segm_pos_thresh=0.5):
        super().__init__()

        self.segm_pos_thresh = segm_pos_thresh
        self.to_tensor = T.ToTensor()

    def __call__(self, img, segm):
        img = self.to_tensor(img)
        segm = self.to_tensor(segm)

        pos_selector = segm > self.segm_pos_thresh
        neg_selector = ~pos_selector

        segm[pos_selector] = 1.0
        segm[neg_selector] = 0.0

        segm = segm.int()

        return img, segm


class ComposeTransform(TransformImgSegm):
    def __init__(self, transforms):
        super().__init__()

        self.transforms = transforms

    def __call__(self, img, segm):
        for transform in self.transforms:
            img, segm = transform(img, segm)

        return img, segm
