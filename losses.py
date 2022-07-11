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

import torch
from torch.nn import functional as F


def binary_cross_entropy_balanced_loss(segm_pred, segm_target):
    segm_pred = segm_pred.flatten()
    segm_target = segm_target.flatten()

    pos_selector = segm_target == 1
    neg_selector = ~pos_selector

    n_pos = pos_selector.count_nonzero()
    n_neg = len(segm_pred) - n_pos

    pos_weight = 1.0 / n_pos if n_pos > 0 else 0.0
    neg_weight = 1.0 / n_neg if n_neg > 0 else 0.0

    weights = torch.zeros_like(segm_target, dtype=torch.float32)
    weights[pos_selector] = pos_weight
    weights[neg_selector] = neg_weight
    weights /= weights.sum()

    loss_per_pixel = F.binary_cross_entropy(
        segm_pred, segm_target.float(), reduction='none'
    )
    loss = (loss_per_pixel * weights).sum()

    return loss
