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

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from datasets import MSRATextSegmDataset
from transforms import (
    ColorJitterTransform, ComposeTransform, ResizeTransform,
    RandomHVFlipTransform, RandomPerspectiveTransform, GaussianBlurTransform,
    NormalizeTransform, ToTensorTransform
)


class TextSegmDataModule(pl.LightningDataModule):
    def __init__(
        self,
        dataset_dir_path,
        batch_size=4,
        n_workers=4,
        input_size=(512, 512),
        pix_mean=[0.485, 0.456, 0.406],
        pix_std=[0.229, 0.224, 0.225]
    ):
        super().__init__()

        self.dataset_dir_path = dataset_dir_path
        self.batch_size = batch_size
        self.n_workers = n_workers

        self.input_size = input_size
        self.pix_mean = pix_mean
        self.pix_std = pix_std

        self.dataset_train = None
        self.dataset_val = None
        self.dataset_test = None

    def setup(self, stage=None):
        # Sanity check executes validation within the fitting stage, so it is
        # required to have the transform and the dataset already available.
        self.dataset_test = self._make_dataset('test')
        # if self._is_fit_stage(stage):
        #     self.dataset_train = self._make_dataset('fit')
        #     self.dataset_val = self._make_dataset('val')
        # elif stage == 'test':
        #     self.dataset_test = self._make_dataset('test')

    def train_dataloader(self):
        # return self._make_dataloader(self.dataset_train, 'fit')
        return self._make_dataloader(self.dataset_test, 'test')

    def val_dataloader(self):
        # return self._make_dataloader(self.dataset_val, 'val')
        return self._make_dataloader(self.dataset_test, 'test')

    def test_dataloader(self):
        return self._make_dataloader(self.dataset_test, 'test')

    def _make_dataloader(self, dataset, stage=None):
        pin_memory = torch.cuda.is_available()
        shuffle = self._is_fit_stage(stage)

        data_loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.n_workers,
            pin_memory=pin_memory
        )

        return data_loader

    def _make_dataset(self, stage=None):
        subset = 'train' if self._is_fit_stage(stage) else stage
        transform = self._make_transform(stage)

        dataset = MSRATextSegmDataset(self.dataset_dir_path, subset, transform)

        return dataset

    def _make_transform(self, stage=None):
        transforms = [ResizeTransform(self.input_size)]

        # if self._is_fit_stage(stage):
        #     transforms.append(ColorJitterTransform(0.2, 0.2, 0.2, 0.2))
        #     transforms.append(GaussianBlurTransform())
        #     transforms.append(RandomHVFlipTransform())
        #     transforms.append(RandomPerspectiveTransform())

        transforms.append(ToTensorTransform())
        transforms.append(NormalizeTransform(self.pix_mean, self.pix_std))

        transform = ComposeTransform(transforms)

        return transform

    @staticmethod
    def _is_fit_stage(stage):
        return (stage is None) or (stage == 'fit')
