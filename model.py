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
from torch import nn
from torch import optim
from torchmetrics import Accuracy, MetricCollection, Precision, Recall, F1Score
from torchvision import models

from losses import binary_cross_entropy_balanced_loss


class SegmentationModel(pl.LightningModule):
    def __init__(
        self,
        n_upsampling_channels=256,
        pretrained=True,
        learning_rate=0.03,
        momentum=0.9,
        weight_decay=0.00005,
        pos_thresh=0.5
    ):
        super().__init__()

        self.model = self._make_model(n_upsampling_channels, pretrained)

        self.learning_rate = learning_rate
        self.momentum = momentum
        self.weight_decay = weight_decay

        params = dict(num_classes=1, threshold=pos_thresh)
        metrics = MetricCollection(
            {
                'accuracy': Accuracy(**params),
                'precision': Precision(**params),
                'recall': Recall(**params),
                'f1_score': F1Score(**params)
            },
            postfix='_epoch'
        )

        self.val_metrics = metrics.clone(prefix='val_')
        self.test_metrics = metrics.clone(prefix='test_')

    def configure_optimizers(self):
        params = [param for param in self.parameters() if param.requires_grad]

        optimizer = optim.SGD(
            params,
            lr=self.learning_rate,
            momentum=self.momentum,
            # weight_decay=self.weight_decay
        )
        lr_scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=2, gamma=0.9
        )

        return [optimizer], [lr_scheduler]

    def training_step(self, batch, batch_idx):
        img, segm_target = batch
        _, loss = self._calc_pred_and_loss(img, segm_target)

        self.log('train_loss_step', loss, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        return self._val_or_test_step(batch, self.val_metrics)

    def validation_epoch_end(self, outputs):
        self._val_or_test_epoch_end(self.val_metrics)

    def test_step(self, batch, batch_idx):
        return self._val_or_test_step(batch, self.test_metrics)

    def test_epoch_end(self, outputs):
        self._val_or_test_epoch_end(self.test_metrics)

    def _val_or_test_step(self, batch, metric_collection):
        img, segm_target = batch
        segm_pred, loss = self._calc_pred_and_loss(img, segm_target)

        segm_target = segm_target.flatten()
        segm_pred = segm_pred.flatten()

        metric_collection.update(segm_pred, segm_target)

        return loss

    def _val_or_test_epoch_end(self, metric_collection):
        self.log_dict(metric_collection.compute(), prog_bar=True)
        metric_collection.reset()

    def forward(self, x):
        x = self.model(x)
        x = torch.sigmoid(x)

        return x

    @staticmethod
    def _make_model(n_upsampling_channels, pretrained=True):
        def _upsampling_block(in_channels, out_channels):
            return (
                nn.ConvTranspose2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    output_padding=1
                ), nn.BatchNorm2d(num_features=out_channels),
                nn.LeakyReLU(inplace=True)
            )

        resnet18 = models.resnet18(pretrained=pretrained)

        model = nn.Sequential(
            resnet18.conv1,  # Downsampling stage.
            resnet18.bn1,
            resnet18.relu,
            resnet18.maxpool,
            resnet18.layer1,
            resnet18.layer2,
            *_upsampling_block(128, n_upsampling_channels),  # Upsampling stage.
            *_upsampling_block(n_upsampling_channels, n_upsampling_channels),
            *_upsampling_block(n_upsampling_channels, n_upsampling_channels),
            nn.Conv2d(
                in_channels=n_upsampling_channels,
                out_channels=1,
                kernel_size=1
            )
        )

        return model

    def _calc_pred_and_loss(self, img, segm_target):
        segm_pred = self.forward(img)
        loss = binary_cross_entropy_balanced_loss(segm_pred, segm_target)

        return segm_pred, loss
