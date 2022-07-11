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

import sys
import warnings

import pytorch_lightning as pl
from pytorch_lightning import callbacks

from model import SegmentationModel
from data_modules import TextSegmDataModule


def main():
    dataset_dir_path = '../../datasets/MSRA-TD500'
    model = SegmentationModel(learning_rate=0.05)
    data_module = TextSegmDataModule(dataset_dir_path, batch_size=1)

    checkpoint_cb = callbacks.ModelCheckpoint(
        './checkpoints',
        every_n_epochs=4,
        filename='{epoch}-{val_precision_epoch:0.2f}-{val_recall_epoch:0.2f}'
    )

    trainer = pl.Trainer(
        default_root_dir='./logs',
        gpus=1,
        max_epochs=200,
        enable_progress_bar=True,
        log_every_n_steps=10,
        check_val_every_n_epoch=5,
        overfit_batches=1,
        num_sanity_val_steps=20,
        callbacks=[checkpoint_cb]
    )
    trainer.fit(model, datamodule=data_module)
    trainer.save_checkpoint('./model_checkpoint_final.ckpt')
    trainer.test(model, datamodule=data_module)

    return 0


if __name__ == '__main__':
    warnings.filterwarnings('ignore', category=DeprecationWarning)

    sys.exit(main())
