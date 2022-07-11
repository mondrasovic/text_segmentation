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

import click
import numpy as np
import pytorch_lightning as pl
import torch
from torchvision import transforms
from PIL import Image

from model import SegmentationModel
from data_modules import TextSegmDataModule


@click.command()
@click.argument('input_file_path', type=click.Path(exists=True))
@click.argument('output_file_path', type=click.Path())
def main(input_file_path, output_file_path):
    dataset_dir_path = '../../datasets/MSRA-TD500'
    model = SegmentationModel()
    model.load_from_checkpoint('./model_checkpoint_final.ckpt')
    data_module = TextSegmDataModule(dataset_dir_path)
    transform = data_module._make_transform('test')

    img = Image.open(input_file_path)
    segm_dummy = Image.fromarray(np.zeros((512, 512), dtype=np.uint8))
    img_tensor, _ = transform(img, segm_dummy)
    segm_pred = torch.squeeze(model(torch.unsqueeze(img_tensor, dim=0)), dim=0)
    segm_img = transforms.ToPILImage()(segm_pred)
    segm_img.save(output_file_path)

    return 0


if __name__ == '__main__':
    warnings.filterwarnings('ignore', category=DeprecationWarning)

    sys.exit(main())
