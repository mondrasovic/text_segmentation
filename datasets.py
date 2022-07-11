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

import dataclasses
import pathlib
from typing import Sequence

import cv2 as cv
import numpy as np
from PIL import Image
from torch.utils.data import Dataset


@dataclasses.dataclass(frozen=True)
class TextAnno:
    x: int
    y: int
    width: int
    height: int
    angle_rad: float


@dataclasses.dataclass(frozen=True)
class DatasetItem:
    img_file_path: str
    annos: Sequence[TextAnno]


class MSRATextSegmDataset(Dataset):
    def __init__(self, dataset_dir_path, subset='train', transform=None):
        super().__init__()

        assert subset in ('train', 'val', 'test'), "unrecognized data subset"

        self.items = list(self._init_dataset(dataset_dir_path, subset))
        self.transform = transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        item = self.items[index]

        img = cv.imread(item.img_file_path, cv.IMREAD_COLOR)
        img_height, img_width, _ = img.shape
        segm = self._create_segm_mask_image(img_width, img_height, item.annos)

        img, segm = Image.fromarray(img), Image.fromarray(segm)

        if self.transform is not None:
            img, segm = self.transform(img, segm)

        return img, segm

    @staticmethod
    def _init_dataset(dataset_dir_path, subset='train'):
        dataset_dir = pathlib.Path(dataset_dir_path) / subset

        for img_file in dataset_dir.glob('*.JPG'):
            img_file_path = str(img_file)
            anno_file_path = dataset_dir / f'{img_file.stem}.gt'
            annos = tuple(MSRATextSegmDataset._read_gt_annos(anno_file_path))

            yield DatasetItem(img_file_path, annos)

    @staticmethod
    def _create_segm_mask_image(img_width, img_height, annos):
        segm = np.zeros((img_height, img_width), dtype=np.uint8)

        for anno in annos:
            center_x = anno.x + (anno.width / 2)
            center_y = anno.y + (anno.height / 2)

            rect = (
                (center_x, center_y), (anno.width, anno.height),
                np.degrees(anno.angle_rad)
            )
            box = np.int0(cv.boxPoints(rect))

            cv.drawContours(
                segm, [box], 0, color=255, thickness=-1, lineType=cv.LINE_AA
            )

        return segm

    @staticmethod
    def _read_gt_annos(anno_file_path):
        with open(anno_file_path, 'rt') as in_file:
            for anno_row in in_file:
                tokens = anno_row.split()
                x, y, width, height = map(int, tokens[2:-1])
                angle = float(tokens[-1])
                anno = TextAnno(x, y, width, height, angle)

                yield anno


if __name__ == '__main__':
    from transforms import (
        RandomHVFlipTransform, ResizeTransform, ColorJitterTransform,
        GaussianBlurTransform, RandomPerspectiveTransform, ToTensorTransform,
        ComposeTransform
    )

    def tensor_to_cv(img):
        return (img * 255).numpy().transpose(1, 2, 0).astype(np.uint8)

    hv_flip = RandomHVFlipTransform()
    resize = ResizeTransform((448, 448))
    color_jitter = ColorJitterTransform(0.25, 0.25, 0.25, 0.25)
    blur = GaussianBlurTransform(kernel_size=5)
    perspective = RandomPerspectiveTransform(0.25)
    to_tensor = ToTensorTransform()
    transform = ComposeTransform(
        (
            resize,
            color_jitter,
            blur,
            hv_flip,
            perspective,
            to_tensor,
        )
    )

    dataset = MSRATextSegmDataset(
        '../../datasets/MSRA-TD500', 'train', transform
    )

    for i in range(len(dataset)):
        img, segm = dataset[i]

        cv.imshow('Preview - image', tensor_to_cv(img))
        cv.imshow('Preview - segmentation mask', tensor_to_cv(segm))
        if (cv.waitKey(0) & 0xff) == ord('q'):
            break

    cv.destroyAllWindows()
