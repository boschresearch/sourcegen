#!/usr/local/bin/python3
# Copyright (c) 2022 Robert Bosch GmbH Copyright holder of the paper "Overcoming Shortcut Learning in a Target Domain by Generalizing Basic Visual Factors from a Source Domain" accepted at ECCV 2022.
# All rights reserved.
###
# The paper "Overcoming Shortcut Learning in a Target Domain by Generalizing Basic Visual Factors from a Source Domain" accepted at ECCV 2022.
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.
#
# Author: Elias Eulig, Piyapat Saranrittichai, Volker Fischer
# -*- coding: utf-8 -*-

import numpy as np
from skimage.transform import rescale
from tqdm import trange
import os
from PIL import Image


def get_bbox(im, idx=255):
    """Returns bounding box.
    """

    ys = np.where(np.max(im, axis=0) == idx)
    xs = np.where(np.max(im, axis=1) == idx)
    bbox = (slice(np.min(xs), np.max(xs) + 1), slice(np.min(ys), np.max(ys) + 1))
    return bbox


def process_animals():
    print('Process Animal Dataset')

    assert(os.path.exists('non_rigid_shape_A.zip') and os.path.exists('non_rigid_shape_B.zip'))

    if not os.path.exists('raw/animals'):
        os.makedirs('raw/animals', exist_ok=True)
        os.system('unzip non_rigid_shape_A.zip; unzip non_rigid_shape_B.zip')
        os.system('mv non_rigid_shape_A/* non_rigid_shape_B/* raw/animals')
        os.system('rmdir non_rigid_shape_A non_rigid_shape_B')

    target_classes = ['fish', 'tortoise', 'hen', 'butterfly', 'leopard', 'rabbit', 'duck', 'bird', 'dolphine', 'elephant']
    SHARED_LOADPATH_ANIMALS = os.path.join(os.getcwd(), 'raw/animals')
    SHARED_SAVEPATH_ANIMALS = os.path.join(os.getcwd(), 'processed/animals.npz')
    os.makedirs(os.path.dirname(SHARED_SAVEPATH_ANIMALS), exist_ok=True)

    IMG_SIZE = 40
    OBJ_SIZE = IMG_SIZE // np.sqrt(2)
    TRAIN_TEST_SPLIT = 0.8

    x_train_samples, x_test_samples = [], []

    for c in target_classes:
        c_path = os.path.join(SHARED_LOADPATH_ANIMALS, c)
        c_files = sorted([os.path.join(c_path, f) for f in os.listdir(c_path) if f.endswith('.tif')])

        num_train = int(TRAIN_TEST_SPLIT * len(c_files))

        for idx, file in enumerate(c_files):
            im = np.array(Image.open(file))
            mask = np.where(im > 100, 255, 0)
            bbox = get_bbox(mask, idx=255)
            im = im[bbox]
            im_rescaled = rescale(im, np.min([OBJ_SIZE / im.shape[0], OBJ_SIZE / im.shape[1]]), anti_aliasing=True,
                                  preserve_range=True, order=3).astype('uint8')
            canvas = np.zeros((IMG_SIZE, IMG_SIZE), dtype='uint8')
            x0 = int((IMG_SIZE - 1) / 2 - (im_rescaled.shape[1] - 1) / 2)
            y0 = int((IMG_SIZE - 1) / 2 - (im_rescaled.shape[0] - 1) / 2)
            pos = (slice(y0, y0 + im_rescaled.shape[0]), slice(x0, x0 + im_rescaled.shape[1]))
            canvas[pos] = im_rescaled

            if idx < num_train:
                x_train_samples.append(canvas)
            else:
                x_test_samples.append(canvas)

    x_train_samples = np.array(x_train_samples)
    x_test_samples = np.array(x_test_samples)

    np.savez(SHARED_SAVEPATH_ANIMALS,
             x_train=x_train_samples,
             x_test=x_test_samples)


def process_caltech101():
    print('Process Caltech Dataset')

    assert (os.path.exists('caltech101_silhouettes_28_split1.mat'))
    os.makedirs('raw/caltech101_silhouette', exist_ok=True)
    os.system('cp caltech101_silhouettes_28_split1.mat raw/caltech101_silhouette')


if __name__ == '__main__':
    process_animals()
    process_caltech101()
