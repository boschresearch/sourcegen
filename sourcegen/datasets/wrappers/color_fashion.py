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
# Author: Piyapat Saranrittichai, Volker Fischer
# -*- coding: utf-8 -*-


import torch
import os
import numpy as np

from PIL import Image
from torch.utils.data import Dataset


def load_split_image_names(dataset_dir, split):
    split_filepath = os.path.join(dataset_dir, 'split', '{}.txt'.format(split))
    with open(split_filepath, 'r') as f:
        image_names = [line.strip() for line in f.readlines()]

    return image_names


def load_image_name_attributes_dict(dataset_dir):
    def load_annotation_file(filepath):
        img_name_list = []
        label_list = []
        with open(filepath, 'r') as f:
            for line in f.readlines():
                words = line.strip().split(',')

                img_name_list.append(words[0])
                label_list.append([int(words[1]), int(words[2])])

        return img_name_list, np.array(label_list)

    img_name_list, img_labels = load_annotation_file(os.path.join(dataset_dir, 'annotations.txt'))

    image_name_attributes_dict = {}
    for i, image_name in enumerate(img_name_list):
        image_name_attributes_dict[image_name] = img_labels[i]

    return image_name_attributes_dict


def load_image_metadata_dict(dataset_dir, split):
    image_names = load_split_image_names(dataset_dir, split)

    # load image_filenames
    image_filenames = [os.path.join(dataset_dir, 'images', image_name) for image_name in image_names]

    # load labels
    image_name_attributes_dict = load_image_name_attributes_dict(dataset_dir)
    image_attributes = [image_name_attributes_dict[image_name] for image_name in image_names]

    return image_filenames, image_attributes


class ColorFashion(Dataset):
    def __init__(self, dataset_dir, split):
        assert split in ['train', 'val', 'test']
        self.split = split
        self.dataset_dir = dataset_dir

        self.image_filenames, self.image_attributes = load_image_metadata_dict(dataset_dir, self.split)

    def __getitem__(self, i):
        # get image
        image_filename = self.image_filenames[i]
        image = Image.open(image_filename).convert("RGB")

        return image, torch.tensor(self.image_attributes[i])

    def __len__(self):
        return len(self.image_filenames)

    @staticmethod
    def get_config():
        config = {
            'ds_x_index': 0,
            'ds_attribute_index': 1
        }
        return config
