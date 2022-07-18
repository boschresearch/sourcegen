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
from glob import glob

from PIL import Image
from torch.utils.data import Dataset


def load_label_id_name_dict(dataset_dir):
    label_names = next(os.walk(os.path.join(dataset_dir, 'fruits-360', 'Training')))[1]
    label_names.sort()

    return {i: name for i, name in enumerate(label_names)}


def load_image_metadata_dict(dataset_dir, split):
    label_id_name_dict = load_label_id_name_dict(dataset_dir)

    image_filenames = []
    image_attributes = []
    for label_id, label_name in label_id_name_dict.items():
        for color_id in range(len(label_id_name_dict)):
            target_dir = os.path.join(dataset_dir, 'fruits-360', split, label_name, str(color_id))

            for dir, _, _ in os.walk(target_dir):
                filenames = glob(os.path.join(dir, '*.jpg'))
                n_files = len(filenames)

                if n_files > 0:
                    filenames.sort()
                    image_filenames.extend(filenames)
                    image_attributes.extend([[color_id, label_id]] * n_files)

    return image_filenames, image_attributes


class Fruits360Colorized(Dataset):
    def __init__(self, dataset_dir, split):
        assert split in ['Training', 'Validation', 'Test']
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

    def get_label_id_name_dict(self):
        return load_label_id_name_dict(self.dataset_dir)

    @staticmethod
    def get_config():
        config = {
            'ds_x_index': 0,
            'ds_attribute_index': 1
        }
        return config


if __name__ == '__main__':
    dataset_dir, split = "/fs/scratch/rng_cr_bcai_dl/asp2abt/datasets/Fruits-360-Colorized", "Training"
    print('Loading dataset from {} with split {}'.format(dataset_dir, split))

    dataset = Fruits360Colorized(dataset_dir, split)

    print("Loaded")
    import IPython; IPython.embed()
