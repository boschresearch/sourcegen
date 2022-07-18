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
from torch.utils.data import Dataset

from diagvibsix.dataset.dataset import Dataset as MtmdDatasetRaw
from diagvibsix.utils.auxiliaries import load_yaml
from diagvibsix.utils.dataset_utils import get_mt_labels


class DiagVibDataset(Dataset):
    def __init__(self, spec_yaml, seed):
        cache_path = spec_yaml + '.seed_{}.cache.pkl'.format(seed)
        self.dataset_raw = MtmdDatasetRaw(dataset_spec=load_yaml(spec_yaml), seed=seed, cache_path=cache_path)
        self.num_samples = len(self.dataset_raw.images)

    def __getitem__(self, i):
        image, question_answer, tag = self.dataset_raw.getitem(i).values()
        attribute = get_mt_labels(question_answer)

        return torch.tensor(image[0] / 255, dtype=torch.float32), torch.tensor(attribute)

    def __len__(self):
        return self.num_samples

    def get_config(self):
        config = {
            'ds_x_index': 0,
            'ds_attribute_index': 1,
            'ds_attritube_type_list': self.dataset_raw.tasks
        }
        return config
