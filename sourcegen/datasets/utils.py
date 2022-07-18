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
import numpy as np
import os


def get_comp_pair_dir(comp_params):
    pair_dir = comp_params['root']
    pair_dir = pair_dir if os.path.isdir(pair_dir) else os.path.dirname(pair_dir)
    if ('is_pair_parent_dir' in comp_params) and comp_params['is_pair_parent_dir']:
        pair_dir = os.path.dirname(pair_dir)

    return pair_dir


def get_all_attrs(dataset, ds_attribute_index):
    attrs_set = set()
    for i in range(len(dataset)):
        data = dataset[i]
        attrs_set.add(tuple(data[ds_attribute_index].numpy()))

    return sorted(list(attrs_set))


def get_all_attr_cls_ids_list(dataset, ds_attribute_index):
    num_attributes = len(dataset[0][ds_attribute_index])
    attr_cls_ids_set_list = [set() for _ in range(num_attributes)]
    for i in range(len(dataset)):
        data = dataset[i]

        for j in range(num_attributes):
            attr_cls_id = int(data[ds_attribute_index][j])
            attr_cls_ids_set_list[j].add(attr_cls_id)

    attr_cls_ids_list = [sorted(list(attr_cls_ids_set)) for attr_cls_ids_set in attr_cls_ids_set_list]
    return attr_cls_ids_list


class ToTensorRaw():
    '''
    Take PIL Image and convert to FloatTensor
    '''

    def __init__(self, normalized_val=255.0):
        self.normalized_val = normalized_val

    def __call__(self, image):
        data = np.array(image)

        # transpose to pytorch format
        if len(data.shape) == 3:
            data = np.transpose(data, (2, 0, 1))

        # normalize
        data = data / self.normalized_val

        # transpose to pytorch format
        return torch.FloatTensor(data)