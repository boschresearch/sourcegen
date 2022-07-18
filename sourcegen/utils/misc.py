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


import json
import ast
from pydoc import locate
import math
import pickle
import numpy as np
import copy
from collections import Counter


def load_json(json_file):
    with open(json_file) as f:
        all_config_data = json.load(f)
    return all_config_data


def store_json(data_dict, json_file):
    with open(json_file, 'w') as f:
        json.dump(data_dict, f, indent=4, sort_keys=True)


def refine_data_dict(data_dict_raw, refine_config_settings):
    data_dict = copy.deepcopy(data_dict_raw)
    for refine_config_setting in refine_config_settings:
        # parse input
        words = refine_config_setting.split(':')
        key_list = words[:-2]
        value_type_str = words[-2]
        value_str = words[-1]
        if value_type_str == 'literal':
            value = ast.literal_eval(value_str)
        else:
            value_type = locate(value_type_str)
            value = value_type(value_str)

        # set the new value with the same type as the old value
        set_dict_value_with_key_list(data_dict, key_list, value)

    return data_dict


def set_dict_value_with_key_list(input_dict, key_list, value):
    target = input_dict
    for i, key in enumerate(key_list):
        if isinstance(target, list):
            key = int(key)
        if i == (len(key_list) - 1):
            target[key] = value
        else:
            target = target[key]


def save_obj(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


class MeanAccumulator(object):
    def __init__(self):
        self.current_mean = 0
        self.current_count = 0

    def accumulate(self, val, count):
        new_count = self.current_count + count
        self.current_mean = self.current_mean * (self.current_count / new_count) + val * (count / new_count)
        self.current_count = new_count

    def get_mean(self):
        return self.current_mean


class MeanAccumulatorSet(object):
    def __init__(self, var_names=None):
        self.name_accumulator_dict = None
        if var_names is not None:
            self.reset_name_accumulator_dict(var_names)

    def accumulate(self, name_val_dict, count):
        for name, val in name_val_dict.items():
            self.name_accumulator_dict[name].accumulate(val, count)

    def get_name_mean_dict(self):
        name_mean_dict = {}
        for name in self.name_accumulator_dict.keys():
            name_mean_dict[name] = self.name_accumulator_dict[name].get_mean()

        return name_mean_dict

    def reset_name_accumulator_dict(self, var_names):
        self.name_accumulator_dict = {name: MeanAccumulator() for name in var_names}


class MaxTrackerSet(object):
    def __init__(self, var_names):
        self.name_max_dict = {name: -math.inf for name in var_names}

    def update(self, name_val_dict):
        for name, val in name_val_dict.items():
            if val > self.name_max_dict[name]:
                self.name_max_dict[name] = val

    def get_name_max_dict(self):
        return self.name_max_dict
