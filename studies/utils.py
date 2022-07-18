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


import os
import json


def load_json(json_file):
    with open(json_file) as f:
        all_config_data = json.load(f)
    return all_config_data


def store_json(data_dict, json_file):
    with open(json_file, 'w') as f:
        json.dump(data_dict, f, indent=4, sort_keys=True)


def merge_json_file(input_file_1, input_file_2, output_file):
    data_1 = load_json(input_file_1)
    data_2 = load_json(input_file_2)
    store_json({**data_1, **data_2}, output_file)


def create_merged_dataset_config(dataset_source, dataset_target, repo_dir, config_cache_dir='data/cache/configs',
                                 overwrite=False):
    config_dir = os.path.join(config_cache_dir, 'configs', dataset_source, dataset_target)
    os.makedirs(config_dir, exist_ok=True)
    config_file = os.path.join(config_dir, '{}-{}.json'.format(dataset_source, dataset_target))

    # merge json
    if (not os.path.exists(config_file)) or overwrite:
        dataset_spec_source = os.path.join(repo_dir, f'configs/datasets/sources/{dataset_source}.json')
        dataset_spec_target = os.path.join(repo_dir, f'configs/datasets/targets/{dataset_target}.json')
        merge_json_file(dataset_spec_source, dataset_spec_target, config_file)

    return config_file
