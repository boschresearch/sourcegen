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
import sys
from datetime import datetime
import argparse
import socket

from . import utils


def run_job(method, source_dataset, target_dataset, training_seed):
    # arguments
    repo_dir = os.getcwd()
    method_str = method.replace('/', '-') + '-' + datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%f")
    result_dir = f'{repo_dir}/outputs/study_fully_correlated_{method_str}'
    dataset_spec = utils.create_merged_dataset_config(source_dataset, target_dataset, repo_dir)
    method_spec = f'{repo_dir}/configs/methods/{method}/method_spec.json'

    # run command
    exe = 'sourcegen.train'
    command_str = f'python -m {exe} --result_dir {result_dir} --dataset_spec {dataset_spec} ' \
        f'--method_spec {method_spec} --test --training_seed {training_seed}'
    print(f'Executing {command_str}')
    sys.stdout.flush()
    os.system(command_str)


if __name__ == '__main__':
    # study combinations
    training_seed_list = [2000]
    source_dataset = 'shape_hue_lightness_texture_bg'
    target_dataset_list = ['color_animal_semi_correlated']
    method_list = ['default/FactorSRC-IL', 'default/FactorSRC-IL-LA', 'default/FactorSRC-IL-LAR']

    for training_seed in training_seed_list:
        for target_dataset in target_dataset_list:
            for method in method_list:
                run_job(method, source_dataset, target_dataset, training_seed)
