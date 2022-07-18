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


import argparse


def make_parser():
    parser = argparse.ArgumentParser(description='Running Experiment', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # -------------------------------   General Settings   ------------------------------#
    parser.add_argument('--result_dir', required=True,
                        help='The directory for storing outputs.')
    parser.add_argument('-dev', '--device', type=int, default=0,
                        help='CUDA device to use. If -1, cpu is used instead.')
    parser.add_argument('--test',
                        help='If set, will run the best validation model on the test dataset directly after training.',
                        action='store_true')
    parser.add_argument('--pretrain_model_path', default=None,
                        help='Path to the pretrain model.')

    # -----------------------------------   Dataset   -----------------------------------#
    parser.add_argument('--dataset_spec', required=True,
        help='JSON file for dataset specification')

    # -----------------------------------   Method   -----------------------------------#
    parser.add_argument('--method_spec', required=True,
        help='JSON file for method specification')
    parser.add_argument('--training_seed', type=int, default=2000,
                        help='Seed to use for training setup.')

    return parser
