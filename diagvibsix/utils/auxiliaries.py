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

from imageio import mimsave
import yaml
import os
import argparse
import pickle

__all__ = ['save_gif', 'save_yaml', 'load_yaml', 'dump_config', 'write_git_hash',
           'write_start_command', 'split_train_val', 'save_obj', 'load_obj', 'get_corr_pred', 'get_dataset_tags']


def save_gif(img, savename='sample.png', duration=0.2):
    mimsave(savename, [img[i].transpose((1, 2, 0)) for i in range(img.shape[0])], duration=duration)


def save_yaml(obj, path):
    with open(path, 'w') as outfile:
        yaml.dump(obj, outfile, default_flow_style=False)


def load_yaml(filepath):
    return yaml.load(open(filepath), Loader=yaml.FullLoader)


def save_obj(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def dump_config(args: argparse.Namespace(), path: str):
    save_yaml(vars(args), os.path.join(path, 'args.yaml'))


def write_git_hash(path: str = ''):
    with open(os.path.join(path, 'git_hash.txt'), 'w') as outfile:
        outfile.write(os.environ.get('COMMIT_HASH', 'no hash'))


def write_start_command(path: str, args_string: str):
    with open(os.path.join(path, 'start_command.txt'), 'w') as outfile:
        outfile.write(args_string)


def split_train_val(dataset_spec, tv_split):
    """ Returns two dataset_specs with same specs but number of samples split according to tv_split.
    """

    samples_train = int(dataset_spec['samples'] * tv_split)
    samples_val = dataset_spec['samples'] - samples_train

    dataset_spec_train, dataset_spec_val = dataset_spec.copy(), dataset_spec.copy()
    dataset_spec_train['samples'], dataset_spec_val['samples'] = samples_train, samples_val

    return dataset_spec_train, dataset_spec_val


def get_corr_pred(study_name):
    """
    A study name is of the form CORR-factor1-factor2-factor3_PRED-factor1-factor2.
    If no factors are correlated, then CORR_PRED-factor1.
    This function returns lists of correlated and predicted factors from this string.
    """
    if 'CORR-' in study_name:
        # There are correlations
        corrs, preds = study_name.split('CORR-')[1].split('_PRED-')
        corrs = corrs.split('-')
        preds = preds.split('-')
    else:
        # There are no correlations
        corrs = []
        preds = study_name.split('PRED-')[1].split('-')
    return corrs, preds


def get_dataset_tags(spec):
    tags = [mode['spec']['tag'] for mode in spec['modes']]
    return list(set(tags))
