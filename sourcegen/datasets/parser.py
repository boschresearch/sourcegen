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
import pickle

from . import wrappers as wrappers
from . import processors as processors
from .utils import get_all_attr_cls_ids_list, get_all_attrs, get_comp_pair_dir


def get_dataset_catalog(dataset_spec):
    dataset_catalog = {}

    # sources
    for source_key in ['sources_train', 'sources_val']:
        dataset_catalog[source_key] = []
        for dataset_params in dataset_spec[source_key]:
            dataset_catalog[source_key].append(get_dataset(dataset_params)[0])

    # targets
    for target_key in ['target_train', 'target_val', 'target_test']:
        dataset_catalog[target_key], need_post_conversion = get_dataset(dataset_spec[target_key])

    # post conversion
    if need_post_conversion:
        for target_key in ['target_train', 'target_val', 'target_test']:
            split_name = target_key.split('_')[-1]
            pair_dir = get_comp_pair_dir(dataset_spec[target_key]['comp_params'])

            pairs_filename = os.path.join(pair_dir, f'{split_name}_pairs.pkl')
            if not os.path.exists(pairs_filename):
                dataset, dataset_config = dataset_catalog[target_key]
                split_pairs = get_all_attrs(dataset, dataset_config['ds_attribute_index'])

                with open(pairs_filename, 'wb') as f:
                    pickle.dump(split_pairs, f)

        for target_key in ['target_train', 'target_val', 'target_test']:
            dataset, dataset_config = dataset_catalog[target_key]
            dataset = processors.ConvertedCompositionActivationsDataset(
                dataset, dataset_config['ds_x_index'], dataset_config['ds_attribute_index'],
                **dataset_spec[target_key]['comp_params'])
            dataset.set_activate(dataset_spec[target_key]['activate'])

            dataset_catalog[target_key] = (dataset, dataset_config)

    return dataset_catalog


def get_dataset(dataset_params):
    need_post_conversion = False
    if dataset_params['name'] == 'diag_vib':
        # load
        dataset_raw = wrappers.DiagVibDataset(**dataset_params['params'])
        dataset_config = dataset_raw.get_config()

        attr_cls_ids_list = get_all_attr_cls_ids_list(dataset_raw, dataset_config['ds_attribute_index'])
        dataset_ordered = processors.OrderedAttributesDataset(dataset_raw, attr_cls_ids_list,
                                                            dataset_config['ds_attribute_index'])

        dataset = processors.ActivatedDataset(dataset_ordered, dataset_config['ds_x_index'], dataset_params['root'])
        dataset.set_activate(dataset_params['activate'])

        # update config
        dataset_config['attr_dim_list'] = [len(attr_cls_ids) for attr_cls_ids in attr_cls_ids_list]
    elif dataset_params['name'] == 'diag_vib_composition':
        # load
        dataset_raw = wrappers.DiagVibDataset(**dataset_params['params'])
        dataset_config = dataset_raw.get_config()

        attr_cls_ids_list = get_all_attr_cls_ids_list(dataset_raw, dataset_config['ds_attribute_index'])
        dataset_no_comp = processors.OrderedAttributesDataset(dataset_raw, attr_cls_ids_list,
                                                              dataset_config['ds_attribute_index'])

        # convert to comp dataset
        dataset = dataset_no_comp
        need_post_conversion = True
    elif dataset_params['name'] == 'fruits_360_colorized_composition':
        # get config
        dataset_config = {
            **wrappers.Fruits360Colorized.get_config()
        }

        # load
        dataset_raw = wrappers.Fruits360Colorized(**dataset_params['params'])

        # convert to comp dataset
        dataset = dataset_raw
        need_post_conversion = True
    elif dataset_params['name'] == 'color_fashion':
        # get config
        dataset_config = {
            **wrappers.ColorFashion.get_config()
        }

        # load
        dataset_raw = wrappers.ColorFashion(**dataset_params['params'])

        attr_cls_ids_list = get_all_attr_cls_ids_list(dataset_raw, dataset_config['ds_attribute_index'])
        dataset_no_comp = processors.OrderedAttributesDataset(dataset_raw, attr_cls_ids_list,
                                                              dataset_config['ds_attribute_index'])

        # convert to comp dataset
        dataset = dataset_no_comp
        need_post_conversion = True
    elif dataset_params['name'] == 'ut_zappos_material':
        # get config
        dataset_config = {
            'ds_x_index': 0
        }

        # load
        dataset = wrappers.CompositionDatasetActivations(**dataset_params['params'])
        dataset.set_activate(dataset_params['activate'])

        # process
        dataset_config['attr_dim_list'] = [len(dataset.attrs), len(dataset.objs)]

        return dataset, dataset_config
    elif dataset_params['name'] == 'ao_clevr':
        # get config
        dataset_config = {
            'ds_x_index': 0
        }

        # load
        dataset = wrappers.AOClevrCompositionDatasetActivations(**dataset_params['params'])
        dataset.set_activate(dataset_params['activate'])
    else:
        raise RuntimeError('Dataset with name {} is not supported.'.format(dataset_params['name']))

    return (dataset, dataset_config), need_post_conversion
