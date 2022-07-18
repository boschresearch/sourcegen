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
import logging

from copy import deepcopy
import pickle
from tqdm import tqdm

from .base_dataset import BaseDataset


class CachedDataset(BaseDataset):
    def __init__(self, parent_dataset, storage_file_path=None):
        super().__init__(parent_dataset)

        # fill in the cache
        if (storage_file_path is not None) and (os.path.exists(storage_file_path)):
            # load cache
            with open(storage_file_path, 'rb') as f:
                self.cache_i_item_dict = pickle.load(f)

            logging.info('Load dataset from cache file {}'.format(storage_file_path))
        else:
            logging.info('CacheDataset: Initialize dataset cache of {} elements.'.format(len(parent_dataset)))
            self.cache_i_item_dict = {}
            for i in tqdm(range(len(parent_dataset)), desc='Indexing data for cache'):
                self.cache_i_item_dict[i] = parent_dataset[i]
            logging.info('CacheDataset: Initialization done.')

            if storage_file_path is not None:
                # create directory
                storage_dir = os.path.dirname(storage_file_path)
                if storage_dir:
                    os.makedirs(storage_dir, exist_ok=True)

                # save cache
                with open(storage_file_path, 'wb') as f:
                    pickle.dump(self.cache_i_item_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

                logging.info('Save cache storage to {}'.format(storage_file_path))

    def __getitem__(self, i):
        return deepcopy(self.cache_i_item_dict[i])

    def __len__(self):
        return len(self.cache_i_item_dict)
