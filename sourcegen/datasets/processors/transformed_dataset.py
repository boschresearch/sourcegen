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


from .base_dataset import BaseDataset


class TransformedDataset(BaseDataset):
    def __init__(self, parent_dataset, transform, ds_x_index, ds_index_resize_handling_dict=None):
        super().__init__(parent_dataset)

        self.transform = transform
        self.ds_x_index = ds_x_index
        self.ds_index_resize_handling_dict = ds_index_resize_handling_dict

    def __getitem__(self, i):
        data_tuple = self.parent_dataset[i]
        data_list = list(data_tuple)

        if self.ds_index_resize_handling_dict is not None:
            original_x_size = data_list[self.ds_x_index].size

        # transform data
        data_list[self.ds_x_index] = self.transform(data_list[self.ds_x_index])

        # handling resize
        if self.ds_index_resize_handling_dict is not None:
            new_x_size = tuple(data_list[self.ds_x_index].shape[1:])
            for ds_index, resize_handling_function in self.ds_index_resize_handling_dict.items():
                data_list[ds_index] = resize_handling_function(data_list[ds_index],
                    original_x_size, new_x_size)

        return tuple(data_list)

    def __len__(self):
        return len(self.parent_dataset)
