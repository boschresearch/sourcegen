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


class OrderedAttributesDataset(BaseDataset):
    def __init__(self, parent_dataset, attr_cls_ids_list, ds_attribute_index):
        super().__init__(parent_dataset)

        self.num_attributes = len(attr_cls_ids_list)
        self.old_to_new_attr_cls_dict_list = []
        for i in range(self.num_attributes):
            old_to_new_attr_cls_dict = {int(old_label): int(new_label) for new_label, old_label in enumerate(attr_cls_ids_list[i])}
            self.old_to_new_attr_cls_dict_list.append(old_to_new_attr_cls_dict)
        self.ds_attribute_index = ds_attribute_index


    def __getitem__(self, i):
        data_tuple = self.parent_dataset[i]
        data_list = list(data_tuple)

        # convert attributes
        attributes = data_list[self.ds_attribute_index]
        for attr_id in range(self.num_attributes):
            attributes[attr_id] = self.old_to_new_attr_cls_dict_list[attr_id][int(data_list[self.ds_attribute_index][attr_id])]
        data_list[self.ds_attribute_index] = attributes

        return tuple(data_list)

    def __len__(self):
        return len(self.parent_dataset)
