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
import torch.nn as nn


class BaseModel(nn.Module):
    def __init__(self, input_shape, source_id_attribute_dims_dict, target_attribute_dims):
        super().__init__()
        self.input_shape = input_shape

        self.source_id_attribute_dims_dict = source_id_attribute_dims_dict
        self.target_attribute_dims = target_attribute_dims
        self.num_attributes = len(self.target_attribute_dims)
        for _, source_attribute_dims in self.source_id_attribute_dims_dict.items():
            assert(len(source_attribute_dims) == self.num_attributes)

        # assign domain ids (the last domain is the target domain)
        self.num_domains = len(self.source_id_attribute_dims_dict) + 1
        self.target_domain_id = self.num_domains - 1

    # return a dict containing 'logits_list' as a key
    def forward(self, x, domain_id):
        raise NotImplementedError()

    # return a dict containing 'latents_attr_list' as a key
    def encode(self, x):
        raise NotImplementedError()

    # freeze encoder parameters
    def freeze_encoder(self):
        raise NotImplementedError()

    # return results as the average loss and a list of (output, target) and auxiliary data
    def train_epoch(self, epoch, data_list, domain_ids, learning_rule):
        raise NotImplementedError()

    # return results as the average loss and a list of (output, target) and auxiliary data
    @torch.no_grad()
    def val_epoch(self, epoch, data_list, domain_ids, learning_rule):
        raise NotImplementedError()
