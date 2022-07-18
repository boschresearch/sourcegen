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


from torch import optim
from torch.utils.data import DataLoader


def build_dataloader(dataloader_spec, dataset, shuffle=False):
    dataloader = DataLoader(dataset, shuffle=shuffle, **dataloader_spec)
    return dataloader


def build_optimizer(optimizer_spec, model_parameters):
    if optimizer_spec['name'] == 'Adam':
        return optim.Adam(model_parameters, **optimizer_spec['params'])
    else:
        raise RuntimeError(
            'Unsupported optimizer {}'.format(optimizer_spec['name']))

