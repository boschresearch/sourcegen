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


import numpy as np
from torch.utils.data import Dataset


class MergedFilledDataset(Dataset):
    def __init__(self, parent_dataset0, parent_dataset1, random_seed=0):
        super().__init__()

        self.parent_dataset0 = parent_dataset0
        self.parent_dataset1 = parent_dataset1
        self.random_state = np.random.RandomState(random_seed)

    def __getitem__(self, i):
        if i < len(self.parent_dataset0):
            data0 = self.parent_dataset0[i]
        else:
            i0 = self.random_state.randint(len(self.parent_dataset0))
            data0 = self.parent_dataset0[i0]

        if i < len(self.parent_dataset1):
            data1 = self.parent_dataset1[i]
        else:
            i1 = self.random_state.randint(len(self.parent_dataset1))
            data1 = self.parent_dataset1[i1]

        return (data0, data1)

    def __len__(self):
        return max(len(self.parent_dataset0), len(self.parent_dataset1))
