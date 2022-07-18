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
import matplotlib
import matplotlib.pyplot as plt
plt.style.use('seaborn')
matplotlib.use('Agg')

__all__ = ['StatTracker']


class StatTracker(object):
    def __init__(self):
        self.phase_val_data_dict = None

    def is_init(self):
        return self.phase_val_data_dict is not None

    def initialize(self, phase_val_names_dict):
        self.phase_val_data_dict = {phase: {name: [] for name in names} for (phase, names) in phase_val_names_dict.items()}

    def push_epoch(self, name_val_dict, phase):
        for name in self.phase_val_data_dict[phase]:
            if name in name_val_dict:
                val = float(name_val_dict[name])
                self.phase_val_data_dict[phase][name].append(val)
            else:
                self.phase_val_data_dict[phase][name].append(0)
