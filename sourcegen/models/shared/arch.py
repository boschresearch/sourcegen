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
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import torch.nn as nn


def get_activation_func(activation):
    if activation == "Identity":
        return nn.Identity()
    elif activation == "ReLU":
        return nn.ReLU()
    elif activation == "LeakyReLU":
        return nn.LeakyReLU()
    else:
        raise RuntimeError(f"Activation {activation} is not supported.")


# The following class MLP is from attributes-as-operators
#   (https://github.com/Tushar-N/attributes-as-operators/blob/c59ff784a4c626541e2eb21fa48dc304086beeb2/models/models.py)
# Copyright (c) 2018, licensed under the MIT license,
# cf. 3rd-party-licenses.txt file in the root directory of this source tree.
class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hsize_list=[], activation="Identity", use_batch_norm=False, use_bias=True):
        super().__init__()

        modules = []
        for i in range(len(hsize_list) + 1):
            current_input_dim = input_dim if i == 0 else hsize_list[i-1]
            current_output_dim = output_dim if i == len(hsize_list) else hsize_list[i]

            modules.append(nn.Linear(current_input_dim, current_output_dim, bias=use_bias))

            if i != len(hsize_list):
                if use_batch_norm:
                    modules.append(nn.BatchNorm1d(current_output_dim))
                modules.append(get_activation_func(activation))

        self.layer = nn.Sequential(*modules)

    def forward(self, x):
        return self.layer(x)
