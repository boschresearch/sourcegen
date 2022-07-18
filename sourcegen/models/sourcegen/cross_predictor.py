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
import torch.nn.functional as F
from ..shared.arch import MLP


def cross_entropy_with_label_vectors(logits, label_vectors):
    probs = logits.log_softmax(dim=1)
    return torch.mean(torch.sum(-label_vectors * probs, dim=1))


class CrossFactorPredictorModule(nn.Module):
    def __init__(self, latent_dim, num_sources, output_dim_list):
        super().__init__()
        self.num_sources = num_sources
        self.output_dim_list = output_dim_list
        self.num_outputs = len(self.output_dim_list)

        self.clf_head_dict = nn.ModuleDict()
        for i_src in range(self.num_sources):
            self.clf_head_dict[str(i_src)] = nn.ModuleDict()
            for i_tgt in range(self.num_outputs):
                if i_src != i_tgt:
                    self.clf_head_dict[str(i_src)][str(i_tgt)] = \
                        MLP(latent_dim, self.output_dim_list[i_tgt], hsize_list=[latent_dim], activation='ReLU')

    def forward(self, z_list, labels, against_uniform=False):
        total_loss = 0

        aux_dict = {}
        for i_src in self.clf_head_dict.keys():
            for i_tgt in self.clf_head_dict[i_src].keys():
                logits = self.clf_head_dict[i_src][i_tgt](z_list[int(i_src)])

                if against_uniform:
                    label_vectors = torch.ones_like(logits) / logits.shape[1]
                    loss = cross_entropy_with_label_vectors(logits, label_vectors)
                else:
                    loss = F.cross_entropy(logits, labels[:, int(i_tgt)])
                    # label_vectors = F.one_hot(labels[:, int(i_tgt)], num_classes=logits.shape[1])
                    # cross_entropy_with_label_vectors(logits, label_vectors)

                total_loss += (loss / len(self.clf_head_dict[i_src]))

                # log
                aux_dict['cross_cls_acc_{}-{}'.format(i_src, i_tgt)] = \
                    (logits.argmax(dim=1) == labels[:, int(i_tgt)]).float().mean().item()

        return total_loss, aux_dict
