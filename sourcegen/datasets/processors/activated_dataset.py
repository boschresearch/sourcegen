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
import tqdm
import logging

import torch
import torchvision.models as tmodels
import torchvision.transforms as transforms
import torch.nn as nn

from .base_dataset import BaseDataset
from sourcegen.datasets.wrappers.compositional_dataset_activations import imagenet_transform


class ActivatedDataset(BaseDataset):
    def __init__(self, parent_dataset, ds_x_index, root):
        super().__init__(parent_dataset)

        self.ds_x_index = ds_x_index
        self.transform = imagenet_transform('test')

        # precompute the activations
        feat_file = '{}/features.t7'.format(root)
        if not os.path.exists(feat_file):
            with torch.no_grad():
                self.generate_features(feat_file)

        activation_data = torch.load(feat_file)
        self.activations = activation_data['features']
        self.feat_dim = self.activations[0].shape[0]
        self.activate = True

        logging.info('{} activations loaded from {}'.format(len(self.activations), feat_file))

    def __getitem__(self, i):
        data = list(self.parent_dataset[i])

        if self.activate:
            data[self.ds_x_index] = self.activations[i]
        else:
            x = transforms.ToPILImage()(data[self.ds_x_index])
            data[self.ds_x_index] = self.transform(x)

        return tuple(data)

    def __len__(self):
        return len(self.parent_dataset)

    def generate_features(self, out_file):
        feat_extractor = tmodels.resnet18(pretrained=True)
        feat_extractor.fc = nn.Sequential()
        feat_extractor.eval().cuda()

        image_feats = []
        for i in tqdm.tqdm(range(len(self.parent_dataset)), desc='Extracting features...'):
            img = transforms.ToPILImage()(self.parent_dataset[i][self.ds_x_index])
            img = self.transform(img).unsqueeze(0)

            with torch.no_grad():
                feat = feat_extractor(img.cuda())

            image_feats.append(feat.data.cpu())
        image_feats = torch.cat(image_feats, 0)
        logging.info('features for %d images generated' % (len(image_feats)))

        os.makedirs(os.path.dirname(out_file), exist_ok=True)
        torch.save({'features': image_feats}, out_file)
        logging.info('save generated features to {}'.format(out_file))

    def set_activate(self, val):
        self.activate = val
