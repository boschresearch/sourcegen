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
import numpy as np
import tqdm
import logging
import PIL
import pickle

import torch
import torch.nn as nn
import torchvision.models as tmodels
import torchvision.transforms as transforms

from ..wrappers.compositional_dataset_activations import imagenet_transform

from .base_dataset import BaseDataset
from ..utils import get_all_attr_cls_ids_list, get_comp_pair_dir


def load_pairs_pkl(filename):
    with open(filename, 'rb') as f:
        pairs = pickle.load(f)

    # convert to string
    pairs = [(str(attr), str(obj)) for attr, obj in pairs]

    return pairs

# Convert normal dataset to be similar to CompositionDataset
class ConvertedCompositionDataset(BaseDataset):
    def __init__(self, parent_dataset, ds_x_index, ds_attribute_index, root, phase,
                 is_pair_parent_dir=False):
        super().__init__(parent_dataset)
        self.ds_x_index = ds_x_index
        self.ds_attribute_index = ds_attribute_index
        self.root = root
        self.phase = phase

        # attrs & objs
        all_attr_cls_ids_list = get_all_attr_cls_ids_list(parent_dataset, self.ds_attribute_index)
        self.attrs = [self.convert_parent_attr(p_attr) for p_attr in all_attr_cls_ids_list[0]]
        self.objs = [self.convert_parent_obj(p_obj) for p_obj in all_attr_cls_ids_list[1]]

        # attr2idx & obj2idx
        self.attr2idx = {attr: idx for idx, attr in enumerate(self.attrs)}
        self.obj2idx = {obj: idx for idx, obj in enumerate(self.objs)}

        # pairs
        self.pairs = [(attr, obj) for obj in self.objs for attr in self.attrs]
        self.pair2idx = {pair: idx for idx, pair in enumerate(self.pairs)}

        # train/val/test pairs
        pair_dir = get_comp_pair_dir({'root': root, 'is_pair_parent_dir': is_pair_parent_dir})
        self.train_pairs = load_pairs_pkl(os.path.join(pair_dir, 'train_pairs.pkl'))
        self.val_pairs = load_pairs_pkl(os.path.join(pair_dir, 'val_pairs.pkl'))
        self.test_pairs = load_pairs_pkl(os.path.join(pair_dir, 'test_pairs.pkl'))

        # affordance (all attributes which occur with the objects)
        self.obj_affordance = {obj: self.attrs for obj in self.objs}
        self.train_obj_affordance = self.obj_affordance

        # sample indices
        self.sample_indices = list(range(len(self.parent_dataset)))
        self.sample_pairs = self.train_pairs

    def __getitem__(self, index):
        index = self.sample_indices[index]
        parent_data = self.parent_dataset[index]
        img = parent_data[self.ds_x_index]
        attr = self.convert_parent_attr(parent_data[self.ds_attribute_index][0])
        obj = self.convert_parent_obj(parent_data[self.ds_attribute_index][1])

        data = [img, self.attr2idx[attr], self.obj2idx[obj], self.pair2idx[(attr, obj)]]

        return data

    def __len__(self):
        return len(self.sample_indices)

    def convert_parent_attr(self, parent_attr):
        return str(int(parent_attr))

    def convert_parent_obj(self, parent_obj):
        return str(int(parent_obj))

# Convert normal dataset to be similar to CompositionDatasetActivations
class ConvertedCompositionActivationsDataset(ConvertedCompositionDataset):
    def __init__(self, parent_dataset, ds_x_index, ds_attribute_index, root, phase,
                 use_imagenet_transform=True, transform_keep_aspect_ratio=True, is_pair_parent_dir=False):
        super().__init__(
            parent_dataset, ds_x_index, ds_attribute_index, root, phase,
            is_pair_parent_dir=is_pair_parent_dir)

        # precompute the activations -- weird. Fix pls
        self.use_imagenet_transform = use_imagenet_transform
        self.transform_keep_aspect_ratio = transform_keep_aspect_ratio
        self.transform = imagenet_transform('test', keep_aspect_ratio=self.transform_keep_aspect_ratio) \
            if self.use_imagenet_transform else None

        if self.use_imagenet_transform:
            feat_file = '{}/features_{}.t7'.format(root, phase)
            if not os.path.exists(feat_file):
                with torch.no_grad():
                    self.generate_features(feat_file)

            activation_data = torch.load(feat_file)
            self.activations = activation_data['features']
            self.feat_dim = self.activations[0].shape[0]
            logging.info('{} activations loaded from {}'.format(len(self.activations), feat_file))
        self.activate = True

    def generate_features(self, out_file):
        feat_extractor = tmodels.resnet18(pretrained=True)
        feat_extractor.fc = nn.Sequential()
        feat_extractor.eval().cuda()

        image_feats = []
        for i in tqdm.tqdm(range(len(self.parent_dataset)), desc='Extracting features...'):
            img = self.parent_dataset[i][self.ds_x_index]
            if not isinstance(img, PIL.Image.Image):
                img = transforms.ToPILImage()(img)
            img = self.transform(img).unsqueeze(0)

            with torch.no_grad():
                feat = feat_extractor(img.cuda())

            image_feats.append(feat.data.cpu())
        image_feats = torch.cat(image_feats, 0)
        logging.info('features for %d images generated' % (len(image_feats)))

        os.makedirs(os.path.dirname(out_file), exist_ok=True)
        torch.save({'features': image_feats}, out_file)
        logging.info('save generated features to {}'.format(out_file))

    def __getitem__(self, index):
        data = super(ConvertedCompositionActivationsDataset, self).__getitem__(index)
        index = self.sample_indices[index]
        if self.activate:
            data[0] = self.activations[index]
        else:
            if self.transform is not None:
                img = data[0]
                if not isinstance(img, PIL.Image.Image):
                    img = transforms.ToPILImage()(img)
                data[0] = self.transform(img)
        return data

    def set_activate(self, val):
        self.activate = val
