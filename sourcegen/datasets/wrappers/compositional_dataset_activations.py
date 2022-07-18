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

# This source code is derived from attributes-as-operators
#   (https://github.com/Tushar-N/attributes-as-operators/blob/master/data/dataset.py)
# Copyright (c) 2018, licensed under the MIT license,
# cf. 3rd-party-licenses.txt file in the root directory of this source tree.

import numpy as np
import torch.utils.data as tdata
import torchvision.transforms as transforms
from PIL import Image

import tqdm
import torchvision.models as tmodels
import torch.nn as nn
import torch

import os


class ImageLoader:
    def __init__(self, root):
        self.img_dir = root

    def __call__(self, img):
        file = '%s/%s' % (self.img_dir, img)
        img = Image.open(file).convert('RGB')
        return img


def imagenet_transform(phase, keep_aspect_ratio=True):
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

    if keep_aspect_ratio:
        if phase == 'train':
            transform = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])
        elif phase == 'test' or phase == 'val':
            transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])
    else:
        if phase == 'train':
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])
        elif phase == 'test' or phase == 'val':
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])

    return transform


def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]

#------------------------------------------------------------------------------------------------------------------------------------#


class CompositionDataset(tdata.Dataset):
    def __init__(self, root, phase, split='compositional-split',
            subset=False,
            random_seed=2533,
            load_image=True
    ):
        self.root = root
        self.phase = phase
        self.split = split
        self.random_state = np.random.RandomState(random_seed)
        self.load_image=load_image

        self.feat_dim = None
        self.transform = imagenet_transform(phase)
        self.loader = ImageLoader(self.root+'/images/')

        self.attrs, self.objs, self.pairs, self.train_pairs, self.val_pairs, self.test_pairs = self.parse_split()

        self.train_data, self.val_data, self.test_data = self.get_split_info()
        if self.phase == 'train':
            self.data = self.train_data
        elif self.phase == 'val':
            self.data = self.val_data
        else:
            self.data = self.test_data
        if subset:
            ind = np.arange(len(self.data))
            ind = ind[::len(ind) // 1000]
            self.data = [self.data[i] for i in ind]

        self.obj2idx = {obj: idx for idx, obj in enumerate(self.objs)}
        self.attr2idx = {attr: idx for idx, attr in enumerate(self.attrs)}
        self.pair2idx = {pair: idx for idx, pair in enumerate(self.pairs)}

        print('# train pairs: %d | # val pairs: %d | # test pairs: %d' % (len(
            self.train_pairs), len(self.val_pairs), len(self.test_pairs)))
        print('# train images: %d | # val images: %d | # test images: %d' %
              (len(self.train_data), len(self.val_data), len(self.test_data)))

        # fix later -- affordance thing
        # return {object: all attrs that occur with obj}
        self.obj_affordance = {}
        self.train_obj_affordance = {}
        for _obj in self.objs:
            candidates = [attr for (_, attr, obj) in self.train_data+self.val_data+self.test_data if obj==_obj]
            self.obj_affordance[_obj] = sorted(list(set(candidates)))

            candidates = [attr for (_, attr, obj) in self.train_data if obj==_obj]
            self.train_obj_affordance[_obj] = sorted(list(set(candidates)))

        self.sample_indices = list(range(len(self.data)))
        self.sample_pairs = self.train_pairs

    def get_split_info(self):
        data = torch.load(self.root+'/metadata_{}.t7'.format(self.split))
        train_data, val_data, test_data = [], [], []
        for instance in data:
            image, attr, obj, settype = instance['image'], instance['attr'], instance['obj'], instance['set']

            if attr == 'NA' or (attr,
                                obj) not in self.pairs or settype == 'NA':
                # ignore instances with unlabeled attributes
                # ignore instances that are not in current split
                continue

            data_i = [image, attr, obj]
            if settype == 'train':
                train_data.append(data_i)
            elif settype == 'val':
                val_data.append(data_i)
            else:
                test_data.append(data_i)

        return train_data, val_data, test_data

    def parse_split(self):
        def parse_pairs(pair_list):
            with open(pair_list, 'r') as f:
                pairs = f.read().strip().split('\n')
                pairs = [t.split() for t in pairs]
                pairs = list(map(tuple, pairs))

                if len(pairs) == 1 and len(pairs[0]) == 0:
                    pairs = []

            if len(pairs) > 0:
                attrs, objs = zip(*pairs)
            else:
                attrs, objs = (), ()

            return attrs, objs, pairs

        tr_attrs, tr_objs, tr_pairs = parse_pairs('%s/%s/train_pairs.txt'%(self.root, self.split))
        vl_attrs, vl_objs, vl_pairs = parse_pairs('%s/%s/val_pairs.txt'%(self.root, self.split))
        ts_attrs, ts_objs, ts_pairs = parse_pairs('%s/%s/test_pairs.txt'%(self.root, self.split))

        all_attrs, all_objs = sorted(list(set(tr_attrs + vl_attrs + ts_attrs))), sorted(list(set(tr_objs + vl_objs + ts_objs)))
        all_pairs = sorted(list(set(tr_pairs + vl_pairs + ts_pairs)))

        return all_attrs, all_objs, all_pairs, tr_pairs, vl_pairs, ts_pairs

    def __getitem__(self, index):
        index = self.sample_indices[index]
        image, attr, obj = self.data[index]

        if self.load_image:
            img = self.loader(image)
            img = self.transform(img)
        else:
            img = None

        data = [img, self.attr2idx[attr], self.obj2idx[obj], self.pair2idx[(attr, obj)]]

        return data

    def __len__(self):
        return len(self.sample_indices)


#------------------------------------------------------------------------------------------------------------------------------------#


class CompositionDatasetActivations(CompositionDataset):
    def __init__(self, root, phase, split, subset=False, random_seed=2533):
        super(CompositionDatasetActivations, self).__init__(
            root, phase, split, subset=subset, random_seed=random_seed, load_image=False)

        # precompute the activations -- weird. Fix pls
        feat_file = '{}/features_{}.t7'.format(root, split) #'%s/features.t7'%root
        if not os.path.exists(feat_file):
            with torch.no_grad():
                self.generate_features(feat_file)

        activation_data = torch.load(feat_file)
        self.activations = dict(zip(activation_data['files'], activation_data['features']))
        self.feat_dim = activation_data['features'].size(1)
        self.activate = True
        self.transform = imagenet_transform('test')

        print('%d activations loaded'%(len(self.activations)))

    def generate_features(self, out_file):

        data = self.train_data + self.val_data + self.test_data
        feat_extractor = tmodels.resnet18(pretrained=True)
        feat_extractor.fc = nn.Sequential()
        feat_extractor.eval().cuda()

        image_feats = []
        image_files = []
        for chunk in tqdm.tqdm(
                chunks(data, 512), total=len(data) // 512):
            files, attrs, objs = zip(*chunk)
            imgs = list(map(self.loader, files))
            imgs = list(map(self.transform, imgs))
            feats = feat_extractor(torch.stack(imgs, 0).cuda())
            image_feats.append(feats.data.cpu())
            image_files += files
        image_feats = torch.cat(image_feats, 0)
        print('features for %d images generated' % (len(image_files)))

        torch.save({'features': image_feats, 'files': image_files}, out_file)

    def __getitem__(self, index):
        data = super(CompositionDatasetActivations, self).__getitem__(index)
        if self.activate:
            index = self.sample_indices[index]
            image, attr, obj = self.data[index]
            data[0] = self.activations[image]
        return data

    def set_activate(self, val):
        if val:
            self.activate = True
            self.load_image = False
        else:
            self.activate = False
            self.load_image = True
