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
import os
import numpy as np
from torchvision.models import resnet18
from torch.distributions import Categorical

from sourcegen.utils.misc import MeanAccumulatorSet
from . import cross_predictor


def compute_entropy(probs):
    e = Categorical(probs=probs).entropy()
    return e


# factor-pair association (n_factors x 2)
# shape, hue, texture, position, scale
def get_factor_pair_association(dset_target, dset_source_config, association_type, association_custom_input):
    valid_target_names = ['color_animal', 'color_fruit', 'color_fashion']
    min_target_length = min([len(name) for name in valid_target_names])

    root = dset_target.root
    dset_target_name = None
    while len(root) >= min_target_length:
        tname = os.path.basename(root)

        if tname in valid_target_names:
            dset_target_name = tname
            break
        root = os.path.dirname(root)

    if dset_target_name is None:
        raise RuntimeError('Cannot resolve target name from the root {}.'.format(dset_target.root))

    source_type_list = dset_source_config['ds_attritube_type_list']
    source_type_id_dict = {type: id for id, type in enumerate(source_type_list)}
    n_factors = len(source_type_list)

    # learn
    if association_type == "learn":
        return nn.Parameter(torch.zeros((n_factors, 2)), requires_grad=True)
    elif association_type == "custom":
        assert(association_custom_input is not None)
        factor_pair_association = torch.zeros((n_factors, 2))
        for target_id, factor_id in enumerate(association_custom_input):
            factor_pair_association[factor_id, target_id] = 1
        return nn.Parameter(factor_pair_association, requires_grad=False)

    # manual
    factor_pair_association = torch.zeros((n_factors, 2))
    if dset_target_name in ['color_fruit']:
        # (attr, obj) ~ (hue, shape)
        factor_pair_association[source_type_id_dict['hue'], 0] = 1.0
        factor_pair_association[source_type_id_dict['shape'], 1] = 1.0
    elif 'color_animal' in dset_target_name:
        for i, task in enumerate(dset_target.parent_dataset.parent_dataset.dataset_raw.tasks):
            factor_pair_association[source_type_id_dict[task], i] = 1.0
    elif dset_target_name in ['color_fashion']:
        factor_pair_association[source_type_id_dict['hue'], 0] = 1.0
        factor_pair_association[source_type_id_dict['shape'], 1] = 1.0

    if association_type == "swap":
        factor_pair_association[:, [0, 1]] = factor_pair_association[:, [1, 0]]
    elif association_type == "random":
        factor_pair_association *= 0
        associated_factors = np.random.choice(n_factors, size=2)
        factor_pair_association[associated_factors[0], 0] = 1
        factor_pair_association[associated_factors[1], 1] = 1

    return nn.Parameter(factor_pair_association, requires_grad=False)


class MLP(nn.Module):
    def __init__(self, inp_dim, out_dim, num_layers=1, relu=True, bias=True, bn_mode="none"):
        super(MLP, self).__init__()
        self.bn_mode = bn_mode

        mod = []
        for _ in range(num_layers-1):
            mod.append(nn.Linear(inp_dim, inp_dim, bias=bias))
            if self.bn_mode == 'standard':
                mod.append(nn.BatchNorm1d(inp_dim))
            mod.append(nn.ReLU(True))

        mod.append(nn.Linear(inp_dim, out_dim, bias=bias))
        if relu:
            mod.append(nn.ReLU(True))

        self.mod = nn.ModuleList(mod)

    def forward(self, x, domain=None):
        if self.bn_mode == 'domain':
            assert(domain is not None)

        output = x
        for m in self.mod:
            output = m(output)
        return output


class SourceGenModel(nn.Module):
    def __init__(self, dset_source_config, dset_target, args):
        super().__init__()

        self.dset_source_config = dset_source_config
        self.dset_target = dset_target
        self.args = args

        self.img_feat_dim = self.dset_target.feat_dim
        self.latent_dim = self.args.latent_dim
        self.factor_dim_list = self.dset_source_config['attr_dim_list']
        self.num_factors = len(self.factor_dim_list)
        self.train_epoch = None

        #
        if not self.dset_target.activate:
            resnet_full = resnet18(pretrained=True)
            resnet_without_fcn = list(resnet_full.children())[:-1]
            self.image_embedder = nn.Sequential(
                *resnet_without_fcn,
                nn.Flatten())
        else:
            self.image_embedder = nn.Identity()

        # encoder
        self.factor_encoder_list = nn.ModuleList(
            [MLP(self.img_feat_dim, self.latent_dim, 3, relu=False, bn_mode=self.args.bn_mode) for _ in range(self.num_factors)]
        )

        # source
        self.factor_clf_source_list = nn.ModuleList(
            [MLP(self.latent_dim, out_dim, 2, relu=False) for out_dim in self.factor_dim_list]
        )

        # target
        self.attr_clf_target = MLP(self.latent_dim, len(self.dset_target.attrs), 2, relu=False)
        self.obj_clf_target = MLP(self.latent_dim, len(self.dset_target.objs), 2, relu=False)

        # factor-pair association (n_factors x 2)
        self.factor_pair_association = get_factor_pair_association(self.dset_target, self.dset_source_config,
                                                                   self.args.association_type,
                                                                   self.args.association_custom_input)

        # xpred
        self.factor_xpred = cross_predictor.CrossFactorPredictorModule(
            self.latent_dim, self.num_factors, self.factor_dim_list)

    def train_forward(self, x):
        x_source, x_target = x

        img_source = x_source[0]
        annotations_source = x_source[1]
        img_target = x_target[0]
        annotations_target = (x_target[1], x_target[2])

        # encode (source)
        z_list_source, img_feat_source = self.encode_img_factors(img_source, 0)

        # data loss (source)
        class_logits_list_source = [clf(z) for clf, z in zip(self.factor_clf_source_list, z_list_source)]

        loss_data_source = 0
        for i in range(self.num_factors):
            loss_data_source += F.cross_entropy(class_logits_list_source[i], annotations_source[:, i])
        loss_data_source /= self.num_factors
        loss_data_source = self.args.lambda_data_source * loss_data_source

        # encode (target)
        z_list_target, _ = self.encode_img_factors(img_target, 1)

        # data loss (target)
        if self.args.isolated_latents:
            z_list_target_to_clf = [z.detach() for z in z_list_target]
        else:
            z_list_target_to_clf = z_list_target
        attr_logits, obj_logits, z_ao_target = self.predict_ao(z_list_target_to_clf)

        attr_loss = F.cross_entropy(attr_logits, annotations_target[0])
        obj_loss = F.cross_entropy(obj_logits, annotations_target[1])
        loss_data_target = (attr_loss + obj_loss) / 2

        # independency loss : xpred
        loss_xpred = torch.zeros_like(loss_data_target)
        aux_dict_xpred = {}
        if self.args.lambda_xpred != 0:
            loss_xpred, aux_dict_xpred = self.factor_xpred(z_list_source, annotations_source, against_uniform=True)
            loss_xpred = self.args.lambda_xpred * loss_xpred

        # association loss
        lambda_asso = self.args.lambda_asso
        if self.args.lambda_asso_end_epoch is not None:
            if self.train_epoch < self.args.lambda_asso_start_epoch:
                lambda_asso = 0
            else:
                lambda_asso *= min((self.train_epoch - self.args.lambda_asso_start_epoch) /
                                   (self.args.lambda_asso_end_epoch - self.args.lambda_asso_start_epoch), 1.0)

        loss_asso = torch.zeros_like(loss_data_target)
        asso_mat = self.get_factor_pair_association_normalized()
        if lambda_asso != 0:
            # regularization
            if self.args.asso_reg_type == 'entropy':
                for j in range(asso_mat.shape[1]):
                    loss_asso += compute_entropy(asso_mat[:, j])
            elif self.args.asso_reg_type.split('-')[0] == 'l':
                l_value = float(self.args.asso_reg_type.split('-')[1])
                loss_asso = torch.pow(asso_mat, l_value).sum()
            else:
                raise RuntimeError('Unexpected asso_reg_type {}'.format(self.args.asso_reg_type))

            loss_asso = lambda_asso * loss_asso

        # association suppression loss
        lambda_asso_suppress = self.args.lambda_asso_suppress
        loss_asso_suppress = torch.zeros_like(loss_data_target)
        if lambda_asso_suppress > 0:
            T = 1.0 / (asso_mat.shape[0] - 2)
            for i in range(asso_mat.shape[0]):
                j_max = asso_mat[i, :].argmax().item()
                val_max = asso_mat[i, j_max]
                if val_max >= T:
                    loss_asso_suppress += (asso_mat[i, :].sum() - val_max)*(val_max.detach() - T)
            loss_asso_suppress *= lambda_asso_suppress

        # total loss
        loss = loss_data_target + loss_data_source + loss_xpred + loss_asso + loss_asso_suppress

        aux_dict = {'loss_data_target': loss_data_target.item(),
                    'loss_data_source': loss_data_source.item(),
                    'loss_xpred': loss_xpred.item(),
                    'loss_asso': loss_asso.item(),
                    'loss_asso_suppress': loss_asso_suppress.item(),
                    **aux_dict_xpred}

        # factor accuracy
        for i in range(self.num_factors):
            acc = (class_logits_list_source[i].argmax(dim=1) == annotations_source[:, i]).float().mean().item()
            aux_dict[f'factor_{i}_acc'] = acc

        # pair accuracy
        attr_acc = (attr_logits.argmax(dim=1) == annotations_target[0]).float().mean().item()
        obj_acc = (obj_logits.argmax(dim=1) == annotations_target[1]).float().mean().item()
        aux_dict['attr_acc'] = attr_acc
        aux_dict['obj_acc'] = obj_acc

        return loss, [F.softmax(attr_logits, dim=1), F.softmax(obj_logits, dim=1)], aux_dict

    def val_forward(self, x_target):
        img_target = x_target[0]
        annotations_target = (x_target[1], x_target[2])

        # encode (target)
        z_list_target, _ = self.encode_img_factors(img_target, 1)

        # data loss (target)
        attr_logits, obj_logits, _ = self.predict_ao(z_list_target)
        attr_pred = F.softmax(attr_logits, dim=1)
        obj_pred = F.softmax(obj_logits, dim=1)

        # pair accuracy
        aux_dict = {}
        attr_acc = (attr_pred.argmax(dim=1) == annotations_target[0]).float().mean().item()
        obj_acc = (obj_pred.argmax(dim=1) == annotations_target[1]).float().mean().item()
        aux_dict['attr_acc'] = attr_acc
        aux_dict['obj_acc'] = obj_acc

        return None, [attr_pred, obj_pred], aux_dict

    def forward(self, x):
        if self.training:
            loss, pred, aux_dict = self.train_forward(x)
        else:
            with torch.no_grad():
                if isinstance(x, tuple):
                    loss, pred, aux_dict = self.train_forward(x)
                else:
                    loss, pred, aux_dict = self.val_forward(x)
        return loss, pred, aux_dict

    # domain = 0 (source) or 1 (target)
    def encode_img_factors(self, img, domain):
        feat = self.image_embedder(img)
        z_list = [encoder(feat, domain) for encoder in self.factor_encoder_list]
        return z_list, feat

    def predict_ao(self, z_list_target):
        z_ao_target = self.extract_z_ao(z_list_target)

        attr_logits = self.attr_clf_target(z_ao_target[:, :, 0])
        obj_logits = self.obj_clf_target(z_ao_target[:, :, 1])
        return attr_logits, obj_logits, z_ao_target

    def extract_z_ao(self, z_list_target):
        z_stack_target = torch.stack(z_list_target, dim=2)
        z_ao_target = torch.matmul(z_stack_target, self.get_factor_pair_association_normalized())

        return z_ao_target

    def get_factor_pair_association_normalized(self):
        if not self.factor_pair_association.requires_grad:
            return self.factor_pair_association
        return nn.functional.softmax(self.factor_pair_association, dim=0)

    def set_train_epoch(self, epoch):
        self.train_epoch = epoch
