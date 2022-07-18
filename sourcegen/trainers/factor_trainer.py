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
import argparse
import logging
from tqdm import tqdm
import torch.optim as optim
import numpy as np
import os

from sourcegen.trainers.base_trainer import BaseTrainer
from sourcegen.utils.training_tools import build_dataloader
from sourcegen.utils.misc import MeanAccumulatorSet
from sourcegen.models.sourcegen.models import SourceGenModel
from sourcegen.datasets.processors import MergedFilledDataset
from sourcegen.evaluator import Metrics


class SourceGenTrainer(BaseTrainer):
    def __init__(self, args):
        super().__init__(args)

        self.model = None
        self.optimizer = None
        self.set_validation_criteria('loss_val', np.min)

    def train_epoch(self, epoch):
        self.model.train()

        dataset_source = self.dataset_catalog['sources_train'][0][0]
        dataset_target = self.dataset_catalog['target_train'][0]
        dataset = MergedFilledDataset(dataset_source, dataset_target)

        trainloader = build_dataloader(self.learning_rule['dataloader_spec']['params'], dataset, shuffle=True)

        mean_accumulators = MeanAccumulatorSet()
        for idx, data in tqdm(enumerate(trainloader), desc="Training...", total=len(trainloader)):
            # ignore unused negative samples
            data[1][4] = data[1][4][:, 0]
            data[1][5] = data[1][5][:, 0]

            # convert to cuda
            data_source = [d.cuda() for d in data[0]]
            data_target = [d.cuda() for d in data[1]]

            loss, _, aux_dict = self.model.forward((data_source, data_target))

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if idx == 0:
                mean_accumulators.reset_name_accumulator_dict({'loss_train', *aux_dict.keys()})
            mean_accumulators.accumulate({'loss_train': loss.item(),
                                          **aux_dict}, data_source[0].shape[0])

        log_dict = mean_accumulators.get_name_mean_dict()
        logging.info('Epoch: {} | Loss: {}'.format(epoch, log_dict['loss_train']))

        return log_dict

    @torch.no_grad()
    def validate_epoch(self, epoch):
        del epoch
        self.model.eval()

        # dataset
        dataset_source = self.dataset_catalog['sources_val'][0][0]
        dataset_target = self.dataset_catalog['target_val'][0]
        dataset_target.set_enable_val_neg(True)
        dataset = MergedFilledDataset(dataset_source, dataset_target)

        # dataloader
        valloader = build_dataloader(self.learning_rule['dataloader_spec']['params'], dataset)

        all_pred, all_attr_lab, all_obj_lab = [], [], []

        mean_accumulators = MeanAccumulatorSet()
        for idx, data in enumerate(valloader):
            # ignore unused negative samples
            data[1][4] = data[1][4][:, 0]
            data[1][5] = data[1][5][:, 0]

            # convert to cuda
            data_source = [d.cuda() for d in data[0]]
            data_target = [d.cuda() for d in data[1]]

            # forward pass
            loss, predictions, aux_dict = self.model((data_source, data_target))

            # log
            attr_truth, obj_truth = data_target[1], data_target[2]
            all_pred.append(predictions)
            all_attr_lab.append(attr_truth)
            all_obj_lab.append(obj_truth)

            if idx == 0:
                mean_accumulators.reset_name_accumulator_dict({'loss_val', *aux_dict.keys()})
            mean_accumulators.accumulate({'loss_val': loss.item() if loss is not None else 0,
                                          **aux_dict}, data_target[0].shape[0])

        # reorganized output from batch-wise to sample-wise
        all_attr_lab = torch.cat(all_attr_lab)
        all_obj_lab = torch.cat(all_obj_lab)

        if type(all_pred[0]) is dict:
            all_pred_output = {}
            for k in all_pred[0].keys():
                all_pred_output[k] = torch.cat(
                    [all_pred[i][k] for i in range(len(all_pred))])
        else:
            all_pred_output = []
            for k in range(len(all_pred[0])):
                all_pred_output.append(
                    torch.cat([all_pred[i][k] for i in range(len(all_pred))])
                )

        log_dict = mean_accumulators.get_name_mean_dict()
        return (all_pred_output, all_attr_lab, all_obj_lab), log_dict

    @torch.no_grad()
    def test(self):
        self.model.eval()

        # forward pass
        testloader = build_dataloader(self.learning_rule['dataloader_spec']['params'],
                                     self.dataset_catalog['target_test'][0])

        all_pred, all_attr_lab, all_obj_lab = [], [], []
        mean_accumulators = MeanAccumulatorSet()
        for idx, data in enumerate(testloader):
            data = [d.cuda() for d in data]
            attr_truth, obj_truth = data[1], data[2]
            _, predictions, aux_dict = self.model(data)

            # log
            all_pred.append(predictions)
            all_attr_lab.append(attr_truth)
            all_obj_lab.append(obj_truth)

            if idx == 0:
                mean_accumulators.reset_name_accumulator_dict({*aux_dict.keys()})
            mean_accumulators.accumulate({**aux_dict}, data[0].shape[0])

        # reorganized output from batch-wise to sample-wise
        all_attr_lab = torch.cat(all_attr_lab)
        all_obj_lab = torch.cat(all_obj_lab)

        if type(all_pred[0]) is dict:
            all_pred_output = {}
            for k in all_pred[0].keys():
                all_pred_output[k] = torch.cat(
                    [all_pred[i][k] for i in range(len(all_pred))])
        else:
            all_pred_output = []
            for k in range(len(all_pred[0])):
                all_pred_output.append(
                    torch.cat([all_pred[i][k] for i in range(len(all_pred))])
                )

        log_dict = mean_accumulators.get_name_mean_dict()
        return (all_pred_output, all_attr_lab, all_obj_lab), log_dict


class FactorTrainer(SourceGenTrainer):
    def __init__(self, args):
        super().__init__(args)

        # set default params
        model_params = self.model_spec['params']
        model_args = argparse.Namespace(**model_params['args'])
        
        if not hasattr(model_args, 'lambda_asso'):
            model_args.lambda_asso = 0
        if not hasattr(model_args, 'asso_reg_type'):
            model_args.asso_reg_type = None
        if not hasattr(model_args, 'association_type'):
            model_args.association_type = "auto"
        if not hasattr(model_args, 'lambda_asso_start_epoch'):
            model_args.lambda_asso_start_epoch = 0
        if not hasattr(model_args, 'lambda_asso_end_epoch'):
            model_args.lambda_asso_end_epoch = None
        if not hasattr(model_args, 'lambda_asso_suppress'):
            model_args.lambda_asso_suppress = 0
        if not hasattr(model_args, 'association_custom_input'):
            model_args.association_custom_input = None

        # create model
        self.model = SourceGenModel(self.dataset_catalog['sources_train'][0][1],
                                    self.dataset_catalog['target_train'][0], model_args).to(self.dev)

        # create optimizer
        xpred_params, main_params, asso_params = [], [], []
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if 'factor_xpred' in name:
                    xpred_params.append(param)
                elif 'factor_pair_association' in name:
                    asso_params.append(param)
                else:
                    main_params.append(param)

        main_params = [{'params': main_params}]

        self.optimizer = optim.Adam(
            main_params, lr=self.learning_rule['lr'], weight_decay=self.learning_rule['wd'])
        self.optimizer_xpred = optim.Adam(
            xpred_params, lr=self.learning_rule['lr'], weight_decay=self.learning_rule['wd'])
        self.optimizer_asso = None if len(asso_params) == 0 else optim.Adam(asso_params,
            lr=self.learning_rule['lr'], weight_decay=self.learning_rule['wd'])

        # override manifold model metrics
        self.metrics_val = Metrics('val', self.dataset_catalog['target_val'][0],
                                   is_model_manifold=False)
        self.metrics_test = Metrics('test', self.dataset_catalog['target_test'][0],
                                    is_model_manifold=False)

    def train_epoch(self, epoch):
        self.model.train()
        self.model.set_train_epoch(epoch)

        dataset_source = self.dataset_catalog['sources_train'][0][0]
        dataset_target = self.dataset_catalog['target_train'][0]
        dataset = MergedFilledDataset(dataset_source, dataset_target)

        trainloader = build_dataloader(self.learning_rule['dataloader_spec']['params'], dataset, shuffle=True)
        len_dataloader = len(trainloader)

        mean_accumulators = MeanAccumulatorSet()
        for idx, data in tqdm(enumerate(trainloader), desc="Training...", total=len(trainloader)):
            # ignore unused negative samples
            data[1][4] = data[1][4][:, 0]
            data[1][5] = data[1][5][:, 0]

            # convert to cuda
            data_source = [d.cuda() for d in data[0]]
            data_target = [d.cuda() for d in data[1]]

            # update step for main network
            loss, _, aux_dict = self.model.forward((data_source, data_target))

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # update step for asso
            if self.optimizer_asso is not None:
                loss, _, aux_dict = self.model.forward((data_source, data_target))
                self.optimizer_asso.zero_grad()
                loss.backward()
                self.optimizer_asso.step()

            # update step for xpred
            img_source, annotations_source = data_source[0], data_source[1]

            if self.model.args.lambda_xpred != 0:
                z_list_source = self.model.encode_img_factors(img_source, 0)[0]
                loss_xpred, _ = self.model.factor_xpred(z_list_source, annotations_source, against_uniform=False)

                self.optimizer_xpred.zero_grad()
                loss_xpred.backward()
                self.optimizer_xpred.step()

            # log
            if idx == 0:
                mean_accumulators.reset_name_accumulator_dict({'loss_train', *aux_dict.keys()})
            mean_accumulators.accumulate({'loss_train': loss.item(),
                                          **aux_dict}, data_source[0].shape[0])

        log_dict = mean_accumulators.get_name_mean_dict()
        logging.info('Epoch: {} | Loss: {}'.format(epoch, log_dict['loss_train']))

        # print association matrix
        logging.info('Current Factor Association Matrix: \n{}'.format(
            self.model.get_factor_pair_association_normalized().cpu().data))

        return log_dict

    @torch.no_grad()
    def test(self):
        test_data, test_logs = super().test()

        return test_data, test_logs
