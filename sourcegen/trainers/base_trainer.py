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
import os
import time
import torch
import logging

from sourcegen.datasets.parser import get_dataset_catalog
from sourcegen.evaluator import tracker, Metrics
import sourcegen.utils.misc as misc

__all__ = ['BaseTrainer']


class BaseTrainer(object):
    def __init__(self, args):
        self.args = args
        self.dev = torch.device('cuda', args.device) if args.device >= 0 else torch.device('cpu')

        # Setup Datasets
        self.dataset_catalog = get_dataset_catalog(args.dataset_spec)

        #
        self.model_spec = args.method_spec['model_spec']
        self.learning_rule = args.method_spec['learning_rule']

        # Setup placeholders for model and optimizer
        self.model = None
        self.global_epoch = 0
        self.best_epoch = self.global_epoch

        # Setup logging information
        self.checkpoint_path = os.path.join(self.args.result_dir, 'checkpoints')
        os.makedirs(self.checkpoint_path, exist_ok=True)

        self.stat_tracker = tracker.StatTracker()
        self.metrics_val = Metrics('val', self.dataset_catalog['target_val'][0], num_biases=5)
        self.metrics_test = Metrics('test', self.dataset_catalog['target_test'][0])

        # initialize validation criteria
        self.validation_metric_name = 'metric_open_hm_acc'
        self.validation_comp_fn = np.max
        self.val_logs = None
        self.product_start_epoch = self.learning_rule['product_start_epoch'] \
            if 'product_start_epoch' in self.learning_rule else 0

    def train_epoch(self, epoch):
        raise NotImplementedError()

    @torch.no_grad()
    def validate_epoch(self, epoch):
        raise NotImplementedError()

    @torch.no_grad()
    def test(self):
        raise NotImplementedError()

    def fit(self):
        start_time = time.time()

        # epoch iteration
        for epoch in range(self.learning_rule['num_epochs']):
            logging.info('Train Epoch {}/{}'.format(epoch, self.learning_rule['num_epochs']))
            torch.manual_seed(self.args.training_seed + self.global_epoch)

            # training
            train_logs = self.train_epoch(epoch)
            logging.info(f'Train Logs: {dict(sorted(train_logs.items()))}')

            if ('validation_cycle' not in self.learning_rule) or \
                    (epoch % self.learning_rule['validation_cycle'] == 0) or \
                    (epoch == self.learning_rule['num_epochs'] - 1):
                # validation
                val_data, val_logs = self.validate_epoch(epoch)
                all_pred_dict, all_attr_labels, all_obj_labels = val_data

                # evaluation
                val_metric_logs = self.metrics_val.process(all_pred_dict, all_attr_labels, all_obj_labels)
                self.val_logs = {**val_logs, **val_metric_logs}
                logging.info(f'Val Logs: {dict(sorted(self.val_logs.items()))}')

            # logging
            if not self.stat_tracker.is_init():
                self.stat_tracker.initialize({'train': list(train_logs.keys()), 'val': list(self.val_logs.keys())})
            self.stat_tracker.push_epoch(train_logs, 'train')
            self.stat_tracker.push_epoch(self.val_logs, 'val')
            self.save_checkpoint_with_validation()

            # check termination condition
            logging.info(f'Best epoch so far: {self.best_epoch} (Product Epoch Start at={self.product_start_epoch})')
            if 'term_gap_epochs' in self.learning_rule:
                gap = (self.global_epoch - self.best_epoch)
                max_gap = self.learning_rule['term_gap_epochs']
                if gap >= max_gap:
                    logging.info('Termination criteria reached.')
                    break
                else:
                    logging.info(f'Termination gap = {gap}/{max_gap}')

            self.global_epoch += 1

        # target fitting
        time_elapsed = time.time() - start_time
        logging.info('Training completed in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    def run(self):
        self.model.to(self.dev)

        # load pretrain model
        if self.args.pretrain_model_path is not None:
            checkpoint = torch.load(self.args.pretrain_model_path, map_location=self.dev)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            logging.info('Load pretrain model from {}'.format(self.args.pretrain_model_path))

        # Run training
        self.fit()

        if self.args.test:
            # Load best validation net and run test
            checkpoint_path = os.path.join(self.checkpoint_path, 'best_checkpoint.pt')
            self.load_checkpoint(checkpoint_path, only_weights=True)

            # testing
            logging.info('Run test using best validation network ...')
            test_data, test_logs = self.test()
            all_pred_dict, all_attr_labels, all_obj_labels = test_data

            # evaluation
            test_metric_logs = self.metrics_test.process(all_pred_dict, all_attr_labels, all_obj_labels)
            test_logs = {**test_logs, **test_metric_logs}

            # print
            logging.info(f'Test Logs: {dict(sorted(test_logs.items()))}')

        # Save losses and metric histories to pkl file
        logging.info('Save statistics')
        stat_path = os.path.join(self.args.result_dir, 'stats.pkl')
        misc.save_obj({'stats': self.stat_tracker.phase_val_data_dict}, stat_path)

    def save_checkpoint_with_validation(self):
        validation_metric_values = self.stat_tracker.phase_val_data_dict['val'][self.validation_metric_name]

        current_value = validation_metric_values[self.global_epoch]
        if (self.global_epoch < self.product_start_epoch) or \
                (current_value == self.validation_comp_fn(validation_metric_values[self.product_start_epoch:])):
            self.best_epoch = self.global_epoch

            self.save_checkpoint_at_path(os.path.join(self.checkpoint_path, 'best_checkpoint.pt'))
            logging.info(f'Save checkpoint at epoch {self.global_epoch} with '
                         f'{self.validation_metric_name}={current_value}')

    def load_checkpoint(self, path, only_weights=False):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])

        if not only_weights:
            self.global_epoch = checkpoint['epoch']

    def set_validation_criteria(self, metric_name, comp_fn):
        self.validation_metric_name = metric_name
        self.validation_comp_fn = comp_fn

    def save_checkpoint_at_path(self, path):
        torch.save({
            'global_epoch': self.global_epoch,
            'model_state_dict': self.model.state_dict()
        }, path)
        logging.info(f'Checkpoint saved at {path}')
