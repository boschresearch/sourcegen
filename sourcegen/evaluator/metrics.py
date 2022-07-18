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
import torch
import numpy as np
import pickle
import logging
from sklearn.metrics import auc as auc_func

from . import Evaluator
from sourcegen.utils.misc import MaxTrackerSet

import matplotlib
import matplotlib.pyplot as plt
plt.style.use('seaborn')
matplotlib.use('Agg')


def ensure_border_bound(array):
    min_val = min(array[0], array[-1])
    max_val = max(array[0], array[-1])
    return np.clip(array, min_val, max_val)


def correlated_prediction_analysis(pred_attributes, gt_attributes, attr_dims):
    def analyze_sample_subset(pred_attributes_s, gt_attributes_s, sample_indices, attr_dims_s):
        num_attributes = gt_attributes_s.shape[1]
        assert(num_attributes == 2)

        subset_output_dict = {}
        for i in range(num_attributes):
            accuracy_i = (pred_attributes_s[sample_indices, i] == gt_attributes_s[sample_indices, i]).mean()
            subset_output_dict[f'accuracy_{i}'] = accuracy_i

        # Analyze Failure Mode
        num_attrs, num_objs = attr_dims_s
        obj_gt_attr_pred_mat = np.zeros((num_objs, num_attrs), dtype=int)
        attr_gt_obj_pred_mat = np.zeros((num_attrs, num_objs), dtype=int)

        for gt_attr, pred_attr in zip(gt_attributes_s[sample_indices], pred_attributes_s[sample_indices]):
            obj_gt_attr_pred_mat[gt_attr[1], pred_attr[0]] += 1
            attr_gt_obj_pred_mat[gt_attr[0], pred_attr[1]] += 1

        subset_output_dict['obj_gt_attr_pred_mat'] = obj_gt_attr_pred_mat
        subset_output_dict['attr_gt_obj_pred_mat'] = attr_gt_obj_pred_mat
        return subset_output_dict

    group_analysis_dict = {}

    # correlated
    corr_sample_indices = [i for i, attr in enumerate(gt_attributes) if len(set(attr)) == 1]
    if len(corr_sample_indices) > 0:
        group_analysis_dict['correlated'] = \
            analyze_sample_subset(pred_attributes, gt_attributes, corr_sample_indices, attr_dims)

    # uncorrelated
    uncorr_sample_indices = [i for i, attr in enumerate(gt_attributes) if len(set(attr)) > 1]
    if len(uncorr_sample_indices) > 0:
        group_analysis_dict['uncorrelated'] = \
            analyze_sample_subset(pred_attributes, gt_attributes, uncorr_sample_indices, attr_dims)

    # all
    all_indices = np.arange(pred_attributes.shape[0]).tolist()
    group_analysis_dict['all'] = analyze_sample_subset(pred_attributes, gt_attributes, all_indices, attr_dims)

    return group_analysis_dict


class Metrics(object):
    def __init__(self, phase_name, dataset, is_model_manifold=True, num_biases=50):
        self.phase_name = phase_name
        self.is_model_manifold = is_model_manifold
        self.evaluator = Evaluator(dataset, self.is_model_manifold)

        self.min_bias = -20
        self.max_bias = 20
        self.num_biases = num_biases

    def process(self, all_pred_dict, all_attr_labels, all_obj_labels):
        accuracy_keys = ['open_seen_acc', 'open_unseen_acc', 'open_hm_acc', 'closed_acc']
        metrics_dict = {}

        # compute accuracies (without bias)
        accuracy_dict, _ = self.get_accuracies_with_bias(all_pred_dict, all_attr_labels, all_obj_labels, 0)
        for key in accuracy_keys:
            metrics_dict[key] = accuracy_dict[key]

        # compute accuracies (with bias)
        max_trackers_wb = MaxTrackerSet(accuracy_keys)
        open_seen_acc_list_wb, open_unseen_acc_list_wb = [], []

        bias_list = self._get_bias_list()
        for bias in bias_list:
            accuracies_dict_w_bias, _ = self.get_accuracies_with_bias(all_pred_dict, all_attr_labels, all_obj_labels, bias)
            max_trackers_wb.update(accuracies_dict_w_bias)

            open_seen_acc_list_wb.append(accuracies_dict_w_bias['open_seen_acc'])
            open_unseen_acc_list_wb.append(accuracies_dict_w_bias['open_unseen_acc'])
        open_seen_acc_arr_wb = ensure_border_bound(np.array(open_seen_acc_list_wb))
        open_unseen_acc_arr_wb = ensure_border_bound(np.array(open_unseen_acc_list_wb))
        self._adjust_biases(open_seen_acc_arr_wb, open_unseen_acc_arr_wb, bias_list)

        # compute AUSUC
        metrics_dict_wb = {f'{key}_wb': value for key, value in max_trackers_wb.get_name_max_dict().items()}
        metrics_dict = dict({**metrics_dict, **metrics_dict_wb})
        metrics_dict['ausuc'] = auc_func(open_seen_acc_arr_wb, open_unseen_acc_arr_wb)

        # add metric as prefix
        keys = list(metrics_dict.keys())
        for key in keys:
            metrics_dict[f'metric_{key}'] = metrics_dict.pop(key)

        return metrics_dict

    def get_accuracies_with_bias(self, all_pred_dict, all_attr_labels, all_obj_labels, bias):
        if self.is_model_manifold:
            scores_dict = self.evaluator.score_model(all_pred_dict, all_obj_labels, bias=bias)
        else:
            scores_dict = self.evaluator.score_model(all_pred_dict, all_obj_labels, bias=bias)
        match_stats = self.evaluator.evaluate_predictions(scores_dict, all_attr_labels, all_obj_labels)

        accuracies = list(map(torch.mean, map(torch.cat, zip(match_stats))))
        _, _, closed_acc, _, _, open_seen_acc, open_unseen_acc = accuracies

        if torch.isnan(open_unseen_acc):
            open_hm_acc = open_seen_acc
        else:
            open_hm_acc = (2.0 / (1.0 / open_seen_acc + 1.0 / open_unseen_acc))

        return {
            'open_seen_acc': open_seen_acc.item(),
            'open_unseen_acc': open_unseen_acc.item(),
            'open_hm_acc': open_hm_acc.item(),
            'closed_acc': closed_acc.item()
        }, scores_dict

    def _get_bias_list(self):
        bias_step = (self.max_bias - self.min_bias) / self.num_biases
        return np.arange(self.min_bias, self.max_bias, bias_step)

    def _adjust_biases(self, open_seen_acc_arr, open_unseen_acc_arr, bias_list):
        if self.is_model_manifold:
            min_index = 0
            min_search_seen = np.where(open_seen_acc_arr < open_seen_acc_arr[0])[0]
            min_search_unseen = np.where(open_unseen_acc_arr > open_unseen_acc_arr[0])[0]
            min_search = np.concatenate((min_search_seen, min_search_unseen))
            if min_search.size > 0:
                min_index = max(min_index, min_search.min() - 1)

            max_index = len(bias_list - 1)
            max_search_seen = np.where(open_seen_acc_arr > open_seen_acc_arr[-1])[0]
            max_search_unseen = np.where(open_unseen_acc_arr < open_unseen_acc_arr[-1])[0]
            max_search = np.concatenate((max_search_seen, max_search_unseen))
            if max_search.size > 0:
                max_index = min(max_index, max_search.max() + 1)

            # adjustment
            if min_index != 0 or max_index != len(bias_list - 1):
                self.min_bias = bias_list[min_index]
                self.max_bias = bias_list[max_index]

                # extend border by 10%
                margin = (self.max_bias - self.min_bias) * 0.1
                self.min_bias = self.min_bias - margin
                self.max_bias = self.max_bias + margin
