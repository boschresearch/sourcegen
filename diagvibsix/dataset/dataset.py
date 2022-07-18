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
# Author: Elias Eulig, Piyapat Saranrittichai, Volker Fischer
# -*- coding: utf-8 -*-

import os
import copy
import numpy as np
from tqdm import tqdm
from .config import DATASETS
from .paint_images import Painter
from ..utils.dataset_utils import sample_attribute
from ..utils import load_obj, save_obj


def random_choice(attr):
    """
    attr may be a list or a single attribute. If it is a list, a random choice of that list is returned, if it is
    a single attribute, that attribute is returned
    """
    if isinstance(attr, list):
        return np.random.choice(attr)
    else:
        return attr


def get_answer(semantic_image_spec, question):
    """ Returns the attribute for a certain attribute type of an object.

    Attribute types are e.g. 'category', 'class', 'style'.
    """
    # We never have more than one object and questions can't be ambiguous (otherwise the image couldn't be generated in
    # the first place
    if question == 'category':
        return semantic_image_spec['objs'][0][question].split()[0]
    elif question == 'bg_lightness':
        return semantic_image_spec[question]
    else:           
        return semantic_image_spec['objs'][0][question]


def merge_datasets(dataset_list, overall_samples=None):
    """ Merge all datasets in a list into a single dataset.

        The merged dataset will have as many samples, as all input datasets combined.

        The modes in the merged dataset will simply be the
        concatenation of all modes.

        Each mode will have the same samples as before. Hence, the samples per mode will not change.
        For this the ratio of every single mode is adapted wrt the combined number of samples.

        For now, assume same shape item for all datasets.
    """
    # Start with empty dataset.
    ds_spec = dict()
    ds_spec['samples'] = 0
    ds_spec['modes'] = []
    # Assume same shape over all datasets in list.
    ds_spec['shape'] = dataset_list[0]['shape']
    # Aggregate overall samples.
    if overall_samples is None:
        for ds in dataset_list:
            ds_spec['samples'] += ds['samples']
    else:
        ds_spec['samples'] = overall_samples
    # Concatenate all modes of all datasets and adapt mode ratios.
    for ds in dataset_list:
        for mode in ds['modes']:
            ds_spec['modes'].append(copy.deepcopy(mode))
            # Adapt number of samples of this mode to merged dataset.
            this_mode_samples = float(ds['samples']) * mode['ratio']
            merged_ratio = this_mode_samples / float(ds_spec['samples'])
            ds_spec['modes'][-1]['ratio'] = merged_ratio
    return ds_spec


class Dataset(object):
    """ A dataset is a collection of modes.
    """
    def __init__(self, dataset_spec, seed, cache_path=None):
        self.questions_answers = None
        np.random.seed(seed)

        # The specification of the dataset.
        self.spec = copy.deepcopy(dataset_spec)

        # The tasks
        self.tasks = dataset_spec['tasks']

        # Setup painter
        self.painter = Painter()

        if (cache_path is not None) and (os.path.exists(cache_path)):
            # load cache
            cache_data = load_obj(cache_path)
            self.images = cache_data['images']
            self.image_specs = cache_data['image_specs']
            self.tasks_labels = cache_data['tasks_labels']
            self.permutation = cache_data['permutation']
            print('Load dataset from cache file {}'.format(cache_path))
        else:
            # List of image spec / question / answer of this dataset.
            # Holds the specification dict for each sample.
            self.image_specs = []
            self.images = []
            self.tasks_labels = []

            # Loop over modes.
            for mode_cntr, mode in enumerate(self.spec['modes']):
                print('Draw mode {}/{}'.format(mode_cntr, len(self.spec['modes'])))
                mode['samples'] = int(mode['ratio'] * self.spec['samples'])
                image_specs, images, tasks_labels = self.draw_mode(mode['spec'], mode['samples'])
                self.image_specs += image_specs
                self.images += images
                self.tasks_labels += tasks_labels

            # Permutation of the whole dataset.
            self.permutation = list(range(len(self.images)))
            np.random.shuffle(self.permutation)

            # Save to cache
            if cache_path is not None:
                cache_data = {
                    'images': self.images,
                    'image_specs': self.image_specs,
                    'tasks_labels': self.tasks_labels,
                    'permutation': self.permutation
                }

                # save cache
                print('Save dataset to cache file {}'.format(cache_path))
                save_obj(cache_data, cache_path)

    def draw_mode(self, mode_spec, number_of_samples):
        """ Draws the entire mode incl. questions and answers.
        """
        image_specs = [None for _ in range(number_of_samples)]
        images = [None for _ in range(number_of_samples)]
        tasks_labels = [{task: None for task in self.tasks} for _ in range(number_of_samples)]

        # Loop over all samples to be added.
        for sample_cntr in tqdm(range(number_of_samples), desc='Drawing...', total=number_of_samples):
            image_spec, semantic_image_spec = self.draw_image_spec_from_mode(copy.deepcopy(mode_spec))

            # Get answers to all questions
            for task in tasks_labels[sample_cntr].keys():
                tasks_labels[sample_cntr][task] = get_answer(semantic_image_spec, task)

            image_specs[sample_cntr] = image_spec
            image = self.painter.paint_images(image_spec, self.spec['shape'])
            images[sample_cntr] = image

#             from PIL import Image
#             import IPython; IPython.embed()
#
#             factor_values_dict = {
#                 'shape': [93, 19, 35, 88, 23], # 22, 56, 22, 4, 80, 93],
#                 'hue': ['red', 'yellow', 'green', 'cyan', 'magenta'],
#                 'lightness': ['brighter', 'dark', 'darker', 'bright'],
#                 'texture': ['tiles', 'wood', 'carpet', 'bricks', 'lava'],
#                 'bg_lightness': ['bright', 'penumbra', 'dark']
#             }
#             base_mode_spec = copy.deepcopy(mode_spec)
#             base_mode_spec['objs'][0]['shape'] = [factor_values_dict['shape'][0]]
#             base_mode_spec['objs'][0]['hue'] = [factor_values_dict['hue'][0]]
#             base_mode_spec['objs'][0]['lightness'] = [factor_values_dict['lightness'][0]]
#             base_mode_spec['objs'][0]['texture'] = [factor_values_dict['texture'][0]]
#             base_mode_spec['bg_lightness'] = [factor_values_dict['bg_lightness'][0]]
#
#             for factor, values in factor_values_dict.items():
#                 for cnt, value in enumerate(values):
#                     sample_mode_spec = copy.deepcopy(base_mode_spec)
#                     if factor != 'bg_lightness':
#                         sample_mode_spec['objs'][0][factor] = [value]
#                     else:
#                         sample_mode_spec[factor] = [value]
#
#                     image_spec = self.draw_image_spec_from_mode(sample_mode_spec)[0]
#                     image_data = self.painter.paint_images(image_spec, self.spec['shape'])[0].transpose(2, 1, 0)
#                     Image.fromarray(image_data).save(f'{factor}_{cnt}.jpg')
#
#             for shape_value in factor_values_dict['shape']:
#                 for hue_value in factor_values_dict['hue']:
#                     sample_mode_spec = copy.deepcopy(base_mode_spec)
#                     sample_mode_spec['objs'][0]['shape'] = [shape_value]
#                     sample_mode_spec['objs'][0]['hue'] = [hue_value]
#
#                     image_spec = self.draw_image_spec_from_mode(sample_mode_spec)[0]
#                     image_data = self.painter.paint_images(image_spec, self.spec['shape'])[0].transpose(2, 1, 0)
#                     Image.fromarray(image_data).save(f'sh_{shape_value}_{hue_value}.jpg')

        return image_specs, images, tasks_labels

    def draw_image_spec_from_mode(self, mode_spec):
        """ Draws a single image specification from a mode.
        """
        # Set empty dictionary for each sample.
        image_spec = dict()

        # Add tag to image spec
        image_spec['tag'] = mode_spec['tag'] if 'tag' in mode_spec.keys() else ''

        # Each attribute in a mode can be given as a list (e.g. 'color': ['red', 'blue', 'green']). In such cases we want
        # to sample an attribute specification randomly from that list. If only a single attribute is given, we use that.
        for attr in (set(mode_spec.keys()) - {'objs', 'questions', 'corruptions', 'tag'}):
            mode_spec[attr] = random_choice(mode_spec[attr])

        # Draw background.
        if mode_spec['bg_style'] == 'color':
            image_spec['bg_style'] = 'color'
            image_spec['bg_color'] = sample_attribute('color', mode_spec['bg_color'],
                                                      light_attr=mode_spec['bg_lightness'])
        elif mode_spec['bg_style'] == 'texture':
            image_spec['bg_style'] = 'texture'
            image_spec['bg_texture'] = mode_spec['bg_texture']
            image_spec['bg_color'] = sample_attribute('colorgrad', mode_spec['bg_color'],
                                                      light_attr=mode_spec['bg_lightness'])
        else:
            image_spec['bg_style'] = mode_spec['bg_style']

        # Loop over objects.
        image_spec['objs'] = []
        for obj_spec in mode_spec['objs']:
            # In case list is given for an attribute, sample an attribute specification randomly from that list
            for attr in obj_spec.keys():
                obj_spec[attr] = random_choice(obj_spec[attr])

            obj = dict()
            # Object category / class.
            obj['category'] = obj_spec['category']
            obj['shape'] = obj_spec['shape']

            # Object scale
            obj['scale'] = sample_attribute('scale', obj_spec['scale'])

            # Draw class instance.
            if obj_spec['draw_instance'] == 'with replacement':
                last_instance_idx = DATASETS[obj['category']]['samples'][obj['shape']]
                obj['instance'] = np.random.randint(0, last_instance_idx)
            elif obj_spec['draw_instance'] == 'without replacement':
                raise ValueError

            # Object style.
            obj['style'] = obj_spec['style']
            if obj['style'] == 'color':
                obj['color'] = sample_attribute('color', obj_spec['hue'], light_attr=obj_spec['lightness'])
                obj_spec['texture'] = 'None'
            elif obj['style'] == 'texture':
                obj['color'] = sample_attribute('colorgrad', obj_spec['hue'], light_attr=obj_spec['lightness'])
                obj['texture'] = obj_spec['texture']
            elif obj['style'] == 'full_texture':
                obj['color'] = None
                obj['texture'] = obj_spec['texture']

            # Object position / orientation / velocity / rotation / scale /scaling.
            for attr in ['position', 'scale']:
                obj[attr] = sample_attribute(attr, obj_spec[attr])

            # Add object to sample.
            image_spec['objs'].append(obj)

        return image_spec, mode_spec

    def getitem(self, idx):
        permuted_idx = self.permutation[idx]
        return {'image': self.images[permuted_idx],
                'targets': self.tasks_labels[permuted_idx],
                'tag': self.image_specs[permuted_idx]['tag']}
