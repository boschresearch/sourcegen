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

from ..dataset.config import *
import numpy as np
import colorsys

__all__ = ['answer_to_index', 'answer_to_vec', 'question_to_vec', 'sample_attribute', 'get_mt_labels']


def get_position(semantic_attr):
    return np.random.uniform(*POSITION[semantic_attr][0]), np.random.uniform(*POSITION[semantic_attr][1])


def get_velocity(semantic_attr):
    return np.random.uniform(*VELOCITY[semantic_attr][0]), np.random.uniform(*VELOCITY[semantic_attr][1])


def get_scale(semantic_attr):
    return np.random.uniform(*SCALE[semantic_attr])


def get_scaling(semantic_attr):
    return np.random.uniform(*SCALING[semantic_attr])


def get_rotation(semantic_attr, deg=False):
    rot = np.random.uniform(*ROTATION[semantic_attr])
    if deg:
        return rot * (180 / np.pi)
    else:
        return rot


def get_orientation(semantic_attr, deg=False):
    rot = np.random.uniform(*ORIENTATION[semantic_attr])
    if rot > 2 * np.pi:
        rot -= 2 * np.pi
    if deg:
        return rot * (180 / np.pi)
    else:
        return rot


def get_color(hue_attr, light_attr):

    if hue_attr == 'gray':
        if light_attr == 'bright':
            col = (0., 1., 0.)
        elif light_attr == 'dark':
            col = (0., 0., 0.)
        else:
            col = (0., 0.5, 0.)

        col = colorsys.hls_to_rgb(*col)
        return tuple((int(x * 255.) for x in col))

    light = np.random.uniform(*COL_LIGHTNESS[light_attr])
    hue = np.random.uniform(*COLORS[hue_attr])
    if hue < 1.:
        hue += 360.
    col = (hue / 360., light, SATURATION)
    col = colorsys.hls_to_rgb(*col)

    return tuple((int(x * 255.) for x in col))


def get_colorgrad(hue_attr, light_attr):
    l1 = np.random.uniform(*TEXT_LIGHTNESS[light_attr][0])
    l2 = np.random.uniform(*TEXT_LIGHTNESS[light_attr][1])

    if hue_attr == 'gray':
        col1, col2 = (0., l1, 0.), (0., l2, 0.)
        col1, col2 = colorsys.hls_to_rgb(*col1), colorsys.hls_to_rgb(*col2)
        return tuple((int(x*255.) for x in col1)), tuple((int(x*255.) for x in col2))

    hue = np.random.uniform(*COLORS[hue_attr])
    if hue < 1.:
        hue += 360.
    col1, col2 = (hue / 360., l1, SATURATION), (hue / 360., l2, SATURATION)
    col1, col2 = colorsys.hls_to_rgb(*col1), colorsys.hls_to_rgb(*col2)

    return tuple((int(x * 255.) for x in col1)), tuple((int(x * 255.) for x in col2))


def sample_attribute(name, semantic_attr, **kwargs):
    get_fn = globals()['get_' + name]
    return get_fn(semantic_attr, **kwargs)

"""
import matplotlib.pyplot as plt

plt.figure(figsize=(10,5))
idx = 1
for light in list(COL_LIGHTNESS):
    for col in list(COLORS):
            rgbs = [[get_color(col, light) for i in range(5)] for j in range(5)]
            ax = plt.subplot(len(list(COL_LIGHTNESS)), len(list(COLORS)), idx)
            ax.imshow([[rgbs[i][j] for i in range(5)] for j in range(5)])
            ax.set_title('{} {}'.format(light, col))
            plt.axis('off')
            idx += 1
plt.tight_layout()"""


def answer_to_index(answer, segmentation_shape):
    """
    Translate an answer tuple (obj_id, attribute) into an answer index and a segmentation map. If the task is
    not a segmentation task (the attribute in the answer tuple is not a ndarray), then we return a zero array
    np.zeros(segmentation_shape) of the defined segmentation_shape as segmentation map.

    Parameters
    ----------
    answer : tuple
        Answer tuple, where the first item is the obj_id (not used here) and the second item is the answer.
    segmentation_shape: tuple
        (width, height) tuple of the width and height of the segmentation map.

    Returns
    -------
    uint8
        Answer index. If the task is a segmentation task we return len(ATTRIBUTES_VECTOR)+1 = len(QUESTION['of])+1 as
        the index that indicates a segmentation task.
    uint8 ndarray
        Segmentation map. If the task is not a segmentation task, this is zero everywhere.
    """
    if isinstance(answer[1], np.ndarray):
        answer_index = np.array(len(QUESTION['of']), dtype='int64')
        return answer_index, answer[1]
    else:
        one_hot_answer = np.array([answer[1] == attribute for attribute in QUESTION['of']], dtype='uint8')
        answer_index = np.array(np.argmax(one_hot_answer), dtype='int64')
        return answer_index, np.zeros(segmentation_shape, dtype='uint8')


def answer_to_vec(answer, segmentation_shape):
    """
    Translate an answer tuple (obj_id, attribute) into a one-hot answer vector and a segmentation map. If the task is
    not a segmentation task (the attribute in the answer tuple is not a ndarray), then we return a zero array
    np.zeros(segmentation_shape) of the defined segmentation_shape as segmentation map.

    Parameters
    ----------
    answer : tuple
        Answer tuple, where the first item is the obj_id (not used here) and the second item is the answer.
    segmentation_shape: tuple
        (width, height) tuple of the width and height of the segmentation map.

    Returns
    -------
    uint8 ndarray
        One-hot answer vector of the same length as the ATTRIBUTES vector.
    uint8 ndarray
        Segmentation map. If the task is not a segmentation task, this is zero everywhere.
    """
    if isinstance(answer[1], np.ndarray):
        one_hot_answer = np.zeros(len(QUESTION['of']), dtype='uint8')
        return one_hot_answer, answer[1]
    else:
        one_hot_answer = np.array([answer[1] == attribute for attribute in QUESTION['of']], dtype='uint8')
        return one_hot_answer, np.zeros(segmentation_shape, dtype='uint8')


def question_to_vec(question):
    """
    Translates a question dictionary {'what': ..., 'of': ..., 'when': ...} into a question vector. This vector
    is the concatenation of the 'what', 'of' and 'when' parts of the question.

    Parameters
    ----------
    question : dict
        Question dictionary {'what': ..., 'of': ..., 'when': ...}.

    Returns
    -------
    int8 ndarray
        Question tensor which is a concatenation of the single-hot task definition, the multi-hot attribute
        specification and the timestep relevant for segmentation tasks.
    """
    # One-hot task definition
    what_vec = [question['what'] == QUESTION['what'][i] for i in range(len(QUESTION['what']))]

    # Get the attributes of the question
    question_attributes = [attribute[1] for attribute in question['of']]

    # Multi-hot attributes specification
    of_vec = [True if attribute in question_attributes else False for attribute in QUESTION['of']]

    # Timestep
    when = [question['when']]

    return np.concatenate([np.array(vec, dtype='int8') for vec in [what_vec, of_vec, when]])


def vec_to_question(question_vec):
    """
    Translates a multi-hot question vector into a question dictionary {'what': ..., 'of': ..., 'when': ...}.

    Parameters
    ----------
    question_vec: int8 ndarray
        Question tensor which is a concatenation of the single-hot task definition, the multi-hot attribute
        specification and the timestep relevant for segmentation tasks.


    Returns
    -------
    dict
        Question dictionary {'what': ..., 'of': ..., 'when': ...}.
    """
    n_specs = {part: len(QUESTION[part]) for part in ['what', 'of']}

    what = QUESTION['what'][np.where(question_vec[:n_specs['what']] == 1.)[0][0]]

    of_idx = np.where(question_vec[n_specs['what']:n_specs['what'] + n_specs['of']] == 1.)[0]
    of = [QUESTION['of'][i] for i in of_idx]
    when = question_vec[-1]

    return {'what': what, 'of': of, 'when': when}


def remove_from_str(str, to_remove):
    return str.replace(to_remove, '', 1).replace(' ', '')


def question_to_phrase(question):
    question_phrase = ''
    # what part of the question
    if question['what'] == 'segment':
        question_phrase += 'Segment the'
    else:
        question_phrase += 'What is the ' + question['what'] + ' of the'


    # of part of the question
    of_structure = []
    for attr in question['of']:
        attr_category = [k for k, specs in OBJECT_ATTRIBUTES.items() if attr in specs][0]
        if attr_category == 'lightness':
            of_structure.append((0, ' {} '.format(attr)))
        if attr_category == 'color':
            of_structure.append((1, ' {} '.format(attr)))
        if attr_category == 'category':
            of_structure.append((2, ' {} '.format(attr)))
        if attr_category == 'class':
            of_structure.append((3, ' class {} '.format(attr)))
        if attr_category == 'texture':
            of_structure.append((5, ' a {}  texture '.format(attr)))
        if attr_category == 'orientation':
            of_structure.append((6, ' {} orientation'.format(attr)))
        if attr_category == 'rotation':
            if attr == 'no rotation':
                of_structure.append((7, ' not rotating '))
            else:
                of_structure.append((7, ' rotating {} '.format(attr)))
        if attr_category == 'position':
            if attr == 'center center':
                of_structure.append((8, ' center position'))
            else:
                of_structure.append((8, ' {} position'.format(attr)))
        if attr_category == 'velocity':
            if attr == 'none none':
                of_structure.append((9, ' not moving '))
            else:
                of_structure.append((9, ' moving {}'.format(remove_from_str(attr, 'none'))))
        if attr_category == 'scale':
            of_structure.append((10, ' {} scale'.format(attr)))
        if attr_category == 'scaling':
            if attr == 'no scaling':
                of_structure.append((11, ' not scaling '))
            else:
                of_structure.append((11, ' getting {}'.format(attr)))
    of_positions = [attr[0] for attr in of_structure]
    if 3 not in of_positions:
        of_structure.append((3, ' object '))
    if np.any([pos > 4 for pos in of_positions]):
        of_structure.append((4, ' with '))

    question_phrase += ''.join([attr[1] for attr in sorted(of_structure)])
    question_phrase += ' at timestep {:.0f}'.format(question['when'])
    return question_phrase.replace('  ', ' ')


def index_to_answer(answer_idx):
    """
    Translate an answer tuple (answer_idx, segmentation_map) into a semantic answer.
    """
    if answer_idx == len(QUESTION['of']):
        return 'segmentation'

    return QUESTION['of'][answer_idx]



def get_mt_labels(question_answer):
    def get_question_labels(quest):
        if quest == 'bg_lightness':
            return ['dark', 'penumbra', 'bright']
        return OBJECT_ATTRIBUTES[quest]
    return [np.argmax([cls == answ for cls in get_question_labels(quest)]) for quest, answ in question_answer.items()]
