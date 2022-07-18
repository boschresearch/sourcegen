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

import numpy as np
from imageio import imread
from scipy.ndimage import rotate, binary_erosion, binary_opening, binary_dilation
from .config import DATASETS, TEXTURES, load_dataset
from skimage.transform import rescale, resize
THRESHOLD = 150
DATASETS = load_dataset(DATASETS)

__all__ = ['Painter']


def extract_boundary(mask, radius):
    mask = mask // 255

    if radius == 1:
        mask_eroded = binary_erosion(mask, structure=np.ones((3, 3))).astype('uint8')
        return (mask - mask_eroded) * 255
    else:
        mask_eroded = binary_erosion(mask, structure=np.ones((radius + 1, radius + 1))).astype('uint8')
        mask_dilated = binary_dilation(mask, structure=np.ones((radius + 1, radius + 1))).astype('uint8')

        return (mask_dilated - mask_eroded) * 255


def random_crop(image, size=(128, 128), axes=(-2, -1)):
    """
    Perform random crop of a given image.

    Parameters
    ----------
    image : ndarray
        Image array from which to crop
    size : tuple
        Tuple specifying the crop size along the two dimensions. Default=(128, 128)
    axes : tuple
        Axes that define the dimension in which to crop. Default=(-2, -1), last two axes.

    Returns
    -------
    ndarray
        Image crop of the given size
    """

    x = np.random.randint(image.shape[axes[0]] - size[0])
    y = np.random.randint(image.shape[axes[1]] - size[1])
    slc = [slice(None)] * image.ndim
    slc[axes[0]] = slice(x, x + size[0])
    slc[axes[1]] = slice(y, y + size[1])
    return image[tuple(slc)]


def enhance_contrast(img, low_percentile=2, high_percentile=98):
    minval = np.percentile(img, low_percentile)
    maxval = np.percentile(img, high_percentile)
    img = np.clip(img, minval, maxval)
    return (((img - minval) / (maxval - minval)) * 255).astype('uint8')


def create_canvas(shape, spec):
    if spec['bg_style'] == 'color':
        img = np.stack([np.full(shape, spec['bg_color'][c], dtype='uint16') for c in range(3)], axis=1)
    elif spec['bg_style'] == 'texture':
        texture = random_crop(imread(TEXTURES[spec['bg_texture']]), size=shape[1:]).astype('uint16')
        col = spec['bg_color']
        img = np.stack([(col[0][c] * texture + col[1][c] * (255 - texture)) // 255 for c in range(3)], axis=0)
        img = np.stack([img] * shape[0], axis=0)

    elif spec['bg_style'] == 'noise_static':
        noise = (np.random.random(size=(3, *shape[1:])) * 255.).astype('uint16')
        img = np.stack([noise] * shape[0], axis=0)
    elif spec['bg_style'] == 'noise_dynamic':
        img = (np.random.random(size=(shape[0], 3, *shape[1:])) * 255.).astype('uint16')
    else:
        raise ValueError('Provided background style {} not supported'.format(spec['bg_style']))
    return img


def create_object(obj_spec):
    obj_dataset = DATASETS[obj_spec['category']]
    obj_plain = obj_dataset['X'][sum(obj_dataset['samples'][:obj_spec['shape']]) + obj_spec['instance']]
    obj_plain = obj_plain.astype('uint16')

    # Get alpha mask to blend with background
    if 'fashion-mnist' in obj_spec['category']:
        alpha = (obj_plain.copy() > 0).astype('uint8')
        alpha = binary_opening(alpha, structure=np.ones((3, 3)), iterations=2).astype('uint8')
        alpha = binary_erosion(alpha, structure=np.ones((3, 3)), iterations=2).astype('uint8')
        alpha = np.stack([255*alpha] * 3, axis=0).astype('uint8')
    else:
        alpha = np.where(obj_plain.copy() > THRESHOLD, 255, 0).astype('uint8')
        alpha = np.stack([alpha] * 3, axis=0)

    # Stylize object
    if obj_spec['style'] == 'color':
        obj = np.stack([np.full_like(obj_plain, obj_spec['color'][c], dtype='uint16') for c in range(3)], axis=0)
    elif obj_spec['style'] == 'texture':
        col = obj_spec['color']
        texture = random_crop(imread(TEXTURES[obj_spec['texture']]), size=obj_plain.shape).astype('uint16')
        obj = np.stack([(col[0][c] * texture + col[1][c] * (255 - texture)) // 255 for c in range(3)], axis=0)
    elif obj_spec['style'] == 'noise_static':
        obj = (np.random.random(size=(3, *obj_plain.shape)) * 255.).astype('uint16')
    elif obj_spec['style'] == 'noise_dynamic':
        obj = np.stack([obj_plain]*3, axis=0)
    elif obj_spec['style'] == 'boundary':
        obj = np.full((3, *obj_plain.shape), 255, dtype='uint16')
        seg = alpha.copy()
        alpha = np.stack([extract_boundary(alpha[0], 2)] * 3, axis=0)
        return obj, alpha, seg
    else:
        raise ValueError('Provided object style {} not supported'.format(obj_spec['style']))

    return obj, alpha, alpha


def rel_to_abs_pos(rel_pos, img_size, obj_size):
    return int((img_size - 1) * rel_pos - (obj_size - 1) / 2)


def pos_vel_to_abs(rel_pos, rel_vel, img_shape, obj_size):
    abs_pos = (rel_to_abs_pos(rel_pos[0], img_shape[1], obj_size[1]),
               rel_to_abs_pos(rel_pos[1], img_shape[2], obj_size[2]))
    abs_vel = (int(rel_vel[0] * img_shape[1]), int(rel_vel[1] * img_shape[2]))
    return abs_pos, abs_vel


def paint_images(spec, shape):
    """
    Paint an image from a given sample specification.

    Parameters
    ----------
    spec : dict
        Sample specification dictionary containing background and object attributes.
    question: dict
        Dictionary containing the question. Is used to determine whether the task is to segment.
    answer: tuple
        Tuple where the first entry is the object_id about which the question was and second tuple is the answer.
        If the task is to segment the answer entry (None by default) is replaced by the segmentation mask. Otherwise
        the answer is returned without modification.
    shape : tuple
        Desired sample size (TimeFrames, XSize, YSize). Default=(5, 128, 128).
    rescale_factor : float, int
        Rescale factor to use for each of the objects. Default size of each object is 28x28.

    Returns
    -------
    ndarray
        Painted sample of shape TimeFrames x 3 x XSize x YSize.
    """
    img_frames = shape[0]

    # Create canvas
    img = create_canvas(shape, spec)

    # Create objects
    for obj_idx, obj_spec in enumerate(spec['objs']):
        obj, alpha, seg = create_object(obj_spec)

        # Here, we fix scaling, orientation, rotation and velocity
        obj_spec['scaling'] = 1.
        obj_spec['rotation'] = 0.
        obj_spec['velocity'] = (0., 0.)
        obj_spec['orientation'] = 0.

        # Place object on canvas
        for frame in range(shape[0]):
            orientation = np.rad2deg(obj_spec['orientation'] + frame * obj_spec['rotation'])
            scale = obj_spec['scale'] + frame * obj_spec['scaling']

            frame_obj = rescale(rotate(obj.copy(), orientation, axes=(1, 2), reshape=False).transpose(1, 2, 0),
                                scale, anti_aliasing=True, preserve_range=True, multichannel=True).transpose(2, 0, 1).astype('uint16')
            frame_alpha = rescale(rotate(alpha.copy(), orientation, axes=(1, 2), reshape=False).transpose(1, 2, 0),
                                  scale, anti_aliasing=True, preserve_range=True, multichannel=True).transpose(2, 0, 1).astype('uint16')

            pos, vel = pos_vel_to_abs(obj_spec['position'], obj_spec['velocity'], shape, frame_obj.shape)

            if obj_spec['style'] == 'noise_dynamic':
                frame_obj = (np.random.random(size=frame_obj.shape) * 255.).astype('uint16')

            min_x, min_y = pos[0] + frame * vel[0], pos[1] + frame * vel[1]
            obj_corners = [min_x, min_x + frame_obj.shape[1], min_y, min_y + frame_obj.shape[2]]
            # Account for cases where the object is (partially) outside the image
            obj_img_corners = [np.max([0, obj_corners[0]]), np.min([img.shape[2], obj_corners[1]]),
                               np.max([0, obj_corners[2]]), np.min([img.shape[3], obj_corners[3]])]
            diff = [np.clip(np.abs(obj_img_corners[i] - obj_corners[i]), a_min=None, a_max=frame_obj.shape[axis]) \
                    for i, axis in zip(range(4), [1]*2 + [2]*2)]

            img_crop = img[frame, :, obj_img_corners[0]:obj_img_corners[1], obj_img_corners[2]:obj_img_corners[3]]

            # If frame_obj is partially outside img, then crop it accordingly. Do the same for alpha and seg
            frame_obj = frame_obj[:, diff[0]:frame_obj.shape[1] - diff[1], diff[2]:frame_obj.shape[2] - diff[3]]
            frame_alpha = frame_alpha[:, diff[0]:frame_alpha.shape[1] - diff[1], diff[2]:frame_alpha.shape[2] - diff[3]]

            frame_obj = (frame_alpha * frame_obj + (255 - frame_alpha) * img_crop) // 255
            img[frame, :, obj_img_corners[0]:obj_img_corners[1], obj_img_corners[2]:obj_img_corners[3]] = frame_obj

    return img[:img_frames].astype('uint8')


class Painter(object):
    """
    Just a wrapper class for the functions above. This avoids the loading of textures for every image that is
    painted
    """
    def __init__(self):
        # preprocess texture
        texture_full_size = (64, 64)
        self.textures = {}
        for texture, path in TEXTURES.items():
            texture_img = imread(path, as_gray=False, pilmode="RGB")
            texture_img = resize(texture_img, texture_full_size, anti_aliasing=True, preserve_range=True)
            self.textures[texture] = texture_img

    def create_canvas(self, shape, spec):
        if spec['bg_style'] == 'color':
            img = np.stack([np.full(shape, spec['bg_color'][c], dtype='uint16') for c in range(3)], axis=1)
        elif spec['bg_style'] == 'texture':
            texture_raw = self.textures[obj_spec['texture']].mean(axis=2).astype('uint16')

            col = spec['bg_color']
            texture = random_crop(texture_raw, size=shape[1:]).astype('uint16')
            img = np.stack([(col[0][c] * texture + col[1][c] * (255 - texture)) // 255 for c in range(3)], axis=0)
            img = np.stack([img] * shape[0], axis=0)

        elif spec['bg_style'] == 'noise_static':
            noise = (np.random.random(size=(3, *shape[1:])) * 255.).astype('uint16')
            img = np.stack([noise] * shape[0], axis=0)
        elif spec['bg_style'] == 'noise_dynamic':
            img = (np.random.random(size=(shape[0], 3, *shape[1:])) * 255.).astype('uint16')
        else:
            raise ValueError('Provided background style {} not supported'.format(spec['bg_style']))
        return img

    def create_object(self, obj_spec):
        obj_dataset = DATASETS[obj_spec['category']]
        obj_plain = obj_dataset['X'][sum(obj_dataset['samples'][:obj_spec['shape']]) + obj_spec['instance']]
        obj_plain = obj_plain.astype('uint16')

        # Get alpha mask to blend with background
        if 'fashion-mnist' in obj_spec['category']:
            alpha = (obj_plain.copy() > 0).astype('uint8')
            alpha = binary_opening(alpha, structure=np.ones((3, 3)), iterations=2).astype('uint8')
            alpha = binary_erosion(alpha, structure=np.ones((3, 3)), iterations=2).astype('uint8')
            alpha = np.stack([255 * alpha] * 3, axis=0).astype('uint8')
        else:
            alpha = np.where(obj_plain.copy() > THRESHOLD, 255, 0).astype('uint8')
            alpha = np.stack([alpha] * 3, axis=0)

        # Stylize object
        if obj_spec['style'] == 'color':
            obj = np.stack([np.full_like(obj_plain, obj_spec['color'][c], dtype='uint16') for c in range(3)],
                           axis=0)
        elif obj_spec['style'] == 'texture':
            texture_raw = self.textures[obj_spec['texture']].mean(axis=2).astype('uint16')

            col = obj_spec['color']
            texture = random_crop(texture_raw, size=obj_plain.shape).astype('uint16')
            obj = np.stack([(col[0][c] * texture + col[1][c] * (255 - texture)) // 255 for c in range(3)], axis=0)
        elif obj_spec['style'] == 'full_texture':
            texture_raw = self.textures[obj_spec['texture']].astype('uint16')

            texture = random_crop(texture_raw.transpose(2, 0, 1), size=obj_plain.shape).astype('uint16')
            obj = texture
        elif obj_spec['style'] == 'noise_static':
            obj = (np.random.random(size=(3, *obj_plain.shape)) * 255.).astype('uint16')
        elif obj_spec['style'] == 'noise_dynamic':
            obj = np.stack([obj_plain] * 3, axis=0)
        elif obj_spec['style'] == 'boundary':
            obj = np.full((3, *obj_plain.shape), 255, dtype='uint16')
            seg = alpha.copy()
            alpha = np.stack([extract_boundary(alpha[0], 2)] * 3, axis=0)
            return obj, alpha, seg
        else:
            raise ValueError('Provided object style {} not supported'.format(obj_spec['style']))

        return obj, alpha, alpha

    def paint_images(self, spec, shape):
        """
        Paint an image from a given sample specification.

        Parameters
        ----------
        spec : dict
            Sample specification dictionary containing background and object attributes.
        question: dict
            Dictionary containing the question. Is used to determine whether the task is to segment.
        answer: tuple
            Tuple where the first entry is the object_id about which the question was and second tuple is the answer.
            If the task is to segment the answer entry (None by default) is replaced by the segmentation mask. Otherwise
            the answer is returned without modification.
        shape : tuple
            Desired sample size (TimeFrames, XSize, YSize). Default=(5, 128, 128).
        rescale_factor : float, int
            Rescale factor to use for each of the objects. Default size of each object is 28x28.

        Returns
        -------
        ndarray
            Painted sample of shape TimeFrames x 3 x XSize x YSize.
        """
        img_frames = shape[0]

        # Create canvas
        img = self.create_canvas(shape, spec)

        # Create objects
        for obj_idx, obj_spec in enumerate(spec['objs']):
            obj, alpha, seg = self.create_object(obj_spec)

            # Here, we fix scaling, orientation, rotation and velocity
            obj_spec['scaling'] = 1.
            obj_spec['rotation'] = 0.
            obj_spec['velocity'] = (0., 0.)
            obj_spec['orientation'] = 0.

            # Place object on canvas
            for frame in range(shape[0]):
                orientation = np.rad2deg(obj_spec['orientation'] + frame * obj_spec['rotation'])
                scale = obj_spec['scale'] + frame * obj_spec['scaling']

                frame_obj = rescale(rotate(obj.copy(), orientation, axes=(1, 2), reshape=False).transpose(1, 2, 0),
                                    scale, anti_aliasing=True, preserve_range=True, multichannel=True).transpose(2,
                                                                                                                 0,
                                                                                                                 1).astype(
                    'uint16')
                frame_alpha = rescale(
                    rotate(alpha.copy(), orientation, axes=(1, 2), reshape=False).transpose(1, 2, 0),
                    scale, anti_aliasing=True, preserve_range=True, multichannel=True).transpose(2, 0, 1).astype(
                    'uint16')

                pos, vel = pos_vel_to_abs(obj_spec['position'], obj_spec['velocity'], shape, frame_obj.shape)

                if obj_spec['style'] == 'noise_dynamic':
                    frame_obj = (np.random.random(size=frame_obj.shape) * 255.).astype('uint16')

                min_x, min_y = pos[0] + frame * vel[0], pos[1] + frame * vel[1]
                obj_corners = [min_x, min_x + frame_obj.shape[1], min_y, min_y + frame_obj.shape[2]]
                # Account for cases where the object is (partially) outside the image
                obj_img_corners = [np.max([0, obj_corners[0]]), np.min([img.shape[2], obj_corners[1]]),
                                   np.max([0, obj_corners[2]]), np.min([img.shape[3], obj_corners[3]])]
                diff = [
                    np.clip(np.abs(obj_img_corners[i] - obj_corners[i]), a_min=None, a_max=frame_obj.shape[axis]) \
                    for i, axis in zip(range(4), [1] * 2 + [2] * 2)]

                img_crop = img[frame, :, obj_img_corners[0]:obj_img_corners[1],
                           obj_img_corners[2]:obj_img_corners[3]]

                # If frame_obj is partially outside img, then crop it accordingly. Do the same for alpha and seg
                frame_obj = frame_obj[:, diff[0]:frame_obj.shape[1] - diff[1], diff[2]:frame_obj.shape[2] - diff[3]]
                frame_alpha = frame_alpha[:, diff[0]:frame_alpha.shape[1] - diff[1],
                              diff[2]:frame_alpha.shape[2] - diff[3]]

                frame_obj = (frame_alpha * frame_obj + (255 - frame_alpha) * img_crop) // 255
                img[frame, :, obj_img_corners[0]:obj_img_corners[1],
                obj_img_corners[2]:obj_img_corners[3]] = frame_obj

        return img[:img_frames].astype('uint8')
