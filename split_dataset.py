#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = "Han"
__email__ = "liuhan132@foxmail.com"

import re
import random
import os
import json
import argparse
import numpy as np
import tensorflow as tf
import collections
import copy

MAX_NUM_IMAGES_PER_CLASS = 2 ** 27 - 1  # ~134M


def create_image_lists(image_dir, testing_percentage, validation_percentage):
    """Builds a list of training images from the file system.

    Analyzes the sub folders in the image directory, splits them into stable
    training, testing, and validation sets, and returns a data structure
    describing the lists of images for each label and their paths.

    Args:
      image_dir: String path to a folder containing subfolders of images.
      testing_percentage: Integer percentage of the images to reserve for tests.
      validation_percentage: Integer percentage of images reserved for validation.

    Returns:
      An OrderedDict containing an entry for each label subfolder, with images
      split into training, testing, and validation sets within each label.
      The order of items defines the class indices.
    """
    if not tf.gfile.Exists(image_dir):
        print("Image directory '" + image_dir + "' not found.")
        return None
    result = collections.OrderedDict()
    sub_dirs = sorted(x[0] for x in tf.gfile.Walk(image_dir))
    # The root directory comes first, so skip it.
    is_root_dir = True
    for sub_dir in sub_dirs:
        if is_root_dir:
            is_root_dir = False
            continue
        extensions = sorted(set(os.path.normcase(ext)  # Smash case on Windows.
                                for ext in ['JPEG', 'JPG', 'jpeg', 'jpg', 'png']))
        file_list = []
        dir_name = os.path.basename(
            # tf.gfile.Walk() returns sub-directory with trailing '/' when it is in
            # Google Cloud Storage, which confuses os.path.basename().
            sub_dir[:-1] if sub_dir.endswith('/') else sub_dir)

        if dir_name == image_dir:
            continue
        print("Looking for images in '" + dir_name + "'")
        for extension in extensions:
            file_glob = os.path.join(image_dir, dir_name, '[^.]*.' + extension)
            file_list.extend(tf.gfile.Glob(file_glob))
        if not file_list:
            print('No files found')
            continue
        if len(file_list) < 20:
            print(
                'WARNING: Folder has less than 20 images, which may cause issues.')
        elif len(file_list) > MAX_NUM_IMAGES_PER_CLASS:
            print(
                'WARNING: Folder {} has more than {} images. Some images will '
                'never be selected.'.format(dir_name, MAX_NUM_IMAGES_PER_CLASS))
        label_name = re.sub(r'[^a-z0-9]+', ' ', dir_name.lower())

        # random split dataset
        base_name_list = list(map(lambda x: os.path.basename(x),
                                  file_list))
        random.shuffle(base_name_list)
        test_end = len(base_name_list) * testing_percentage // 100
        valid_end = test_end + len(base_name_list) * validation_percentage // 100

        testing_images = base_name_list[:test_end]
        validation_images = base_name_list[test_end:valid_end]
        training_images = base_name_list[valid_end:]

        result[label_name] = {
            'dir': dir_name,
            'training': training_images,
            'testing': testing_images,
            'validation': validation_images,
        }
    return result


def oversample(image_lists, oversampling_num):
    if oversampling_num < 0:
        return image_lists

    oversample_image_lists = collections.OrderedDict()
    for name, value in image_lists.items():
        training_images = value['training']

        mulitple = oversampling_num // len(training_images)
        if mulitple == 0:
            random.shuffle(training_images)
            training_images = training_images[:oversampling_num]
        else:
            oversampling_images = []
            for _ in range(mulitple):
                oversampling_images += training_images
            sample_images = random.sample(training_images, oversampling_num - len(oversampling_images))
            training_images = oversampling_images + sample_images

        value['training'] = training_images
        oversample_image_lists[name] = value
    return oversample_image_lists


def concat_image_lists(base_image_lists, append_image_lists):
    new_image_lists = copy.deepcopy(base_image_lists)

    for name, value in append_image_lists.items():
        print('Concating label: ' + name)
        if name in new_image_lists:
            assert new_image_lists[name]['dir'] == value['dir']

            new_image_lists[name]['training'].extend(value['training'])
            random.shuffle(value['training'])

            new_image_lists[name]['testing'].extend(value['testing'])
            random.shuffle(value['testing'])

            new_image_lists[name]['validation'].extend(value['validation'])
            random.shuffle(value['validation'])
        else:
            new_image_lists[name] = value

    return new_image_lists


def filter_image_lists(image_lists, min_num):
    filtered_image_lists = collections.OrderedDict()
    dropped_image_lists = collections.OrderedDict()
    for name, value in image_lists.items():
        print('Filtering label: ' + name)

        num_images = len(value['training']) + len(value['testing']) + len(value['validation'])
        if num_images < min_num:
            dropped_image_lists[name] = value
        else:
            filtered_image_lists[name] = value

    return filtered_image_lists, dropped_image_lists


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.set_random_seed(seed)


def save_image_lists(image_lists, out_file):
    with open(out_file, 'w') as wf:
        json.dump(image_lists, wf, indent=2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--seed',
        type=int,
        default=123,
        help='Set random seed.'
    )
    parser.add_argument(
        '--base_image_lists_dir',
        type=str,
        default='',
        help='Set the base image lists dir'
    )
    parser.add_argument(
        '--out_image_lists_dir',
        type=str,
        default='',
        help='Set the output image lists dir'
    )
    parser.add_argument(
        '--image_dir',
        type=str,
        default='',
        help='Path to folders of labeled images.'
    )
    parser.add_argument(
        '--testing_percentage',
        type=int,
        default=10,
        help='What percentage of images to use as a test set.'
    )
    parser.add_argument(
        '--validation_percentage',
        type=int,
        default=10,
        help='What percentage of images to use as a validation set.'
    )
    parser.add_argument(
        '--oversampling_num',
        type=int,
        default=-1,
        help='Oversamping number'
    )
    parser.add_argument(
        '--min_num',
        type=int,
        default=-1,
        help='Minimal number of label images.'
    )
    parser.add_argument(
        '--dropped_image_lists_dir',
        type=str,
        default='',
        help='Where dropped image lists saved.'
    )
    FLAGS, unparsed = parser.parse_known_args()
    set_random_seed(FLAGS.seed)

    # Look at the folder structure, and create lists of all the images.
    image_lists = create_image_lists(FLAGS.image_dir, FLAGS.testing_percentage,
                                     FLAGS.validation_percentage)

    # concat other image lists
    if FLAGS.base_image_lists_dir != '':
        with open(FLAGS.base_image_lists_dir, 'r') as f:
            base_image_lists = json.load(f)
        image_lists = concat_image_lists(base_image_lists, image_lists)

    # filter with number of images
    if FLAGS.min_num > 0:
        image_lists, dropped_image_lists = filter_image_lists(image_lists, FLAGS.min_num)
        save_image_lists(dropped_image_lists, FLAGS.dropped_image_lists_dir)

    # oversample
    if FLAGS.oversampling_num > 0:
        image_lists = oversample(image_lists, FLAGS.oversampling_num)

    save_image_lists(image_lists, FLAGS.out_image_lists_dir)
