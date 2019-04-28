#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = "Han"
__email__ = "liuhan132@foxmail.com"

import re
import random
import os
import json
import argparse
import tensorflow as tf
import collections

MAX_NUM_IMAGES_PER_CLASS = 2 ** 27 - 1  # ~134M


def create_image_lists(image_dir, testing_percentage, validation_percentage, oversampling_num=-1):
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
        tf.logging.error("Image directory '" + image_dir + "' not found.")
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
        tf.logging.info("Looking for images in '" + dir_name + "'")
        for extension in extensions:
            file_glob = os.path.join(image_dir, dir_name, '[^.]*.' + extension)
            file_list.extend(tf.gfile.Glob(file_glob))
        if not file_list:
            tf.logging.warning('No files found')
            continue
        if len(file_list) < 20:
            tf.logging.warning(
                'WARNING: Folder has less than 20 images, which may cause issues.')
        elif len(file_list) > MAX_NUM_IMAGES_PER_CLASS:
            tf.logging.warning(
                'WARNING: Folder {} has more than {} images. Some images will '
                'never be selected.'.format(dir_name, MAX_NUM_IMAGES_PER_CLASS))
        label_name = re.sub(r'[^a-z0-9]+', ' ', dir_name.lower())
        # training_images = []
        # testing_images = []
        # validation_images = []
        #
        # for file_name in file_list:
        #     base_name = os.path.basename(file_name)
        #     # We want to ignore anything after '_nohash_' in the file name when
        #     # deciding which set to put an image in, the data set creator has a way of
        #     # grouping photos that are close variations of each other. For example
        #     # this is used in the plant disease data set to group multiple pictures of
        #     # the same leaf.
        #     hash_name = re.sub(r'_nohash_.*$', '', file_name)
        #     # This looks a bit magical, but we need to decide whether this file should
        #     # go into the training, testing, or validation sets, and we want to keep
        #     # existing files in the same set even if more files are subsequently
        #     # added.
        #     # To do that, we need a stable way of deciding based on just the file name
        #     # itself, so we do a hash of that and then use that to generate a
        #     # probability value that we use to assign it.
        #     hash_name_hashed = hashlib.sha1(tf.compat.as_bytes(hash_name)).hexdigest()
        #     percentage_hash = ((int(hash_name_hashed, 16) %
        #                         (MAX_NUM_IMAGES_PER_CLASS + 1)) *
        #                        (100.0 / MAX_NUM_IMAGES_PER_CLASS))
        #     if percentage_hash < validation_percentage:
        #         validation_images.append(base_name)
        #     elif percentage_hash < (testing_percentage + validation_percentage):
        #         testing_images.append(base_name)
        #     else:
        #         training_images.append(base_name)

        # random split dataset
        base_name_list = list(map(lambda x: os.path.basename(x),
                                  file_list))
        random.shuffle(base_name_list)
        test_end = len(base_name_list) * testing_percentage // 100
        valid_end = test_end + len(base_name_list) * validation_percentage // 100

        testing_images = base_name_list[:test_end]
        validation_images = base_name_list[test_end:valid_end]
        training_images = base_name_list[valid_end:]

        # oversampling
        if oversampling_num > 0:
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

        result[label_name] = {
            'dir': dir_name,
            'training': training_images,
            'testing': testing_images,
            'validation': validation_images,
        }
    return result


def save_image_lists(image_lists, out_file):
    with open(out_file, 'w') as wf:
        json.dump(image_lists, wf, indent=2)


# todo: load from base_image_list
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--seed',
        type=int,
        default=123,
        help='Set random seed.'
    )
    parser.add_argument(
        '--base_image_list',
        type=str,
        help='Set the base image list'
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
    FLAGS, unparsed = parser.parse_known_args()
    setattr(FLAGS, 'image_lists_dir', FLAGS.output_path + '/image_lists.json')

    # Look at the folder structure, and create lists of all the images.
    image_lists = create_image_lists(FLAGS.image_dir, FLAGS.testing_percentage,
                                     FLAGS.validation_percentage, FLAGS.oversampling_num)
    save_image_lists(image_lists, FLAGS.image_lists_dir)