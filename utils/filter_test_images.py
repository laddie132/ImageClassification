#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = "Han"
__email__ = "liuhan132@foxmail.com"

"""Filter the images for manual testing"""

import os
import json
from shutil import copy
from utils.dataset_fun import load_label_eng_zh


def main():
    out_path = '/home/lh/weed_photos_resize_v2_test/'
    src_path = '/home/lh/weed_photos_resize_v2/'
    image_lists_path = '../data/weed_image_lists_oversample.json'
    meta_data_path = '../data/weed_meta_data.json'

    label_eng_zh = load_label_eng_zh(meta_data_path)

    with open(image_lists_path, 'r') as f:
        image_lists = json.load(f)

    for name, value in image_lists.items():
        this_src_dir = src_path + value['dir']
        this_out_dir = out_path + label_eng_zh[name]
        os.mkdir(this_out_dir)

        print('copying ' + name)

        for image_name in value['testing']:
            image_src_path = os.path.join(this_src_dir, image_name)
            image_out_path = os.path.join(this_out_dir, image_name)

            copy(image_src_path, image_out_path)


if __name__ == '__main__':
    main()
