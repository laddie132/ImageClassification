#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = "Han"
__email__ = "liuhan132@foxmail.com"

import re
import os
import json
import math
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


root_path = '/home/lh/weed_photos/'


def load_label_eng_zh(meta_data_path):
    """
    get the english label to chinese label dictionary
    :return:
    """
    with open(meta_data_path, 'r', encoding='utf-8') as f:
        meta_data = json.load(f)

    eng_zh = {}
    for _, ele in meta_data.items():
        for en_dirname, des in ele['classes'].items():
            zh_label = des['name']
            en_label = re.sub(r'[^a-z0-9]+', ' ', en_dirname.lower())

            print(en_label, zh_label)

            eng_zh[en_label] = ele['name'] + '-' + zh_label

    return eng_zh


def rename(meta_data_path):
    """
    rename the path name from chinese to english
    :param meta_data_path:
    :return:
    """
    with open(meta_data_path, 'r') as f:
        meta_data = json.load(f)

    wheat_names = meta_data['wheat']['classes']
    corn_names = meta_data['corn']['classes']

    wheat_path = root_path + 'wheat'
    for en_name in wheat_names:
        src_path = os.path.join(wheat_path, wheat_names[en_name]['name'])
        if os.path.exists(src_path):
            os.rename(src_path,
                      os.path.join(wheat_path, en_name))

    corn_path = root_path + 'corn'
    for en_name in corn_names:
        src_path = os.path.join(corn_path, corn_names[en_name]['name'])
        if os.path.exists(src_path):
            os.rename(src_path,
                      os.path.join(corn_path, en_name))

    print('success')


def fix_amount():
    """
    fix amount error in meta data
    :return:
    """
    with open('../data/weed_meta_data.json', 'r') as f:
        weed_meta_data = json.load(f)

    with open('../data/weed_image_lists.json', 'r') as f:
        weed_image_lists = json.load(f)

    for name, value in weed_image_lists.items():
        amount = len(value['training']) + len(value['testing']) + len(value['validation'])
        dirname = value['dir']
        
        if dirname in weed_meta_data['wheat']['classes']:
            weed_meta_data['wheat']['classes'][dirname]['amount'] = amount
        else:
            weed_meta_data['corn']['classes'][dirname]['amount'] = amount

    with open('../data/weed_meta_data_new.json', 'w') as wf:
        json.dump(weed_meta_data, wf, indent=2, ensure_ascii=False)


def get_amount(meta_data_path, image_lists_path, out_path):
    """
    get the amount of images from the image lists file
    :param meta_data_path:
    :param image_lists_path:
    :param out_path:
    :return:
    """
    label_eng_zh = load_label_eng_zh(meta_data_path)

    amount_json = {}
    with open(image_lists_path, 'r') as f:
        weed_image_lists = json.load(f)

    for name, value in weed_image_lists.items():
        amount = len(value['training']) + len(value['testing']) + len(value['validation'])
        dir_name = value['dir']
        zh_label = label_eng_zh[name]

        amount_json[name] = {'dir': dir_name, 'name': zh_label, 'amount': amount}

    with open(out_path, 'w') as wf:
        json.dump(amount_json, wf, indent=2, ensure_ascii=False)


def analysis_amount(meta_data_path):
    with open(meta_data_path, 'r') as f:
        amount_data = json.load(f)

    all_amount = []
    for ele in amount_data:
        ele_amount = []
        for cs in amount_data[ele]['classes']:
            ele_amount.append(int(amount_data[ele]['classes'][cs]['amount']))

        print('class: %s' % ele)
        print('max amount: ', max(ele_amount))
        print('min amount: ', min(ele_amount))
        print('ave amount: ', sum(ele_amount) / len(ele_amount))
        print('sum amount: ', sum(ele_amount))

        all_amount += ele_amount

    print('all:')
    print('max amount: ', max(all_amount))
    print('min amount: ', min(all_amount))
    print('ave amount: ', sum(all_amount) / len(all_amount))
    print('sum amount: ', sum(all_amount))

    sns.swarmplot(x=all_amount)

    plt.figure()
    sns.boxplot(x=all_amount)


if __name__ == '__main__':
    meta_data_path = '../data/weed_meta_data.json'
    # rename(meta_data_path)
    # fix_amount()
    get_amount(meta_data_path,
               image_lists_path='../data/weed_image_lists_mix_dropped.json',
               out_path='../data/weed_amount_mix_dropped.json')
    # analysis_amount(meta_data_path)
    # plt.show()
