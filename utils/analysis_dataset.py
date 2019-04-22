#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = "Han"
__email__ = "liuhan132@foxmail.com"


import os
import json
import math
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


root_path = '/home/lh/weed_photos/'


def rename():
    with open('../data/meta_data.json', 'r') as f:
        eng_zh = json.load(f)

    wheat_names = eng_zh['wheat']['classes']
    corn_names = eng_zh['corn']['classes']

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


def ana_amount():
    with open('../data/meta_data.json', 'r') as f:
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
    # rename()
    ana_amount()
    plt.show()