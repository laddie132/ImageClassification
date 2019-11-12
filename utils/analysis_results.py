#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = "Han"
__email__ = "liuhan132@foxmail.com"

import json
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def plot_acc_dataset(result_path):
    with open(result_path, 'r') as f:
        results = json.load(f)

    acc = list(map(lambda x: x[1]['acc'], results.items()))
    data_num = list(map(lambda x: x[1]['all_number'], results.items()))

    plt.scatter(data_num, acc)


def filter_dataset(result_path):
    with open(result_path, 'r') as f:
        results = json.load(f)

    for k, v in results.items():
        if v['all_number'] < 100:
            print(k, v['all_number'], v['acc'])


def map_eng_zh(meta_data_path):
    with open(meta_data_path, 'r', encoding='utf-8') as f:
        meta_data = json.load(f)

    map_names = {}
    for k, v in meta_data.items():
        map_names[k] = v['name']

    return map_names


def create_results_table(result_path, meta_data_path, output_path):
    with open(result_path, 'r') as f:
        results = json.load(f)

    map_names = map_eng_zh(meta_data_path)
    labels = list(results.keys())

    result_names = []
    result_accs = []
    result_accs_top3 = []
    result_all_nums = []
    result_confuse_1 = []
    result_confuse_2 = []
    result_confuse_3 = []
    for _, v in results.items():
        result_names.append(map_names[v['dir']])
        result_accs.append(round(v['acc'], 2))
        result_accs_top3.append(round(v['acc_top3'], 2))
        result_all_nums.append(v['all_number'])

        confuse_vector = [0 for _ in range(len(results))]
        for wk, wv in v['wrong_images'].items():
            for lk, lv in wv.items():
                confuse_vector[labels.index(lk)] += eval(lv)

        top_confuse = np.argsort(-np.array(confuse_vector))[:3]
        top_confuse_name = list(map(lambda x: labels[x], top_confuse))
        top_confuse_name_zh = list(map(
            lambda x: map_names[results[x]['dir']],
            top_confuse_name
        ))

        if confuse_vector[top_confuse[0]] > 0:
            result_confuse_1.append(top_confuse_name_zh[0])
        else:
            result_confuse_1.append('None')
        if confuse_vector[top_confuse[1]] > 0:
            result_confuse_2.append(top_confuse_name_zh[1])
        else:
            result_confuse_2.append('None')
        if confuse_vector[top_confuse[2]] > 0:
            result_confuse_3.append(top_confuse_name_zh[2])
        else:
            result_confuse_3.append('None')

    df = pd.DataFrame({'Name': result_names, 'Amount': result_all_nums, 'Acc': result_accs,
                       'Acc(Top-3)': result_accs_top3, 'Top-1 Confuse': result_confuse_1,
                       'Top-2 Confuse': result_confuse_2, 'Top-3 Confuse': result_confuse_3})
    df.to_excel(output_path, index=False)


if __name__ == '__main__':
    meta_data_path = '../data/weed_meta_data.json'
    result_path = '../outputs/weed-v2-inaturalist-inception-inception_resnet/test_results.json'
    output_path = '../outputs/weed-v2-inaturalist-inception-inception_resnet/results_table.xls'
    # plot_acc_dataset(result_path)
    # plt.show()

    create_results_table(result_path, meta_data_path, output_path)
