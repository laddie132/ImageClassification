#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = "Han"
__email__ = "liuhan132@foxmail.com"

import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def plot_acc_dataset(result_path):
    with open(result_path, 'r') as f:
        results = json.load(f)

    acc = list(map(lambda x: x[1]['acc'], results.items()))
    data_num = list(map(lambda x: x[1]['all_number'], results.items()))

    plt.scatter(data_num, acc)


if __name__ == '__main__':
    plot_acc_dataset('../outputs/weed-sample/test_results.json')
    plt.show()