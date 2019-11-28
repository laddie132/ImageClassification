#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = "Han"
__email__ = "liuhan132@foxmail.com"

import re
import json
from utils.dataset_fun import load_label_eng_zh


def transform():
    meta_data_path = '../data/weed_meta_data.json'
    eng_zh = load_label_eng_zh(meta_data_path)

    labels = []
    with open('../outputs/weed-v3-1-inaturalist-inception/output_labels.txt', 'r') as f:
        for line in f.readlines():
            labels.append(line.strip())

    out_json = []

    for l in labels:
        out_json.append({'cname': eng_zh[l], 'description': ''})

    with open('../data/v3-1/weed_app_labels.json', 'w', encoding='utf-8') as wf:
        json.dump(out_json, wf, indent=2, ensure_ascii=False)


if __name__ == '__main__':
    transform()
