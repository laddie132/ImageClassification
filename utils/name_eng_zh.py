#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = "Han"
__email__ = "liuhan132@foxmail.com"


import json


def transform():
    with open('../data/weed_meta_data.json', 'r', encoding='utf-8') as f:
        meta_data = json.load(f)

    eng_zh = {}
    for _, ele in meta_data.items():
        for big_en, des in ele['classes'].items():
            zh = des['name']
            en = big_en.lower().replace('-', ' ').replace('_', ' ')

            print(en, zh)

            eng_zh[en] = ele['name'] + '-' + zh

    labels = []
    with open('../outputs/weed-sample/output_labels.txt', 'r') as f:
        for line in f.readlines():
            labels.append(line.strip())

    out_json = []

    for l in labels:
        out_json.append({'cname': eng_zh[l], 'description': ''})

    with open('../data/weed_labels.json', 'w', encoding='utf-8') as wf:
        json.dump(out_json, wf, indent=2, ensure_ascii=False)


if __name__ == '__main__':
    transform()
