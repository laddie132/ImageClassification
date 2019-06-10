#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = "Han"
__email__ = "liuhan132@foxmail.com"

import argparse
import requests
import json
import tensorflow as tf
import numpy as np
import base64
import time


def load_labels(label_file):
    label = []
    proto_as_ascii_lines = tf.gfile.GFile(label_file).readlines()
    for l in proto_as_ascii_lines:
        label.append(l.rstrip())
    return label


if __name__ == "__main__":
    file_name = "data/test_weed.jpg"
    label_file = "outputs/weed-mix-sample/output_labels.txt"
    model_name = "default"
    model_version = 2
    enable_ssl = True
    ip = 'www.cropphoto.cn'

    parser = argparse.ArgumentParser()
    parser.add_argument("--image", help="image to be processed")
    parser.add_argument("--labels", help="name of file containing labels")
    parser.add_argument("--model_name", help="name of predict model")
    parser.add_argument("--model_version", type=int, help="version of predict model")
    parser.add_argument("--enable_ssl", type=bool, help="if use https")
    args = parser.parse_args()

    if args.image:
        file_name = args.image
    if args.labels:
        label_file = args.labels
    if args.model_name:
        model_name = args.model_name
    if args.enable_ssl:
        enable_ssl = args.enable_ssl

    with open(file_name, "rb") as image_file:
        encoded_string = str(base64.urlsafe_b64encode(image_file.read()), "utf-8")

    if enable_ssl:
        endpoint = "https://%s:8500" % ip
    else:
        endpoint = "http://%s:8500" % ip

    json_data = {"model_name": model_name,
                 # "model_version": model_version,
                 "data": {"image": encoded_string}
                 }

    t1 = time.time()

    result = requests.post(endpoint, json=json_data)
    res = np.array(json.loads(result.text)["prediction"][0])

    print('Cost: %.2f s' % (time.time() - t1))

    indexes = np.argsort(-res)
    labels = load_labels(label_file)
    top_k = 3
    for i in range(top_k):
        idx = indexes[i]
        print(labels[idx], res[idx])

    # testing average testing cost & QPS
    test_iter_num = 20

    t2 = time.time()
    for i in range(test_iter_num):
        result = requests.post(endpoint, json=json_data)

        if i % 10 == 0:
            print('No.%d' % i)

    all_time = time.time() - t2

    ave_time = all_time / test_iter_num
    qps = test_iter_num / all_time

    print('Per image testing time: %.2f(s)' % ave_time)
    print('QPS: %.2f' % qps)
