#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = "Han"
__email__ = "liuhan132@foxmail.com"

import argparse
import os
import json
import numpy as np
import tensorflow as tf


def load_graph(model_file):
    graph = tf.Graph()
    graph_def = tf.GraphDef()

    with open(model_file, "rb") as f:
        graph_def.ParseFromString(f.read())
    with graph.as_default():
        tf.import_graph_def(graph_def)

    return graph


def read_tensor_from_image_list(file_names,
                                input_height=299,
                                input_width=299,
                                input_mean=0,
                                input_std=255):
    input_name = "file_reader"
    output_name = "normalized"

    image_tensors = []
    for name in file_names:
        file_reader = tf.read_file(name, input_name)
        if name.endswith(".png"):
            image_reader = tf.image.decode_png(
                file_reader, channels=3, name="png_reader")
        elif name.endswith(".gif"):
            image_reader = tf.squeeze(
                tf.image.decode_gif(file_reader, name="gif_reader"))
        elif name.endswith(".bmp"):
            image_reader = tf.image.decode_bmp(file_reader, name="bmp_reader")
        else:
            image_reader = tf.image.decode_jpeg(
                file_reader, channels=3, name="jpeg_reader")
        float_caster = tf.cast(image_reader, tf.float32)
        dims_expander = tf.expand_dims(float_caster, 0)
        resized = tf.image.resize_bilinear(dims_expander, [input_height, input_width])

        image_tensors.append(resized)
    resized_tensors = tf.concat(image_tensors, axis=0)
    normalized = tf.divide(tf.subtract(resized_tensors, [input_mean]), [input_std])
    sess = tf.Session()
    result = sess.run(normalized)

    return result


def load_labels(label_file):
    label = []
    proto_as_ascii_lines = tf.gfile.GFile(label_file).readlines()
    for l in proto_as_ascii_lines:
        label.append(l.rstrip())
    return label


if __name__ == '__main__':
    image_root_path = '/home/lh/weed_photos_resize/'
    output_path = '/home/lh/WeedClassification/outputs/weed-sample/'
    image_list_file = "data/weed_image_lists_oversample.json"
    results_file = output_path + 'test_results.json'
    model_file = output_path + "frozen_graph.pb"
    label_file = output_path + "output_labels.txt"
    input_height = 299
    input_width = 299
    input_mean = 0
    input_std = 255
    input_layer = "Placeholder"
    output_layer = "final_result"
    top_n = 3

    parser = argparse.ArgumentParser()
    parser.add_argument("--image", help="image to be processed")
    parser.add_argument("--graph", help="graph/model to be executed")
    parser.add_argument("--labels", help="name of file containing labels")
    parser.add_argument("--input_height", type=int, help="input height")
    parser.add_argument("--input_width", type=int, help="input width")
    parser.add_argument("--input_mean", type=int, help="input mean")
    parser.add_argument("--input_std", type=int, help="input std")
    parser.add_argument("--input_layer", help="name of input layer")
    parser.add_argument("--output_layer", help="name of output layer")
    parser.add_argument("--top_n", type=int, help='Top-n accuracy')
    args = parser.parse_args()

    if args.graph:
        model_file = args.graph
    if args.image:
        file_name = args.image
    if args.labels:
        label_file = args.labels
    if args.input_height:
        input_height = args.input_height
    if args.input_width:
        input_width = args.input_width
    if args.input_mean:
        input_mean = args.input_mean
    if args.input_std:
        input_std = args.input_std
    if args.input_layer:
        input_layer = args.input_layer
    if args.output_layer:
        output_layer = args.output_layer

    graph = load_graph(model_file)
    sess = tf.Session(graph=graph)

    input_name = "import/" + input_layer + ':0'
    output_name = "import/" + output_layer + ':0'
    input_tensor = graph.get_tensor_by_name(input_name)
    output_tensor = graph.get_tensor_by_name(output_name)

    labels = load_labels(label_file)

    # read images
    with open(image_list_file, 'r') as f:
        image_lists = json.load(f)

    output_results = {}

    for k, v in image_lists.items():
        cur_result = {'dir': v['dir']}

        test_images = v['testing']
        dir_name = v['dir']
        images_path = list(map(lambda x: os.path.join(image_root_path,
                                                      os.path.join(dir_name, x)), test_images))
        image_tensor = read_tensor_from_image_list(images_path)

        results = sess.run(output_tensor,
                           {input_tensor: image_tensor})

        top_k = np.argsort(-results, axis=1)[:, :top_n]

        ground_truth_id = labels.index(k)
        is_right = np.sum((top_k == ground_truth_id), axis=1) > 0

        wrong_idx = np.where(is_right == 0)[0].tolist()

        cur_wrong_imgs = {}
        for wid in wrong_idx:
            wname = test_images[wid]
            wpredict = top_k[wid].tolist()
            wpredict_label = list(map(lambda x: labels[x], wpredict))
            wpredict_prob = list(map(lambda x: '%.2f' % results[wid, x], wpredict))

            cur_wrong_imgs[wname] = dict(zip(wpredict_label, wpredict_prob))

        cur_result['wrong_images'] = cur_wrong_imgs

        acc = sum(is_right) / len(test_images)
        cur_result['acc'] = acc
        cur_result['top-n'] = top_n
        cur_result['train_number'] = len(set(v['training']))
        cur_result['test_number'] = len(test_images)
        cur_result['valid_number'] = len(v['validation'])
        cur_result['all_number'] = cur_result['test_number'] + cur_result['train_number'] + cur_result['valid_number']

        output_results[k] = cur_result

        print('%s with top-%d Acc: %.2f' % (k, top_n, acc * 100))

    with open(results_file, 'w') as wf:
        json.dump(output_results, wf, indent=2)
