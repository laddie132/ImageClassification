#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = "Han"
__email__ = "liuhan132@foxmail.com"

import argparse
import time
import tensorflow as tf


"""Test the QPS on server"""


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


if __name__ == '__main__':
    test_image_path = '../data/test_weed.jpg'
    output_path = '../outputs/weed-inaturalist-inception-inception_resnet/'
    model_file = output_path + "frozen_graph.pb"
    input_height = 299
    input_width = 299
    input_mean = 0
    input_std = 255
    input_layer = "Placeholder"
    output_layer = "final_result"
    top_n = 3
    test_iter_num = 20

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

    image_tensor = read_tensor_from_image_list([test_image_path])
    results = sess.run(output_tensor,
                       {input_tensor: image_tensor})

    ave_time = 0.
    for i in range(test_iter_num):

        t1 = time.time()
        image_tensor = read_tensor_from_image_list([test_image_path])
        results = sess.run(output_tensor,
                           {input_tensor: image_tensor})
        t2 = time.time()

        ave_time += (t2 - t1)

        if i % 10 == 0:
            print('No.%d testing time: %.2f(s)' % (i, t2-t1))

    qps = test_iter_num / ave_time
    ave_time /= test_iter_num

    print('Per image ave test cost time: %.2f(s)' % ave_time)
    print('QPS: %.2f' % qps)
