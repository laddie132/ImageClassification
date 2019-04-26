#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = "Han"
__email__ = "liuhan132@foxmail.com"

import tensorflow as tf
from tensorflow.python.saved_model import tag_constants


def convert(model_path, graph_file_name, final_tensor_name):
    with tf.Session(graph=tf.Graph()) as sess:
        tf.saved_model.loader.load(sess, [tag_constants.SERVING], model_path)
        graph = sess.graph

        output_graph_def = tf.graph_util.convert_variables_to_constants(
            sess, graph.as_graph_def(), [final_tensor_name])

    with tf.gfile.GFile(graph_file_name, 'wb') as f:
        f.write(output_graph_def.SerializeToString())


if __name__ == '__main__':
    convert('/home/lh/WeedClassification/outputs/weed-sample/saved_model',
            '/home/lh/WeedClassification/outputs/weed-sample/frozen_graph.pb',
            'final_result')