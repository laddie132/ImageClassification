#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = "Han"
__email__ = "liuhan132@foxmail.com"

import json, argparse, time

import tensorflow as tf

from flask import Flask, request
from flask_cors import CORS


def load_graph(model_file):
    graph = tf.Graph()
    graph_def = tf.GraphDef()

    with open(model_file, "rb") as f:
        graph_def.ParseFromString(f.read())
    with graph.as_default():
        tf.import_graph_def(graph_def)

    return graph

##################################################
# API part
##################################################
app = Flask(__name__)
cors = CORS(app)


@app.route("/api/predict", methods=['POST'])
def predict():
    start = time.time()

    data = request.data.decode("utf-8")
    if data == "":
        params = request.form
        x_in = json.loads(params['x'])
    else:
        params = json.loads(data)
        x_in = params['x']

    ##################################################
    # Tensorflow part
    ##################################################
    y_out = persistent_sess.run(y, feed_dict={
        x: x_in
        # x: [[3, 5, 7, 4, 5, 1, 1, 1, 1, 1]] # < 45
    })
    ##################################################
    # END Tensorflow part
    ##################################################

    json_data = json.dumps({'y': y_out.tolist()})
    print("Time spent handling the request: %f" % (time.time() - start))

    return json_data


##################################################
# END API part
##################################################

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--frozen_model_filename", default="outputs/weed-sample/frozen_graph.pb", type=str,
                        help="Frozen model file to import")
    parser.add_argument("--gpu_memory", default=.2, type=float, help="GPU memory per process")
    args = parser.parse_args()

    ##################################################
    # Tensorflow part
    ##################################################
    print('Loading the model')
    graph = load_graph(args.frozen_model_filename)

    input_name = "import/Placeholder:0"
    output_name = "import/final_result:0"
    x = graph.get_tensor_by_name(input_name)
    y = graph.get_tensor_by_name(output_name)

    print('Starting Session, setting the GPU memory usage to %f' % args.gpu_memory)
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_memory)
    sess_config = tf.ConfigProto(gpu_options=gpu_options)
    persistent_sess = tf.Session(graph=graph, config=sess_config)
    ##################################################
    # END Tensorflow part
    ##################################################

    print('Starting the API')
    app.run()