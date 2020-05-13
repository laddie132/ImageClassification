#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = "Han"
__email__ = "liuhan132@foxmail.com"

import json, argparse, time

import tensorflow as tf

from flask import Flask, request
from flask_cors import CORS


##################################################
# API part
##################################################
app = Flask(__name__)
cors = CORS(app)


@app.route("/", methods=['POST'])
def predict():
    start = time.time()

    data = request.data.decode("utf-8")
    if data == "":
        params = request.form
        x_in = json.loads(params['data']['image'])
    else:
        params = json.loads(data)
        x_in = params['data']['image']

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

    json_data = json.dumps({'prediction': y_out.tolist()})
    print("Time spent handling the request: %f" % (time.time() - start))

    return json_data


##################################################
# END API part
##################################################

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path",
        default="outputs/saved_model_b64",
        type=str,
        help="Frozen model file to import")
    args = parser.parse_args()


    ##################################################
    # Tensorflow part
    ##################################################
    print('Loading the model')
    with tf.Session() as sess:
        tf.saved_model.loader.load(sess, ["serve"], args.model_path)
        graph = tf.get_default_graph()
    persistent_sess = tf.Session(graph=graph)

    input_name = "base64_string:0"
    output_name = "myOutput:0"
    x = graph.get_tensor_by_name(input_name)
    y = graph.get_tensor_by_name(output_name)

    ##################################################
    # END Tensorflow part
    ##################################################

    print('Starting the API')
    app.run(host='0.0.0.0', port=8500, debug=False, ssl_context=('test.pem', 'test.key'))
