# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
# NOTICE: This work was derived from tensorflow/examples/image_retraining
# and modified to use TensorFlow Hub modules.

# pylint: disable=line-too-long
r"""Simple transfer learning with image modules from TensorFlow Hub.

This example shows how to train an image classifier based on any
TensorFlow Hub module that computes image feature vectors. By default,
it uses the feature vectors computed by Inception V3 trained on ImageNet.
For more options, search https://tfhub.dev for image feature vector modules.

The top layer receives as input a 2048-dimensional vector (assuming
Inception V3) for each image. We train a softmax layer on top of this
representation. If the softmax layer contains N labels, this corresponds
to learning N + 2048*N model parameters for the biases and weights.

Here's an example, which assumes you have a folder containing class-named
subfolders, each full of images for each label. The example folder flower_photos
should have a structure like this:

~/flower_photos/daisy/photo1.jpg
~/flower_photos/daisy/photo2.jpg
...
~/flower_photos/rose/anotherphoto77.jpg
...
~/flower_photos/sunflower/somepicture.jpg

The subfolder names are important, since they define what label is applied to
each image, but the filenames themselves don't matter. (For a working example,
download http://download.tensorflow.org/example_images/flower_photos.tgz
and run  tar xzf flower_photos.tgz  to unpack it.)

Once your images are prepared, and you have pip-installed tensorflow-hub and
a sufficiently recent version of tensorflow, you can run the training with a
command like this:

```bash
python retrain.py --image_dir ~/flower_photos
```

You can replace the image_dir argument with any folder containing subfolders of
images. The label for each image is taken from the name of the subfolder it's
in.

This produces a new model file that can be loaded and run by any TensorFlow
program, for example the tensorflow/examples/label_image sample code.

By default this script will use the highly accurate, but comparatively large and
slow Inception V3 model architecture. It's recommended that you start with this
to validate that you have gathered good training data, but if you want to deploy
on resource-limited platforms, you can try the `--tfhub_module` flag with a
Mobilenet model. For more information on Mobilenet, see
https://research.googleblog.com/2017/06/mobilenets-open-source-models-for.html

For example:

Run floating-point version of Mobilenet:

```bash
python retrain.py --image_dir ~/flower_photos \
    --tfhub_module https://tfhub.dev/google/imagenet/mobilenet_v1_100_224/feature_vector/1
```

Run Mobilenet, instrumented for quantization:

```bash
python retrain.py --image_dir ~/flower_photos/ \
    --tfhub_module https://tfhub.dev/google/imagenet/mobilenet_v1_100_224/quantops/feature_vector/1
```

These instrumented models can be converted to fully quantized mobile models via
TensorFlow Lite.

There are different Mobilenet models to choose from, with a variety of file
size and latency options.
  - The first number can be '100', '075', '050', or '025' to control the number
    of neurons (activations of hidden layers); the number of weights (and hence
    to some extent the file size and speed) shrinks with the square of that
    fraction.
  - The second number is the input image size. You can choose '224', '192',
    '160', or '128', with smaller sizes giving faster speeds.

To use with TensorBoard:

By default, this script will log summaries to /tmp/retrain_logs directory

Visualize the summaries with this command:

tensorboard --logdir /tmp/retrain_logs

To use with Tensorflow Serving, run this tool with --saved_model_dir set
to some increasingly numbered export location under the model base path, e.g.:

```bash
python retrain.py (... other args as before ...) \
    --saved_model_dir=/tmp/saved_models/$(date +%s)/
tensorflow_model_server --port=9000 --model_name=my_image_classifier \
    --model_base_path=/tmp/saved_models/
```
"""
# pylint: enable=line-too-long

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import collections
from datetime import datetime
import hashlib
import os.path
import random
import re
import sys
from functools import reduce

import json
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

FLAGS = None

MAX_NUM_IMAGES_PER_CLASS = 2 ** 27 - 1  # ~134M

# The location where variable checkpoints will be stored.
CHECKPOINT_NAME = '/tmp/_retrain_checkpoint'

# A module is understood as instrumented for quantization with TF-Lite
# if it contains any of these ops.
FAKE_QUANT_OPS = ('FakeQuantWithMinMaxVars',
                  'FakeQuantWithMinMaxVarsPerChannel')


def get_image_path(image_lists, label_name, index, image_dir, category):
    """Returns a path to an image for a label at the given index.

    Args:
      image_lists: OrderedDict of training images for each label.
      label_name: Label string we want to get an image for.
      index: Int offset of the image we want. This will be moduloed by the
      available number of images for the label, so it can be arbitrarily large.
      image_dir: Root folder string of the subfolders containing the training
      images.
      category: Name string of set to pull images from - training, testing, or
      validation.

    Returns:
      File system path string to an image that meets the requested parameters.

    """
    if label_name not in image_lists:
        tf.logging.fatal('Label does not exist %s.', label_name)
    label_lists = image_lists[label_name]
    if category not in label_lists:
        tf.logging.fatal('Category does not exist %s.', category)
    category_list = label_lists[category]
    if not category_list:
        tf.logging.fatal('Label %s has no images in the category %s.',
                         label_name, category)
    mod_index = index % len(category_list)
    base_name = category_list[mod_index]
    sub_dir = label_lists['dir']
    full_path = os.path.join(image_dir, sub_dir, base_name)
    return full_path


def get_bottleneck_path(image_lists, label_name, index, bottleneck_dir,
                        category, module_name):
    """Returns a path to a bottleneck file for a label at the given index.

    Args:
      image_lists: OrderedDict of training images for each label.
      label_name: Label string we want to get an image for.
      index: Integer offset of the image we want. This will be moduloed by the
      available number of images for the label, so it can be arbitrarily large.
      bottleneck_dir: Folder string holding cached files of bottleneck values.
      category: Name string of set to pull images from - training, testing, or
      validation.
      module_name: The name of the image module being used.

    Returns:
      File system path string to an image that meets the requested parameters.
    """
    module_name = (module_name.replace('://', '~')  # URL scheme.
                   .replace('/', '~')  # URL and Unix paths.
                   .replace(':', '~').replace('\\', '~'))  # Windows paths.
    return get_image_path(image_lists, label_name, index, bottleneck_dir,
                          category) + '_' + module_name + '.txt'


def create_module_graph(module_spec):
    """Creates a graph and loads Hub Module into it.

    Args:
      module_spec: the hub.ModuleSpec for the image module being used.

    Returns:
      graph: the tf.Graph that was created.
      bottleneck_tensor: the bottleneck values output by the module.
      resized_input_tensor: the input images, resized as expected by the module.
      wants_quantization: a boolean, whether the module has been instrumented
        with fake quantization ops.
    """
    height, width = hub.get_expected_image_size(module_spec[0])
    with tf.Graph().as_default() as graph:
        resized_input_tensor = tf.placeholder(tf.float32, [None, height, width, 3])

        bottleneck_tensor = []
        for spec in module_spec:
            m = hub.Module(spec, trainable=FLAGS.module_trainable)
            bottleneck_tensor.append(m(resized_input_tensor))

        bottleneck_tensor = tf.concat(bottleneck_tensor, axis=-1)
        wants_quantization = any(node.op in FAKE_QUANT_OPS
                                 for node in graph.as_graph_def().node)
    return graph, bottleneck_tensor, resized_input_tensor, wants_quantization


def run_bottleneck_on_image(sess, image_data, image_data_tensor,
                            decoded_image_tensor, resized_input_tensor,
                            bottleneck_tensor):
    """Runs inference on an image to extract the 'bottleneck' summary layer.

    Args:
      sess: Current active TensorFlow Session.
      image_data: String of raw JPEG data.
      image_data_tensor: Input data layer in the graph.
      decoded_image_tensor: Output of initial image resizing and preprocessing.
      resized_input_tensor: The input node of the recognition graph.
      bottleneck_tensor: Layer before the final softmax.

    Returns:
      Numpy array of bottleneck values.
    """
    # First decode the JPEG image, resize it, and rescale the pixel values.
    resized_input_values = sess.run(decoded_image_tensor,
                                    {image_data_tensor: image_data})
    # Then run it through the recognition network.
    bottleneck_values = sess.run(bottleneck_tensor,
                                 {resized_input_tensor: resized_input_values})
    bottleneck_values = np.squeeze(bottleneck_values)
    return bottleneck_values


def ensure_dir_exists(dir_name):
    """Makes sure the folder exists on disk.

    Args:
      dir_name: Path string to the folder we want to create.
    """
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)


def create_bottleneck_file(bottleneck_path, image_lists, label_name, index,
                           image_dir, category, sess, jpeg_data_tensor,
                           decoded_image_tensor, resized_input_tensor,
                           bottleneck_tensor):
    """Create a single bottleneck file."""
    tf.logging.debug('Creating bottleneck at ' + bottleneck_path)
    image_path = get_image_path(image_lists, label_name, index,
                                image_dir, category)
    if not tf.gfile.Exists(image_path):
        tf.logging.fatal('File does not exist %s', image_path)
    image_data = tf.gfile.GFile(image_path, 'rb').read()
    try:
        bottleneck_values = run_bottleneck_on_image(
            sess, image_data, jpeg_data_tensor, decoded_image_tensor,
            resized_input_tensor, bottleneck_tensor)
    except Exception as e:
        raise RuntimeError('Error during processing file %s (%s)' % (image_path,
                                                                     str(e)))
    bottleneck_string = ','.join(str(x) for x in bottleneck_values)
    with open(bottleneck_path, 'w') as bottleneck_file:
        bottleneck_file.write(bottleneck_string)


def get_or_create_bottleneck(sess, image_lists, label_name, index, image_dir,
                             category, bottleneck_dir, jpeg_data_tensor,
                             decoded_image_tensor, resized_input_tensor,
                             bottleneck_tensor, module_name):
    """Retrieves or calculates bottleneck values for an image.

    If a cached version of the bottleneck data exists on-disk, return that,
    otherwise calculate the data and save it to disk for future use.

    Args:
      sess: The current active TensorFlow Session.
      image_lists: OrderedDict of training images for each label.
      label_name: Label string we want to get an image for.
      index: Integer offset of the image we want. This will be modulo-ed by the
      available number of images for the label, so it can be arbitrarily large.
      image_dir: Root folder string of the subfolders containing the training
      images.
      category: Name string of which set to pull images from - training, testing,
      or validation.
      bottleneck_dir: Folder string holding cached files of bottleneck values.
      jpeg_data_tensor: The tensor to feed loaded jpeg data into.
      decoded_image_tensor: The output of decoding and resizing the image.
      resized_input_tensor: The input node of the recognition graph.
      bottleneck_tensor: The output tensor for the bottleneck values.
      module_name: The name of the image module being used.

    Returns:
      Numpy array of values produced by the bottleneck layer for the image.
    """
    label_lists = image_lists[label_name]
    sub_dir = label_lists['dir']
    sub_dir_path = os.path.join(bottleneck_dir, sub_dir)
    ensure_dir_exists(sub_dir_path)
    bottleneck_path = get_bottleneck_path(image_lists, label_name, index,
                                          bottleneck_dir, category, module_name)
    if not os.path.exists(bottleneck_path):
        create_bottleneck_file(bottleneck_path, image_lists, label_name, index,
                               image_dir, category, sess, jpeg_data_tensor,
                               decoded_image_tensor, resized_input_tensor,
                               bottleneck_tensor)
    with open(bottleneck_path, 'r') as bottleneck_file:
        bottleneck_string = bottleneck_file.read()
    did_hit_error = False
    try:
        bottleneck_values = [float(x) for x in bottleneck_string.split(',')]
    except ValueError:
        tf.logging.warning('Invalid float found, recreating bottleneck')
        did_hit_error = True
    if did_hit_error:
        create_bottleneck_file(bottleneck_path, image_lists, label_name, index,
                               image_dir, category, sess, jpeg_data_tensor,
                               decoded_image_tensor, resized_input_tensor,
                               bottleneck_tensor)
        with open(bottleneck_path, 'r') as bottleneck_file:
            bottleneck_string = bottleneck_file.read()
        # Allow exceptions to propagate here, since they shouldn't happen after a
        # fresh creation
        bottleneck_values = [float(x) for x in bottleneck_string.split(',')]
    return bottleneck_values


def cache_bottlenecks(sess, image_lists, image_dir, bottleneck_dir,
                      jpeg_data_tensor, decoded_image_tensor,
                      resized_input_tensor, bottleneck_tensor, module_name):
    """Ensures all the training, testing, and validation bottlenecks are cached.

    Because we're likely to read the same image multiple times (if there are no
    distortions applied during training) it can speed things up a lot if we
    calculate the bottleneck layer values once for each image during
    preprocessing, and then just read those cached values repeatedly during
    training. Here we go through all the images we've found, calculate those
    values, and save them off.

    Args:
      sess: The current active TensorFlow Session.
      image_lists: OrderedDict of training images for each label.
      image_dir: Root folder string of the subfolders containing the training
      images.
      bottleneck_dir: Folder string holding cached files of bottleneck values.
      jpeg_data_tensor: Input tensor for jpeg data from file.
      decoded_image_tensor: The output of decoding and resizing the image.
      resized_input_tensor: The input node of the recognition graph.
      bottleneck_tensor: The penultimate output layer of the graph.
      module_name: The name of the image module being used.

    Returns:
      Nothing.
    """
    how_many_bottlenecks = 0
    ensure_dir_exists(bottleneck_dir)
    for label_name, label_lists in image_lists.items():
        for category in ['training', 'testing', 'validation']:
            category_list = label_lists[category]
            for index, unused_base_name in enumerate(category_list):
                get_or_create_bottleneck(
                    sess, image_lists, label_name, index, image_dir, category,
                    bottleneck_dir, jpeg_data_tensor, decoded_image_tensor,
                    resized_input_tensor, bottleneck_tensor, module_name)

                how_many_bottlenecks += 1
                if how_many_bottlenecks % 100 == 0:
                    tf.logging.info(
                        str(how_many_bottlenecks) + ' bottleneck files created.')


def get_image_data_tensor(sess, decoded_image_tensor, image_data_tensor,
                          image_lists, label_name, index, image_dir, category):
    image_path = get_image_path(image_lists, label_name, index,
                                image_dir, category)
    if not tf.gfile.Exists(image_path):
        tf.logging.fatal('File does not exist %s', image_path)
    image_data = tf.gfile.GFile(image_path, 'rb').read()
    try:
        resized_input_values = sess.run(decoded_image_tensor,
                                        {image_data_tensor: image_data})
        resized_input_values = np.squeeze(resized_input_values)
    except Exception as e:
        raise RuntimeError('Error during processing file %s (%s)' % (image_path,
                                                                     str(e)))

    return resized_input_values


def get_random_cached_bottlenecks(sess, image_lists, top_image_labels, how_many, category,
                                  bottleneck_dir, image_dir, jpeg_data_tensor,
                                  decoded_image_tensor, resized_input_tensor,
                                  bottleneck_tensor, module_name, module_trainable):
    """Retrieves bottleneck values for cached images.

    If no distortions are being applied, this function can retrieve the cached
    bottleneck values directly from disk for images. It picks a random set of
    images from the specified category.

    Args:
      sess: Current TensorFlow Session.
      image_lists: OrderedDict of training images for each label.
      how_many: If positive, a random sample of this size will be chosen.
      If negative, all bottlenecks will be retrieved.
      category: Name string of which set to pull from - training, testing, or
      validation.
      bottleneck_dir: Folder string holding cached files of bottleneck values.
      image_dir: Root folder string of the subfolders containing the training
      images.
      jpeg_data_tensor: The layer to feed jpeg image data into.
      decoded_image_tensor: The output of decoding and resizing the image.
      resized_input_tensor: The input node of the recognition graph.
      bottleneck_tensor: The bottleneck output layer of the CNN graph.
      module_name: The name of the image module being used.

    Returns:
      List of bottleneck arrays, their corresponding ground truths, and the
      relevant filenames.
    """
    class_count = len(image_lists.keys())
    bottlenecks = []
    ground_truths = []
    ground_truths_top_label = []
    filenames = []
    if how_many >= 0:
        # Retrieve a random sample of bottlenecks.
        for unused_i in range(how_many):
            label_index = random.randrange(class_count)
            label_name = list(image_lists.keys())[label_index]
            top_label_name = extract_top_label(label_name)
            top_label_index = top_image_labels.index(top_label_name)
            image_index = random.randrange(MAX_NUM_IMAGES_PER_CLASS + 1)
            image_name = get_image_path(image_lists, label_name, image_index,
                                        image_dir, category)

            if module_trainable:
                bottleneck = get_image_data_tensor(sess, decoded_image_tensor, jpeg_data_tensor,
                                                   image_lists, label_name, image_index, image_dir, category)
            else:
                bottleneck = get_or_create_bottleneck(
                    sess, image_lists, label_name, image_index, image_dir, category,
                    bottleneck_dir, jpeg_data_tensor, decoded_image_tensor,
                    resized_input_tensor, bottleneck_tensor, module_name)
            bottlenecks.append(bottleneck)
            ground_truths.append(label_index)
            ground_truths_top_label.append(top_label_index)
            filenames.append(image_name)
    else:
        # Retrieve all bottlenecks.
        for label_index, label_name in enumerate(image_lists.keys()):
            top_label_name = extract_top_label(label_name)
            top_label_index = top_image_labels.index(top_label_name)

            for image_index, image_name in enumerate(
                    image_lists[label_name][category]):
                image_name = get_image_path(image_lists, label_name, image_index,
                                            image_dir, category)
                if module_trainable:
                    bottleneck = get_image_data_tensor(sess, decoded_image_tensor, jpeg_data_tensor,
                                                       image_lists, label_name, image_index, image_dir, category)
                else:
                    bottleneck = get_or_create_bottleneck(
                        sess, image_lists, label_name, image_index, image_dir, category,
                        bottleneck_dir, jpeg_data_tensor, decoded_image_tensor,
                        resized_input_tensor, bottleneck_tensor, module_name)
                bottlenecks.append(bottleneck)
                ground_truths.append(label_index)
                ground_truths_top_label.append(top_label_index)
                filenames.append(image_name)

    return bottlenecks, ground_truths, ground_truths_top_label, filenames


def get_random_distorted_bottlenecks(
        sess, image_lists, how_many, category, image_dir, input_jpeg_tensor,
        distorted_image, resized_input_tensor, bottleneck_tensor):
    """Retrieves bottleneck values for training images, after distortions.

    If we're training with distortions like crops, scales, or flips, we have to
    recalculate the full model for every image, and so we can't use cached
    bottleneck values. Instead we find random images for the requested category,
    run them through the distortion graph, and then the full graph to get the
    bottleneck results for each.

    Args:
      sess: Current TensorFlow Session.
      image_lists: OrderedDict of training images for each label.
      how_many: The integer number of bottleneck values to return.
      category: Name string of which set of images to fetch - training, testing,
      or validation.
      image_dir: Root folder string of the subfolders containing the training
      images.
      input_jpeg_tensor: The input layer we feed the image data to.
      distorted_image: The output node of the distortion graph.
      resized_input_tensor: The input node of the recognition graph.
      bottleneck_tensor: The bottleneck output layer of the CNN graph.

    Returns:
      List of bottleneck arrays and their corresponding ground truths.
    """
    class_count = len(image_lists.keys())
    bottlenecks = []
    ground_truths = []
    for unused_i in range(how_many):
        label_index = random.randrange(class_count)
        label_name = list(image_lists.keys())[label_index]
        image_index = random.randrange(MAX_NUM_IMAGES_PER_CLASS + 1)
        image_path = get_image_path(image_lists, label_name, image_index, image_dir,
                                    category)
        if not tf.gfile.Exists(image_path):
            tf.logging.fatal('File does not exist %s', image_path)
        jpeg_data = tf.gfile.GFile(image_path, 'rb').read()
        # Note that we materialize the distorted_image_data as a numpy array before
        # sending running inference on the image. This involves 2 memory copies and
        # might be optimized in other implementations.
        distorted_image_data = sess.run(distorted_image,
                                        {input_jpeg_tensor: jpeg_data})
        bottleneck_values = sess.run(bottleneck_tensor,
                                     {resized_input_tensor: distorted_image_data})
        bottleneck_values = np.squeeze(bottleneck_values)
        bottlenecks.append(bottleneck_values)
        ground_truths.append(label_index)
    return bottlenecks, ground_truths


def should_distort_images(flip_left_right, random_crop, random_scale,
                          random_brightness):
    """Whether any distortions are enabled, from the input flags.

    Args:
      flip_left_right: Boolean whether to randomly mirror images horizontally.
      random_crop: Integer percentage setting the total margin used around the
      crop box.
      random_scale: Integer percentage of how much to vary the scale by.
      random_brightness: Integer range to randomly multiply the pixel values by.

    Returns:
      Boolean value indicating whether any distortions should be applied.
    """
    return (flip_left_right or (random_crop != 0) or (random_scale != 0) or
            (random_brightness != 0))


def add_input_distortions(flip_left_right, random_crop, random_scale,
                          random_brightness, module_spec):
    """Creates the operations to apply the specified distortions.

    During training it can help to improve the results if we run the images
    through simple distortions like crops, scales, and flips. These reflect the
    kind of variations we expect in the real world, and so can help train the
    model to cope with natural data more effectively. Here we take the supplied
    parameters and construct a network of operations to apply them to an image.

    Cropping
    ~~~~~~~~

    Cropping is done by placing a bounding box at a random position in the full
    image. The cropping parameter controls the size of that box relative to the
    input image. If it's zero, then the box is the same size as the input and no
    cropping is performed. If the value is 50%, then the crop box will be half the
    width and height of the input. In a diagram it looks like this:

    <       width         >
    +---------------------+
    |                     |
    |   width - crop%     |
    |    <      >         |
    |    +------+         |
    |    |      |         |
    |    |      |         |
    |    |      |         |
    |    +------+         |
    |                     |
    |                     |
    +---------------------+

    Scaling
    ~~~~~~~

    Scaling is a lot like cropping, except that the bounding box is always
    centered and its size varies randomly within the given range. For example if
    the scale percentage is zero, then the bounding box is the same size as the
    input and no scaling is applied. If it's 50%, then the bounding box will be in
    a random range between half the width and height and full size.

    Args:
      flip_left_right: Boolean whether to randomly mirror images horizontally.
      random_crop: Integer percentage setting the total margin used around the
      crop box.
      random_scale: Integer percentage of how much to vary the scale by.
      random_brightness: Integer range to randomly multiply the pixel values by.
      graph.
      module_spec: The hub.ModuleSpec for the image module being used.

    Returns:
      The jpeg input layer and the distorted result tensor.
    """
    input_height, input_width = hub.get_expected_image_size(module_spec)
    input_depth = hub.get_num_image_channels(module_spec)
    jpeg_data = tf.placeholder(tf.string, name='DistortJPGInput')
    decoded_image = tf.image.decode_jpeg(jpeg_data, channels=input_depth)
    # Convert from full range of uint8 to range [0,1] of float32.
    decoded_image_as_float = tf.image.convert_image_dtype(decoded_image,
                                                          tf.float32)
    decoded_image_4d = tf.expand_dims(decoded_image_as_float, 0)
    margin_scale = 1.0 + (random_crop / 100.0)
    resize_scale = 1.0 + (random_scale / 100.0)
    margin_scale_value = tf.constant(margin_scale)
    resize_scale_value = tf.random_uniform(shape=[],
                                           minval=1.0,
                                           maxval=resize_scale)
    scale_value = tf.multiply(margin_scale_value, resize_scale_value)
    precrop_width = tf.multiply(scale_value, input_width)
    precrop_height = tf.multiply(scale_value, input_height)
    precrop_shape = tf.stack([precrop_height, precrop_width])
    precrop_shape_as_int = tf.cast(precrop_shape, dtype=tf.int32)
    precropped_image = tf.image.resize_bilinear(decoded_image_4d,
                                                precrop_shape_as_int)
    precropped_image_3d = tf.squeeze(precropped_image, axis=[0])
    cropped_image = tf.random_crop(precropped_image_3d,
                                   [input_height, input_width, input_depth])
    if flip_left_right:
        flipped_image = tf.image.random_flip_left_right(cropped_image)
    else:
        flipped_image = cropped_image
    brightness_min = 1.0 - (random_brightness / 100.0)
    brightness_max = 1.0 + (random_brightness / 100.0)
    brightness_value = tf.random_uniform(shape=[],
                                         minval=brightness_min,
                                         maxval=brightness_max)
    brightened_image = tf.multiply(flipped_image, brightness_value)
    distort_result = tf.expand_dims(brightened_image, 0, name='DistortResult')
    return jpeg_data, distort_result


def variable_summaries(var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)


def add_final_retrain_ops(class_count, top_class_count, final_tensor_name, final_top_tensor_name,
                          bottleneck_tensor, quantize_layer, is_training):
    """Adds a new softmax and fully-connected layer for training and eval.

    We need to retrain the top layer to identify our new classes, so this function
    adds the right operations to the graph, along with some variables to hold the
    weights, and then sets up all the gradients for the backward pass.

    The set up for the softmax and fully-connected layers is based on:
    https://www.tensorflow.org/tutorials/mnist/beginners/index.html

    Args:
      class_count: Integer of how many categories of things we're trying to
          recognize.
      final_tensor_name: Name string for the new final node that produces results.
      bottleneck_tensor: The output of the main CNN graph.
      quantize_layer: Boolean, specifying whether the newly added layer should be
          instrumented for quantization with TF-Lite.
      is_training: Boolean, specifying whether the newly add layer is for training
          or eval.

    Returns:
      The tensors for the training and cross entropy results, and tensors for the
      bottleneck input and ground truth input.
    """
    batch_size, bottleneck_tensor_size = bottleneck_tensor.get_shape().as_list()
    assert batch_size is None, 'We want to work with arbitrary batch size.'
    with tf.name_scope('input'):

        if not FLAGS.module_trainable:
            bottleneck_input = tf.placeholder_with_default(
                bottleneck_tensor,
                shape=[batch_size, bottleneck_tensor_size],
                name='BottleneckInputPlaceholder')
        else:
            bottleneck_input = bottleneck_tensor

        ground_truth_input = tf.placeholder(
            tf.int64, [batch_size], name='GroundTruthInput')

        ground_truth_input_top = tf.placeholder(
            tf.int64, [batch_size], name='GroundTruthInputTop')

    # Add linear layer
    with tf.name_scope('final_linear'):
        with tf.name_scope('weights'):
            initial_value = tf.truncated_normal(
                [bottleneck_tensor_size, bottleneck_tensor_size], stddev=0.001)
            layer_weights = tf.Variable(initial_value, name='final_weights')

        with tf.name_scope('biases'):
            layer_biases = tf.Variable(tf.zeros([bottleneck_tensor_size]), name='final_biases')

        with tf.name_scope('Wx_plus_b'):
            bottleneck_output = tf.matmul(bottleneck_input, layer_weights) + layer_biases
            bottleneck_output = tf.nn.relu(bottleneck_output)

    # Organizing the following ops so they are easier to see in TensorBoard.
    layer_name = 'final_retrain_ops'
    with tf.name_scope(layer_name):
        with tf.name_scope('weights'):
            initial_value = tf.truncated_normal(
                [bottleneck_tensor_size, class_count], stddev=0.001)
            layer_weights = tf.Variable(initial_value, name='final_weights')
            variable_summaries(layer_weights)

        with tf.name_scope('biases'):
            layer_biases = tf.Variable(tf.zeros([class_count]), name='final_biases')
            variable_summaries(layer_biases)

        with tf.name_scope('Wx_plus_b'):
            logits = tf.matmul(bottleneck_output, layer_weights) + layer_biases
            tf.summary.histogram('pre_activations', logits)

    final_tensor = tf.nn.softmax(logits, name=final_tensor_name)

    # top label
    layer_name = 'final_retrain_ops_top'
    with tf.name_scope(layer_name):
        with tf.name_scope('weights'):
            top_initial_value = tf.truncated_normal(
                [bottleneck_tensor_size, top_class_count], stddev=0.001)
            top_layer_weights = tf.Variable(top_initial_value, name='final_weights')
            variable_summaries(top_layer_weights)

        with tf.name_scope('biases'):
            top_layer_biases = tf.Variable(tf.zeros([top_class_count]), name='final_biases')
            variable_summaries(top_layer_biases)

        with tf.name_scope('Wx_plus_b'):
            top_logits = tf.matmul(bottleneck_output, top_layer_weights) + top_layer_biases
            tf.summary.histogram('pre_activations', top_logits)

    final_top_tensor = tf.nn.softmax(top_logits, name=final_top_tensor_name)

    # The tf.contrib.quantize functions rewrite the graph in place for
    # quantization. The imported model graph has already been rewritten, so upon
    # calling these rewrites, only the newly added final layer will be
    # transformed.
    if quantize_layer:
        if is_training:
            tf.contrib.quantize.create_training_graph()
        else:
            tf.contrib.quantize.create_eval_graph()

    tf.summary.histogram('activations', final_tensor)

    # If this is an eval graph, we don't need to add loss ops or an optimizer.
    if not is_training:
        return None, None, None, bottleneck_input, ground_truth_input, ground_truth_input_top, \
               final_tensor, final_top_tensor

    with tf.name_scope('cross_entropy'):
        cross_entropy_mean = tf.losses.sparse_softmax_cross_entropy(
            labels=ground_truth_input, logits=logits)

    with tf.name_scope('cross_entropy_top'):
        cross_entropy_mean_top = tf.losses.sparse_softmax_cross_entropy(
            labels=ground_truth_input_top, logits=top_logits)

    tf.summary.scalar('cross_entropy', cross_entropy_mean)

    # todo: Adam optimizer
    with tf.name_scope('train'):
        # optimizer = tf.train.GradientDescentOptimizer(FLAGS.learning_rate)
        optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)
        train_step_sub = optimizer.minimize(cross_entropy_mean)
        train_step_top = optimizer.minimize(cross_entropy_mean_top)

        train_step = tf.group(train_step_sub, train_step_top)

    return (train_step, cross_entropy_mean, cross_entropy_mean_top, bottleneck_input, ground_truth_input,
            ground_truth_input_top, final_tensor, final_top_tensor)


def add_evaluation_step(result_tensor, ground_truth_tensor, layer_name):
    """Inserts the operations we need to evaluate the accuracy of our results.

    Args:
      result_tensor: The new final node that produces results.
      ground_truth_tensor: The node we feed ground truth data
      into.

    Returns:
      Tuple of (evaluation step, prediction).
    """
    with tf.name_scope(layer_name):
        with tf.name_scope('correct_prediction'):
            prediction = tf.argmax(result_tensor, 1)
            correct_prediction = tf.equal(prediction, ground_truth_tensor)
        with tf.name_scope('accuracy'):
            evaluation_step = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar(layer_name, evaluation_step)
    return evaluation_step, prediction


def run_final_eval(train_session, module_spec, module_name, class_count, top_class_count, image_lists, top_image_labels,
                   jpeg_data_tensor, decoded_image_tensor,
                   resized_image_tensor, bottleneck_tensor):
    """Runs a final evaluation on an eval graph using the test data set.

    Args:
      train_session: Session for the train graph with the tensors below.
      module_spec: The hub.ModuleSpec for the image module being used.
      class_count: Number of classes
      image_lists: OrderedDict of training images for each label.
      jpeg_data_tensor: The layer to feed jpeg image data into.
      decoded_image_tensor: The output of decoding and resizing the image.
      resized_image_tensor: The input node of the recognition graph.
      bottleneck_tensor: The bottleneck output layer of the CNN graph.
    """
    test_bottlenecks, \
    test_ground_truth, test_ground_truth_top_label, \
    test_filenames = (
        get_random_cached_bottlenecks(train_session, image_lists, top_image_labels,
                                      FLAGS.test_batch_size,
                                      'testing', FLAGS.bottleneck_dir,
                                      FLAGS.image_dir, jpeg_data_tensor,
                                      decoded_image_tensor, resized_image_tensor,
                                      bottleneck_tensor, module_name, FLAGS.module_trainable))

    (eval_session, _, bottleneck_input, ground_truth_input, ground_truth_input_top, evaluation_step,
     evaluation_top_step, prediction, prediction_top) = build_eval_session(module_spec, class_count, top_class_count)
    test_accuracy, test_accuracy_top, predictions = eval_session.run(
        [evaluation_step, evaluation_top_step, prediction],
        feed_dict={
            bottleneck_input: test_bottlenecks,
            ground_truth_input: test_ground_truth,
            ground_truth_input_top: test_ground_truth_top_label
        })
    tf.logging.info('Final test accuracy = %.1f%% (N=%d)' %
                    (test_accuracy * 100, len(test_bottlenecks)))
    tf.logging.info('Final test top-accuracy = %.1f%% (N=%d)' %
                    (test_accuracy_top * 100, len(test_bottlenecks)))

    if FLAGS.print_misclassified_test_images:
        tf.logging.info('=== MISCLASSIFIED TEST IMAGES ===')
        for i, test_filename in enumerate(test_filenames):
            if predictions[i] != test_ground_truth[i]:
                tf.logging.info('%70s  %s' % (test_filename,
                                              list(image_lists.keys())[predictions[i]]))


def build_eval_session(module_spec, class_count, top_class_count):
    """Builds an restored eval session without train operations for exporting.

    Args:
      module_spec: The hub.ModuleSpec for the image module being used.
      class_count: Number of classes

    Returns:
      Eval session containing the restored eval graph.
      The bottleneck input, ground truth, eval step, and prediction tensors.
    """
    # If quantized, we need to create the correct eval graph for exporting.
    eval_graph, bottleneck_tensor, resized_input_tensor, wants_quantization = (
        create_module_graph(module_spec))

    eval_sess = tf.Session(graph=eval_graph)
    with eval_graph.as_default():
        # Add the new layer for exporting.
        (_, _, _, bottleneck_input,
         ground_truth_input, ground_truth_input_top, final_tensor, final_top_tensor) = add_final_retrain_ops(
            class_count, top_class_count, FLAGS.final_tensor_name,
            FLAGS.final_top_tensor_name, bottleneck_tensor,
            wants_quantization, is_training=False)

        # Now we need to restore the values from the training graph to the eval
        # graph.
        tf.train.Saver().restore(eval_sess, CHECKPOINT_NAME)

        evaluation_step, prediction = add_evaluation_step(final_tensor,
                                                          ground_truth_input,
                                                          'accuracy')
        evaluation_top_step, prediction_top = add_evaluation_step(final_top_tensor,
                                                                  ground_truth_input_top,
                                                                  'accuracy_top')

        if FLAGS.module_trainable:
            bottleneck_input = resized_input_tensor

    return (eval_sess, resized_input_tensor, bottleneck_input, ground_truth_input, ground_truth_input_top,
            evaluation_step, evaluation_top_step, prediction, prediction_top)


def save_graph_to_file(graph_file_name, module_spec, class_count, top_class_count):
    """Saves an graph to file, creating a valid quantized one if necessary."""
    sess, _, _, _, _, _, _, _, _ = build_eval_session(module_spec, class_count, top_class_count)
    graph = sess.graph

    output_graph_def = tf.graph_util.convert_variables_to_constants(
        sess, graph.as_graph_def(), [FLAGS.final_tensor_name])

    with tf.gfile.GFile(graph_file_name, 'wb') as f:
        f.write(output_graph_def.SerializeToString())


def prepare_file_system():
    ensure_dir_exists(FLAGS.output_path)

    # Set up the directory we'll write summaries to for TensorBoard
    if tf.gfile.Exists(FLAGS.summaries_dir):
        tf.gfile.DeleteRecursively(FLAGS.summaries_dir)
    tf.gfile.MakeDirs(FLAGS.summaries_dir)
    if FLAGS.intermediate_store_frequency > 0:
        ensure_dir_exists(FLAGS.intermediate_output_graphs_dir)
    return


def add_jpeg_decoding(module_spec):
    """Adds operations that perform JPEG decoding and resizing to the graph..

    Args:
      module_spec: The hub.ModuleSpec for the image module being used.

    Returns:
      Tensors for the node to feed JPEG data into, and the output of the
        preprocessing steps.
    """
    input_height, input_width = hub.get_expected_image_size(module_spec[0])
    input_depth = hub.get_num_image_channels(module_spec[0])
    jpeg_data = tf.placeholder(tf.string, name='DecodeJPGInput')
    decoded_image = tf.image.decode_jpeg(jpeg_data, channels=input_depth)
    # Convert from full range of uint8 to range [0,1] of float32.
    decoded_image_as_float = tf.image.convert_image_dtype(decoded_image,
                                                          tf.float32)
    decoded_image_4d = tf.expand_dims(decoded_image_as_float, 0)
    resize_shape = tf.stack([input_height, input_width])
    resize_shape_as_int = tf.cast(resize_shape, dtype=tf.int32)
    resized_image = tf.image.resize_bilinear(decoded_image_4d,
                                             resize_shape_as_int)
    return jpeg_data, resized_image


def export_model(module_spec, class_count, top_class_count, saved_model_dir):
    """Exports model for serving.

    Args:
      module_spec: The hub.ModuleSpec for the image module being used.
      class_count: The number of classes.
      saved_model_dir: Directory in which to save exported model and variables.
    """
    # The SavedModel should hold the eval graph.
    sess, in_image, _, _, _, _, _, _, _ = build_eval_session(module_spec, class_count, top_class_count)
    with sess.graph.as_default() as graph:
        tf.saved_model.simple_save(
            sess,
            saved_model_dir,
            inputs={'image': in_image},
            outputs={'prediction': graph.get_tensor_by_name('final_result:0')},
            legacy_init_op=tf.group(tf.tables_initializer(), name='legacy_init_op')
        )


def logging_level_verbosity(logging_verbosity):
    """Converts logging_level into TensorFlow logging verbosity value

    Args:
      logging_level: String value representing logging level: 'DEBUG', 'INFO',
      'WARN', 'ERROR', 'FATAL'
    """
    name_to_level = {
        'FATAL': tf.logging.FATAL,
        'ERROR': tf.logging.ERROR,
        'WARN': tf.logging.WARN,
        'INFO': tf.logging.INFO,
        'DEBUG': tf.logging.DEBUG
    }

    try:
        return name_to_level[logging_verbosity]
    except Exception as e:
        raise RuntimeError('Not supported logs verbosity (%s). Use one of %s.' %
                           (str(e), list(name_to_level)))


def logging_to_file():
    import logging
    log = logging.getLogger('tensorflow')

    # create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # create file handler which logs even debug messages
    fh = logging.FileHandler(FLAGS.output_path + '/train.log')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    log.addHandler(fh)


def load_image_lists():
    with open(FLAGS.image_lists_dir, 'r') as rf:
        image_lists = json.load(rf, object_hook=collections.OrderedDict)
    return image_lists


def set_random_seed():
    random.seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)
    tf.set_random_seed(FLAGS.seed)


def extract_top_label(image_label):
    return image_label.split(' ')[0]


def read_top_labels(image_lists):
    label_names = image_lists.keys()
    top_label_names = set(map(
        lambda x: extract_top_label(x),
        label_names
    ))
    return list(top_label_names)


def main(_):
    # Prepare necessary directories that can be used during training
    prepare_file_system()
    set_random_seed()

    # Needed to make sure the logging output is visible.
    # See https://github.com/tensorflow/tensorflow/issues/3047
    logging_to_file()
    logging_verbosity = logging_level_verbosity(FLAGS.logging_verbosity)
    tf.logging.set_verbosity(logging_verbosity)

    if not FLAGS.image_dir:
        tf.logging.error('Must set flag --image_dir.')
        return -1

    image_lists = load_image_lists()
    top_image_labels = read_top_labels(image_lists)
    top_class_count = len(top_image_labels)
    class_count = len(image_lists.keys())
    if class_count == 0:
        tf.logging.error('No valid folders of images found at ' + FLAGS.image_dir)
        return -1
    if class_count == 1:
        tf.logging.error('Only one valid folder of images found at ' +
                         FLAGS.image_dir +
                         ' - multiple classes are needed for classification.')
        return -1

    # See if the command-line flags mean we're applying any distortions.
    do_distort_images = should_distort_images(
        FLAGS.flip_left_right, FLAGS.random_crop, FLAGS.random_scale,
        FLAGS.random_brightness)

    # Set up the pre-trained graph.
    module_name = reduce(lambda x, y: x + '_' + y, FLAGS.tfhub_module)
    module_spec = [hub.load_module_spec(ele) for ele in FLAGS.tfhub_module]
    graph, bottleneck_tensor, resized_image_tensor, wants_quantization = (
        create_module_graph(module_spec))

    # Add the new layer that we'll be training.
    with graph.as_default():
        (train_step, cross_entropy, cross_entropy_top, bottleneck_input,
         ground_truth_input, ground_truth_input_top, final_tensor, final_top_tensor) = add_final_retrain_ops(
            class_count, top_class_count, FLAGS.final_tensor_name, FLAGS.final_top_tensor_name, bottleneck_tensor,
            wants_quantization, is_training=True)

        if FLAGS.module_trainable:
            bottleneck_input = resized_image_tensor

    # growth GPU config
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.Session(config=config, graph=graph) as sess:
        # Initialize all weights: for the module to their pretrained values,
        # and for the newly added retraining layer to random initial values.
        init = tf.global_variables_initializer()
        sess.run(init)

        # Set up the image decoding sub-graph.
        jpeg_data_tensor, decoded_image_tensor = add_jpeg_decoding(module_spec)

        if do_distort_images:
            # We will be applying distortions, so set up the operations we'll need.
            (distorted_jpeg_data_tensor,
             distorted_image_tensor) = add_input_distortions(
                FLAGS.flip_left_right, FLAGS.random_crop, FLAGS.random_scale,
                FLAGS.random_brightness, module_spec)
        elif not FLAGS.module_trainable:
            # We'll make sure we've calculated the 'bottleneck' image summaries and
            # cached them on disk.
            cache_bottlenecks(sess, image_lists, FLAGS.image_dir,
                              FLAGS.bottleneck_dir, jpeg_data_tensor,
                              decoded_image_tensor, resized_image_tensor,
                              bottleneck_tensor, module_name)

        # Create the operations we need to evaluate the accuracy of our new layer.
        evaluation_step, _ = add_evaluation_step(final_tensor, ground_truth_input, 'accuracy')
        evaluation_top_step, _ = add_evaluation_step(final_top_tensor, ground_truth_input_top, 'accuracy_top')

        # Merge all the summaries and write them out to the summaries_dir
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(FLAGS.summaries_dir + 'train',
                                             sess.graph)

        validation_writer = tf.summary.FileWriter(
            FLAGS.summaries_dir + 'validation')

        # Create a train saver that is used to restore values into an eval graph
        # when exporting models.
        train_saver = tf.train.Saver()

        # Run the training for as many cycles as requested on the command line.
        for i in range(FLAGS.how_many_training_steps):
            # Get a batch of input bottleneck values, either calculated fresh every
            # time with distortions applied, or from the cache stored on disk.
            if do_distort_images:
                (train_bottlenecks,
                 train_ground_truth) = get_random_distorted_bottlenecks(
                    sess, image_lists, FLAGS.train_batch_size, 'training',
                    FLAGS.image_dir, distorted_jpeg_data_tensor,
                    distorted_image_tensor, resized_image_tensor, bottleneck_tensor)
            else:
                (train_bottlenecks,
                 train_ground_truth,
                 train_ground_truth_top_label, _) = get_random_cached_bottlenecks(
                    sess, image_lists, top_image_labels, FLAGS.train_batch_size, 'training',
                    FLAGS.bottleneck_dir, FLAGS.image_dir, jpeg_data_tensor,
                    decoded_image_tensor, resized_image_tensor, bottleneck_tensor,
                    module_name, FLAGS.module_trainable)

            # Feed the bottlenecks and ground truth into the graph, and run a training
            # step. Capture training summaries for TensorBoard with the `merged` op.
            train_summary, _ = sess.run(
                [merged, train_step],
                feed_dict={bottleneck_input: train_bottlenecks,
                           ground_truth_input: train_ground_truth,
                           ground_truth_input_top: train_ground_truth_top_label})
            train_writer.add_summary(train_summary, i)

            # Every so often, print out how well the graph is training.
            is_last_step = (i + 1 == FLAGS.how_many_training_steps)
            if (i % FLAGS.eval_step_interval) == 0 or is_last_step:
                train_accuracy, train_top_accuracy, cross_entropy_value, cross_entropy_value_top = sess.run(
                    [evaluation_step, evaluation_top_step, cross_entropy, cross_entropy_top],
                    feed_dict={bottleneck_input: train_bottlenecks,
                               ground_truth_input: train_ground_truth,
                               ground_truth_input_top: train_ground_truth_top_label})
                tf.logging.info('%s: Step %d: Train accuracy = %.1f%%, Train-Top accuracy = %.1f%%' %
                                (datetime.now(), i, train_accuracy * 100, train_top_accuracy * 100))
                tf.logging.info('%s: Step %d: Cross entropy = %f, Cross-Top entropy = %f' %
                                (datetime.now(), i, cross_entropy_value, cross_entropy_value_top))
                # TODO: Make this use an eval graph, to avoid quantization
                # moving averages being updated by the validation set, though in
                # practice this makes a negligable difference.
                validation_bottlenecks, \
                validation_ground_truth, \
                validation_ground_truth_top_label, _ = (
                    get_random_cached_bottlenecks(
                        sess, image_lists, top_image_labels, FLAGS.validation_batch_size, 'validation',
                        FLAGS.bottleneck_dir, FLAGS.image_dir, jpeg_data_tensor,
                        decoded_image_tensor, resized_image_tensor, bottleneck_tensor,
                        module_name, FLAGS.module_trainable))
                # Run a validation step and capture training summaries for TensorBoard
                # with the `merged` op.
                validation_summary, validation_accuracy, validation_accuracy_top = sess.run(
                    [merged, evaluation_step, evaluation_top_step],
                    feed_dict={bottleneck_input: validation_bottlenecks,
                               ground_truth_input: validation_ground_truth,
                               ground_truth_input_top: validation_ground_truth_top_label})
                validation_writer.add_summary(validation_summary, i)
                tf.logging.info('%s: Step %d: Validation accuracy = %.1f%%, Validation-Top accuracy = %.1f%%, (N=%d)' %
                                (datetime.now(), i, validation_accuracy * 100, validation_accuracy_top * 100,
                                 len(validation_bottlenecks)))

            # Store intermediate results
            intermediate_frequency = FLAGS.intermediate_store_frequency

            if (intermediate_frequency > 0 and (i % intermediate_frequency == 0)
                    and i > 0):
                # If we want to do an intermediate save, save a checkpoint of the train
                # graph, to restore into the eval graph.
                train_saver.save(sess, CHECKPOINT_NAME)
                intermediate_file_name = (FLAGS.intermediate_output_graphs_dir +
                                          'intermediate_' + str(i) + '.pb')
                tf.logging.info('Save intermediate result to : ' +
                                intermediate_file_name)
                save_graph_to_file(intermediate_file_name, module_spec,
                                   class_count, top_class_count)

        # After training is complete, force one last save of the train checkpoint.
        train_saver.save(sess, CHECKPOINT_NAME)

        # We've completed all our training, so run a final test evaluation on
        # some new images we haven't used before.
        run_final_eval(sess, module_spec, module_name, class_count, top_class_count, image_lists,
                       top_image_labels, jpeg_data_tensor, decoded_image_tensor, resized_image_tensor,
                       bottleneck_tensor)

        # Write out the trained graph and labels with the weights stored as
        # constants.
        tf.logging.info('Save final result to : ' + FLAGS.output_graph)
        if wants_quantization:
            tf.logging.info('The model is instrumented for quantization with TF-Lite')
        save_graph_to_file(FLAGS.output_graph, module_spec, class_count, top_class_count)
        with tf.gfile.GFile(FLAGS.output_labels, 'w') as f:
            f.write('\n'.join(image_lists.keys()) + '\n')
        with tf.gfile.GFile(FLAGS.output_top_labels, 'w') as f:
            f.write('\n'.join(top_image_labels) + '\n')

        if FLAGS.saved_model_dir:
            export_model(module_spec, class_count, top_class_count, FLAGS.saved_model_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--seed',
        type=int,
        default=123,
        help='Set random seed.'
    )
    parser.add_argument(
        '--image_dir',
        type=str,
        default='',
        help='Path to folders of labeled images.'
    )
    parser.add_argument(
        '--intermediate_output_graphs_dir',
        type=str,
        default='/tmp/intermediate_graph/',
        help='Where to save the intermediate graphs.'
    )
    parser.add_argument(
        '--intermediate_store_frequency',
        type=int,
        default=0,
        help="""\
         How many steps to store intermediate graph. If "0" then will not
         store.\
      """
    )
    parser.add_argument(
        '--output_path',
        type=str,
        default='outputs/default',
        help='Where to save the output results.'
    )
    parser.add_argument(
        '--image_lists_dir',
        type=str,
        default='data/weed_image_lists_oversample.json',
        help='Where to load image list.'
    )
    parser.add_argument(
        '--how_many_training_steps',
        type=int,
        default=4000,
        help='How many training steps to run before ending.'
    )
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=0.001,
        help='How large a learning rate to use when training.'
    )
    parser.add_argument(
        '--eval_step_interval',
        type=int,
        default=10,
        help='How often to evaluate the training results.'
    )
    parser.add_argument(
        '--train_batch_size',
        type=int,
        default=100,
        help='How many images to train on at a time.'
    )
    parser.add_argument(
        '--test_batch_size',
        type=int,
        default=-1,
        help="""\
      How many images to test on. This test set is only used once, to evaluate
      the final accuracy of the model after training completes.
      A value of -1 causes the entire test set to be used, which leads to more
      stable results across runs.\
      """
    )
    parser.add_argument(
        '--validation_batch_size',
        type=int,
        default=100,
        help="""\
      How many images to use in an evaluation batch. This validation set is
      used much more often than the test set, and is an early indicator of how
      accurate the model is during training.
      A value of -1 causes the entire validation set to be used, which leads to
      more stable results across training iterations, but may be slower on large
      training sets.\
      """
    )
    parser.add_argument(
        '--print_misclassified_test_images',
        default=False,
        help="""\
      Whether to print out a list of all misclassified test images.\
      """,
        action='store_true'
    )
    parser.add_argument(
        '--final_tensor_name',
        type=str,
        default='final_result',
        help="""\
      The name of the output classification layer in the retrained graph.\
      """
    )
    parser.add_argument(
        '--final_top_tensor_name',
        type=str,
        default='final_result_top',
        help="""\
          The name of the output classification layer in the retrained graph.\
          """
    )
    parser.add_argument(
        '--flip_left_right',
        default=False,
        help="""\
      Whether to randomly flip half of the training images horizontally.\
      """,
        action='store_true'
    )
    parser.add_argument(
        '--random_crop',
        type=int,
        default=0,
        help="""\
      A percentage determining how much of a margin to randomly crop off the
      training images.\
      """
    )
    parser.add_argument(
        '--random_scale',
        type=int,
        default=0,
        help="""\
      A percentage determining how much to randomly scale up the size of the
      training images by.\
      """
    )
    parser.add_argument(
        '--random_brightness',
        type=int,
        default=0,
        help="""\
      A percentage determining how much to randomly multiply the training image
      input pixels up or down by.\
      """
    )
    parser.add_argument(
        '--tfhub_module',
        type=str,
        nargs='+',
        default=['https://tfhub.dev/google/imagenet/inception_v3/feature_vector/3'],
        help="""\
      Which TensorFlow Hub module to use. For more options,
      search https://tfhub.dev for image feature vector modules.\
      """)
    parser.add_argument(
        '--module_trainable',
        type=int,
        default=0
    )
    parser.add_argument(
        '--logging_verbosity',
        type=str,
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARN', 'ERROR', 'FATAL'],
        help='How much logging output should be produced.')
    FLAGS, unparsed = parser.parse_known_args()

    setattr(FLAGS, 'output_labels', FLAGS.output_path + '/output_labels.txt')
    setattr(FLAGS, 'output_top_labels', FLAGS.output_path + '/output_top_labels.txt')
    setattr(FLAGS, 'output_graph', FLAGS.output_path + '/frozen_graph.pb')
    setattr(FLAGS, 'summaries_dir', FLAGS.output_path + '/retrain_logs/')
    setattr(FLAGS, 'bottleneck_dir', 'outputs/bottleneck/')
    setattr(FLAGS, 'saved_model_dir', FLAGS.output_path + '/saved_model/')

    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
