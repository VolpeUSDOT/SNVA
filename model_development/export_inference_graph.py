# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
r"""Saves out a GraphDef containing the architecture of the model.

To use it, run something like this, with a model name defined by slim:

bazel build tensorflow_models/research/slim:export_inference_graph
bazel-bin/tensorflow_models/research/slim/export_inference_graph \
--model_name=inception_v3 --output_file=/tmp/inception_v3_inf_graph.pb

If you then want to use the resulting model with your own or pretrained
checkpoints as part of a mobile model, you can run freeze_graph to get a graph
def with the variables inlined as constants using:

bazel build tensorflow/python/tools:freeze_graph
bazel-bin/tensorflow/python/tools/freeze_graph \
--input_graph=/tmp/inception_v3_inf_graph.pb \
--input_checkpoint=/tmp/checkpoints/inception_v3.ckpt \
--input_binary=true --output_graph=/tmp/frozen_inception_v3.pb \
--output_node_names=InceptionV3/Predictions/Reshape_1

The output node names will vary depending on the model, but you can inspect and
estimate them using the summarize_graph tool:

bazel build tensorflow/tools/graph_transforms:summarize_graph
bazel-bin/tensorflow/tools/graph_transforms/summarize_graph \
--in_graph=/tmp/inception_v3_inf_graph.pb

To run the resulting graph in C++, you can look at the label_image sample code:

bazel build tensorflow/examples/label_image:label_image
bazel-bin/tensorflow/examples/label_image/label_image \
--image=${HOME}/Pictures/flowers.jpg \
--input_layer=input \
--output_layer=InceptionV3/Predictions/Reshape_1 \
--graph=/tmp/frozen_inception_v3.pb \
--labels=/tmp/imagenet_slim_labels.txt \
--input_mean=0 \
--input_std=255

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow.python.platform import gfile
from nets import nets_factory
from tensorflow.python.tools.freeze_graph import freeze_graph_with_def_protos
from tensorflow.python.tools.optimize_for_inference_lib import optimize_for_inference

slim = tf.contrib.slim

tf.app.flags.DEFINE_string(
  'checkpoint_path', None,
  'A directory containing checkpoints or an absolute path to a checkpoint file.')

tf.app.flags.DEFINE_string(
  'model_name', None, 'The name of the architecture to save.')

tf.app.flags.DEFINE_integer(
  'image_size', None,
  'The image size to use, otherwise use the model default_image_size.')

tf.app.flags.DEFINE_integer(
  'num_classes', 2,
  'The number of classes that the model was trained to predict.')

tf.app.flags.DEFINE_string(
  'output_file', '', 'Where to save the resulting file to.')

tf.app.flags.DEFINE_string(
  'input_node_name', 'images',
  'Name of the tensor through which images are input to the network')

tf.app.flags.DEFINE_string(
  'output_node_name', None,
  'Name of the tensor through which images are input to the network')

FLAGS = tf.app.flags.FLAGS


def main(_):
  if not FLAGS.output_file:
    raise ValueError('You must supply the path to save to with --output_file')

  tf.logging.set_verbosity(tf.logging.INFO)

  with tf.Graph().as_default() as graph:
    network_fn = nets_factory.get_network_fn(
      FLAGS.model_name, num_classes=FLAGS.num_classes, is_training=False)

    image_size = FLAGS.image_size or network_fn.default_image_size

    images = tf.placeholder(tf.float32, name=FLAGS.input_node_name,
                            shape=[None, image_size, image_size, 3])

    network_fn(images)

    graph_def = graph.as_graph_def()

    saver_def = tf.train.Saver().as_saver_def()

    if tf.gfile.IsDirectory(FLAGS.checkpoint_path):
      checkpoint_path = tf.train.latest_checkpoint(FLAGS.checkpoint_path)
    else:
      checkpoint_path = FLAGS.checkpoint_path

    frozen_graph_def = freeze_graph_with_def_protos(
      graph_def, saver_def, checkpoint_path, FLAGS.output_node_name,
      restore_op_name=None, filename_tensor_name=None,
      output_graph=None, clear_devices=True, initializer_nodes=None)

    optimized_graph_def = optimize_for_inference(
      frozen_graph_def, [FLAGS.input_node_name],
      [FLAGS.output_node_name], tf.float32.as_datatype_enum)

    with gfile.GFile(FLAGS.output_file, 'wb') as output_file:
      output_file.write(optimized_graph_def.SerializeToString())


if __name__ == '__main__':
  tf.app.run()
