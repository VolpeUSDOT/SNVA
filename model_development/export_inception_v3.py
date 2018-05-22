import os
import signal
import sys
import time

import tensorflow as tf
from tensorflow.contrib import slim
from tensorflow.python.framework.graph_util import convert_variables_to_constants
from tensorflow.python.tools.optimize_for_inference_lib import optimize_for_inference

from nets import inception

tf.app.flags.DEFINE_string(
    'checkpoint_path', None,
    'A directory containing checkpoints or an absolute path to a checkpoint file.')

tf.app.flags.DEFINE_string(
    'protobuf_file', None,
    'The absolute path where the .pb file will be written.')

tf.app.flags.DEFINE_integer(
    'num_classes', 2,
    'The number of classes that the model was trained to predict.')

tf.app.flags.DEFINE_float(
    'gpu_memory_fraction', 0.9,
    'The ratio of total memory across all available GPUs to use with this process. '
    'Defaults to a suggested max of 0.9.')

tf.app.flags.DEFINE_integer(
    'gpu_device_num', 0,
    'The device number of a single GPU to use for evaluation on a multi-GPU system. '
    'Defaults to zero.')

tf.app.flags.DEFINE_boolean(
    'cpu_only', False,
    'Explicitly assign all evaluation ops to the CPU on a GPU-enabled system. '
    'Defaults to False.')

FLAGS = tf.app.flags.FLAGS


def interrupt_handler(signal_number, _):
    tf.logging.info('Received interrupt signal (%d). Unsetting CUDA_VISIBLE_DEVICES '
                    'environment variable.', signal_number)
    os.unsetenv('CUDA_VISIBLE_DEVICES')
    sys.exit(0)


def main(_):
  if not FLAGS.checkpoint_path:
    raise ValueError('You must supply the checkpoint directory or file path using '
                     '--checkpoint_path')
  if not FLAGS.protobuf_file:
    raise ValueError('You must supply an absolute path to the protobuf file to be created '
                     'using --protobuf_file')
  if not FLAGS.num_classes:
    raise ValueError('You must supply the number of classes using --num_classes')

  tf.logging.set_verbosity(tf.logging.INFO)

  if FLAGS.cpu_only:
    device_name = '/cpu:0'

    session_config = None

    tf.logging.info('Setting CUDA_VISIBLE_DEVICES environment variable to None.')
    os.putenv('CUDA_VISIBLE_DEVICES', '')
  else:
    device_name = '/gpu:' + str(FLAGS.gpu_device_num)

    gpu_options = tf.GPUOptions(allow_growth=True, per_process_gpu_memory_fraction=float(FLAGS.gpu_memory_fraction))
    session_config = tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options)

    tf.logging.info('Setting CUDA_VISIBLE_DEVICES environment variable to %d.', FLAGS.gpu_device_num)
    os.putenv('CUDA_VISIBLE_DEVICES', str(FLAGS.gpu_device_num))

  signal.signal(signal.SIGINT, interrupt_handler)

  with tf.Graph().as_default(), tf.device(device_name):
    # Inject placeholder into the graph
    images = tf.placeholder(tf.float32, name='images')
    image_size = tf.placeholder(tf.int32, name='image_size')
    images = tf.reshape(images, [-1, image_size, image_size, 3])

    # Load the inception network structure
    with slim.arg_scope(inception.inception_v3_arg_scope()):
      logits, _ = inception.inception_v3(images,
                                         num_classes=FLAGS.num_classes,
                                         is_training=False)
    # Apply softmax function to the logits (output of the last layer of the network)
    probabilities = tf.nn.softmax(logits)

    if tf.gfile.IsDirectory(FLAGS.checkpoint_path):
      model_path = tf.train.latest_checkpoint(FLAGS.checkpoint_path)
    else:
      model_path = FLAGS.checkpoint_path

    tf.logging.info('Exporting {} to {}'.format(FLAGS.checkpoint_path, FLAGS.protobuf_file))

    start_time = time.time()

    # Get the function that initializes the network structure (its variables) with
    # the trained values contained in the checkpoint
    init_fn = slim.assign_from_checkpoint_fn(model_path, slim.get_model_variables())

    with tf.Session(config=session_config) as sess:
      # Now call the initialization function within the session
      init_fn(sess)

      # Convert variables to constants and make sure the placeholder images is included
      # in the graph as well as the other neccesary tensors.
      constant_graph = convert_variables_to_constants(
        sess, sess.graph_def, ["images", "InceptionV3/Predictions/Reshape_1"])

      # Define the input and output layer properly
      optimized_constant_graph = optimize_for_inference(
        constant_graph, ["images"], ["InceptionV3/Predictions/Reshape_1"],
        tf.float32.as_datatype_enum)
      # Write the production ready graph to file.
      dir_name, base_name = os.path.split(FLAGS.protobuf_file)
      tf.train.write_graph(optimized_constant_graph, dir_name, base_name, as_text=False)

    end_time = time.time()
    print('Evaluation elapsed ' + str(end_time - start_time) + ' seconds')

if __name__ == '__main__':
  tf.app.run()

