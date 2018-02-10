# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
"""Generic evaluation script that evaluates a model using a given dataset.
    This variant reads the text file named checkpoint that is stored in
    the checkpoint_path directory and contains a list of all checkpoints
    that have been saved in that directory, and then iteratively evaluates
    each checkpoint. This is useful when training and evaluation are
    performed in sequence rather than in parallel (e.g. using
    eval_image_classifier_checkpoints.py"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import tensorflow as tf
from tensorflow.contrib import slim
from tensorflow.contrib import metrics
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from datasets import dataset_factory
from nets import nets_factory
from preprocessing import preprocessing_factory
import signal
import sys
import os

path = os.path

tf.app.flags.DEFINE_integer(
    'batch_size', 32, 'The number of samples in each batch.')

tf.app.flags.DEFINE_integer(
    'max_num_batches', None,
    'Max number of batches to evaluate by default use all.')

tf.app.flags.DEFINE_string(
    'master', '', 'The address of the TensorFlow master to use.')

tf.app.flags.DEFINE_string(
    'checkpoint_path', '/tmp/tfmodel/',
    'The directory where the model was written to or an absolute path to a '
    'checkpoint file.')

tf.app.flags.DEFINE_string(
    'eval_dir', '/tmp/tfmodel/', 'Directory where the results are saved to.')

tf.app.flags.DEFINE_integer(
    'num_preprocessing_threads', 4,
    'The number of threads used to create the batches.')

tf.app.flags.DEFINE_string(
    'dataset_name', 'imagenet', 'The name of the dataset to load.')

tf.app.flags.DEFINE_string(
    'dataset_split_name', 'test', 'The name of the train/test split.')

tf.app.flags.DEFINE_string(
    'dataset_dir', None, 'The directory where the dataset files are stored.')

tf.app.flags.DEFINE_integer(
    'labels_offset', 0,
    'An offset for the labels in the dataset. This flag is primarily used to '
    'evaluate the VGG and ResNet architectures which do not use a background '
    'class for the ImageNet dataset.')

tf.app.flags.DEFINE_string(
    'model_name', 'inception_v3', 'The name of the architecture to evaluate.')

tf.app.flags.DEFINE_string(
    'preprocessing_name', None, 'The name of the preprocessing to use. If left '
                                'as `None`, then the model_name flag is used.')

tf.app.flags.DEFINE_float(
    'moving_average_decay', None,
    'The decay to use for the moving average.'
    'If left as None, then moving averages are not used.')

tf.app.flags.DEFINE_integer(
    'eval_image_size', None, 'Eval image size')

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
    tf.logging.info(
        'Received interrupt signal (%d). Unsetting CUDA_VISIBLE_DEVICES environment variable.', signal_number)
    os.unsetenv('CUDA_VISIBLE_DEVICES')
    sys.exit(0)


def main(_):
    if not FLAGS.dataset_dir:
        raise ValueError('You must supply the dataset directory using --dataset_dir')

    if not tf.gfile.IsDirectory(FLAGS.checkpoint_path):
        raise ValueError('checkpoint_path must be a directory')

    checkpoint_text_file_path = path.join(FLAGS.checkpoint_path, 'checkpoint')

    if not tf.gfile.Exists(checkpoint_text_file_path):
        raise ValueError('checkpoint_path must contain a text file named checkpoint')

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
        tf_global_step = slim.get_or_create_global_step()

        ######################
        # Select the dataset #
        ######################
        dataset = dataset_factory.get_dataset(
            FLAGS.dataset_name, FLAGS.dataset_split_name, FLAGS.dataset_dir)

        ####################
        # Select the model #
        ####################
        network_fn = nets_factory.get_network_fn(
            FLAGS.model_name,
            num_classes=(dataset.num_classes - FLAGS.labels_offset),
            is_training=False)

        ##############################################################
        # Create a dataset provider that loads data from the dataset #
        ##############################################################
        provider = slim.dataset_data_provider.DatasetDataProvider(
            dataset,
            shuffle=False,
            common_queue_capacity=2 * FLAGS.batch_size,
            common_queue_min=FLAGS.batch_size)
        [image, label] = provider.get(['image', 'label'])
        label -= FLAGS.labels_offset

        #####################################
        # Select the preprocessing function #
        #####################################
        preprocessing_name = FLAGS.preprocessing_name or FLAGS.model_name
        image_preprocessing_fn = preprocessing_factory.get_preprocessing(
            preprocessing_name,
            is_training=False)

        eval_image_size = FLAGS.eval_image_size or network_fn.default_image_size

        image = image_preprocessing_fn(image, eval_image_size, eval_image_size)

        images, labels = tf.train.batch(
            [image, label],
            batch_size=FLAGS.batch_size,
            num_threads=FLAGS.num_preprocessing_threads,
            capacity=5 * FLAGS.batch_size)

        ####################
        # Define the model #
        ####################
        logits, _ = network_fn(images)

        if FLAGS.moving_average_decay:
            variable_averages = tf.train.ExponentialMovingAverage(
                FLAGS.moving_average_decay, tf_global_step)
            variables_to_restore = variable_averages.variables_to_restore(
                slim.get_model_variables())
            variables_to_restore[tf_global_step.op.name] = tf_global_step
        else:
            variables_to_restore = slim.get_variables_to_restore()

        predictions = tf.argmax(logits, 1)
        labels = tf.squeeze(labels)

        # Define the metrics:
        true_positives_name = 'True_Positives'
        true_negatives_name = 'True_Negatives'
        false_positives_name = 'False_Positives'
        false_negatives_name = 'False_Negatives'
        precision_name = 'Precision'
        recall_name = 'Recall'

        names_to_values, names_to_updates = slim.metrics.aggregate_metric_map({
            'Accuracy': metrics.streaming_accuracy(predictions, labels),
            precision_name: metrics.streaming_precision(predictions, labels),
            recall_name: metrics.streaming_recall(predictions, labels),
            true_positives_name: metrics.streaming_true_positives(predictions, labels),
            true_negatives_name: metrics.streaming_true_negatives(predictions, labels),
            false_positives_name: metrics.streaming_false_positives(predictions, labels),
            false_negatives_name: metrics.streaming_false_negatives(predictions, labels)
        })

        names_to_values['Total_Misclassifications'] = tf.add(names_to_values[false_positives_name],
                                                             names_to_values[false_negatives_name])

        def safe_divide(numerator, denominator):
            """Divides two values, returning 0 if the denominator is <= 0.
            Copied from the metric_ops.py protected member function.

            Args:
              numerator: A real `Tensor`.
              denominator: A real `Tensor`, with dtype matching `numerator`.
              name: Name for the returned op.

            Returns:
              0 if `denominator` <= 0, else `numerator` / `denominator`
            """
            return array_ops.where(
                math_ops.greater(denominator, 0),
                math_ops.truediv(numerator, denominator),
                0)

        def f_beta_measure(beta=1.0):
            beta_squared = math_ops.multiply(beta, beta)
            f_value = math_ops.multiply(
                math_ops.add(1.0, beta_squared),
                safe_divide(
                    math_ops.multiply(names_to_values[precision_name], names_to_values[recall_name]),
                    math_ops.add(
                        math_ops.multiply(beta_squared, names_to_values[precision_name]),
                        names_to_values[recall_name]
                    )
                )
            )
            return f_value

        names_to_values['F1'] = f_beta_measure()
        names_to_values['F2'] = f_beta_measure(2.0)
        names_to_values['F0.5'] = f_beta_measure(0.5)

        # Print the summaries to screen.
        for name, value in names_to_values.items():
            summary_name = 'eval/%s' % name
            op = tf.summary.scalar(summary_name, value, collections=[])
            op = tf.Print(op, [value], summary_name)
            tf.add_to_collection(tf.GraphKeys.SUMMARIES, op)

        # TODO(sguada) use num_epochs=1
        if FLAGS.max_num_batches:
            num_batches = FLAGS.max_num_batches
        else:
            # This ensures that we make a single pass over all of the data.
            num_batches = math.ceil(dataset.num_samples / float(FLAGS.batch_size))

        for line in tf.gfile.Open(checkpoint_text_file_path):
            split = line.split(':')
            if split[0] == 'all_model_checkpoint_paths':
                checkpoint_path = split[1].split('\"')[1]

                tf.logging.info('Evaluating %s' % checkpoint_path)

                slim.evaluation.evaluate_once(
                    master=FLAGS.master,
                    checkpoint_path=checkpoint_path,
                    logdir=FLAGS.eval_dir,
                    num_evals=num_batches,
                    eval_op=list(names_to_updates.values()),
                    variables_to_restore=variables_to_restore,
                    session_config=session_config)


if __name__ == '__main__':
    tf.app.run()
