# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
r"""Downloads and converts a particular dataset.

Usage:
```shell

$ python convert_data_set.py \
    --dataset_name=mnist \
    --dataset_dir=/tmp/mnist

$ python convert_data_set.py \
    --dataset_name=cifar10 \
    --dataset_dir=/tmp/cifar10

$ python convert_data_set.py \
    --dataset_name=flowers \
    --dataset_dir=/tmp/flowers
```
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import signal
import sys

import tensorflow as tf

from datasets import dataset

path = os.path
FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string(
  'dataset_name',
  None,
  'The name of the dataset to convert, one of "cifar10", "construction", "railroad", "mnist".')

tf.app.flags.DEFINE_string(
  'dataset_dir',
  None,
  'The directory where the output TFRecords and temporary files are saved.')

tf.app.flags.DEFINE_float(
    'gpu_memory_fraction', 0.9,
    'The ratio of total memory across all available GPUs to use with this process. '
    'Defaults to a suggested max of 0.9.')

tf.app.flags.DEFINE_string(
    'gpu_device_num', '0',
    'The device number of a single GPU to use for evaluation on a multi-GPU system. '
    'Defaults to zero.')

tf.app.flags.DEFINE_boolean(
    'cpu_only', False,
    'Explicitly assign all evaluation ops to the CPU on a GPU-enabled system. '
    'Defaults to False.')


def interrupt_handler(signal_number, _):
    tf.logging.info(
        'Received interrupt signal (%d). Unsetting CUDA_VISIBLE_DEVICES environment variable.', signal_number)
    os.unsetenv('CUDA_VISIBLE_DEVICES')
    sys.exit(0)


def main(_):
  if FLAGS.dataset_name is None:
    raise ValueError('You must specify the dataset name using --dataset_name')
  if FLAGS.dataset_dir is None:
    raise ValueError('You must specify the dataset directory using --dataset_dir')

  tf.logging.set_verbosity(tf.logging.INFO)

  if FLAGS.cpu_only:
      device_name = '/cpu:0'

      session_config = None

      tf.logging.info('Setting CUDA_VISIBLE_DEVICES environment variable to None.')
      os.putenv('CUDA_VISIBLE_DEVICES', '')
  else:
      device_name = '/gpu:' + FLAGS.gpu_device_num

      gpu_options = tf.GPUOptions(allow_growth=True,
                                  per_process_gpu_memory_fraction=FLAGS.gpu_memory_fraction)
      session_config = tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options)

      tf.logging.info('Setting CUDA_VISIBLE_DEVICES environment variable to {}.'.format(
                      FLAGS.gpu_device_num))
      os.putenv('CUDA_VISIBLE_DEVICES', FLAGS.gpu_device_num)

  signal.signal(signal.SIGINT, interrupt_handler)

  dataset.compute_statistics(FLAGS.dataset_dir, FLAGS.dataset_name, session_config, device_name)


if __name__ == '__main__':
  tf.app.run()
