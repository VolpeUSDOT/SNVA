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

$ python download_and_convert_data.py \
    --dataset_name=mnist \
    --dataset_dir=/tmp/mnist

$ python download_and_convert_data.py \
    --dataset_name=cifar10 \
    --dataset_dir=/tmp/cifar10

$ python download_and_convert_data.py \
    --dataset_name=flowers \
    --dataset_dir=/tmp/flowers
```
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from datasets import dataset
from os import path

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string(
    'dataset_name',
    None,
    'The name of the dataset to convert, one of "cifar10", "construction", "railroad", "mnist".')

tf.app.flags.DEFINE_string(
    'dataset_dir',
    None,
    'The directory where the output TFRecords and temporary files are saved.')

tf.app.flags.DEFINE_integer(
    'batch_size',
    32,
    'The name of the dataset to convert, one of "cifar10", "mnist", or something custom')

tf.app.flags.DEFINE_integer(
    'random_seed',
    1,
    'The random seed used to instantiate the pseudo-random number generator '
    'that shuffles the samples before creating TFRecord shards')

tf.app.flags.DEFINE_bool(
    'convert_standard_subsets',
    True,
    'If True, create TFRecords for the samples in the training, dev, and '
    'test sub-directories of datasets_root_dir.'
)

tf.app.flags.DEFINE_bool(
    'convert_eval_subset',
    False,
    'If True, create TFRecords for the samples in the '
    'eval sub-directory of datasets_root_dir.'
)

DATASET_NAMES = ['cifar10', 'flowers', 'mnist']


def main(_):
    if not FLAGS.dataset_name:
        raise ValueError('You must specify the dataset name using --dataset_name')
    if not FLAGS.dataset_dir:
        raise ValueError('You must specify the dataset directory using --dataset_dir')

    subset_names = []

    if FLAGS.convert_standard_subsets:
        subset_names.extend(['training', 'dev', 'test'])

    if FLAGS.convert_eval_subset:
        subset_names.append('eval')

    if len(subset_names) > 0:
        if path.exists(path.join(FLAGS.dataset_dir, FLAGS.dataset_name)) or FLAGS.dataset_name in DATASET_NAMES:
            dataset.convert(FLAGS.dataset_dir, FLAGS.dataset_name, FLAGS.batch_size, FLAGS.random_seed, subset_names)
        else:
            raise ValueError(
                'dataset_name [%s] was not recognized.' % FLAGS.dataset_name)
    else:
        raise ValueError(
            'Dataset conversion aborted because no subsets were specified.'
        )


if __name__ == '__main__':
    tf.app.run()
