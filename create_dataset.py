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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from datasets import dataset

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string(
  'data_source_dir', None,
  'Path to the parent of the class-specific subfolders containing source images.'
)
tf.app.flags.DEFINE_string(
  'data_set_dir', None, 'Path to the directory where data subsets will be created.'
)
tf.app.flags.DEFINE_string(
  'positive_class_name', None, ''
)
tf.app.flags.DEFINE_string(
  'negative_class_name', None, ''
)
tf.app.flags.DEFINE_bool(
  'create_standard_subsets', True,
  'If True, create training, test and dev subsets.'
)
tf.app.flags.DEFINE_bool(
  'create_eval_subset', False,
  'If True, create a subset for evaluation that contains all samples.'
)
tf.app.flags.DEFINE_float(
  'training_ratio', 0.7,
  'The per-class percentage of data to use in training subset construction.'
)
tf.app.flags.DEFINE_float(
  'dev_ratio', 0.2,
  'The per-class percentage of data to use in dev subset construction.'
)
tf.app.flags.DEFINE_integer(
  'random_seed', 1,
  'The random seed used to instantiate the pseudo-random number generator '
  'that shuffles the samples before creating TFRecord shards, and that balances'
  'class-wise unbalanced data sets via subsampling.'
)
tf.app.flags.DEFINE_bool(
  'balance_subsets', False,
  'If True, make sure that each class has an equal number of samples. Balancing is performed by '
  'stratifying the sampling over the collection of data \'sub\' sources. In turn, the '
  'over-representation of a class in one sub source will not be compensated for by the '
  'over-representation another class in some other sub source.'
)


def main(_):
  if FLAGS.data_source_dir is None:
    raise ValueError('You must specify the data source directory path using --data_source_dir')
  if not tf.gfile.Exists(FLAGS.data_source_dir):
    raise ValueError('The specified data source directory path does not exist')
  if FLAGS.data_set_dir is None:
    raise ValueError('You must specify the data set directory path using --data_set_dir')
  if FLAGS.positive_class_name is None:
    raise ValueError('You must specify the name of the positive class using --positive_class_name')
  if FLAGS.negative_class_name is None:
    raise ValueError('You must specify the name of the negative class using --negative_class_name')
  if (FLAGS.training_ratio + FLAGS.dev_ratio) > (0.9 + 1e-7):
    raise ValueError('At least 10% of data should be reserved for the test set')

  dataset.create(class_dir_names=[FLAGS.positive_class_name, FLAGS.negative_class_name],
                 create_standard_subsets=FLAGS.create_standard_subsets,
                 create_eval_subset=FLAGS.create_eval_subset,
                 data_source_dir=FLAGS.data_source_dir,
                 data_set_dir=FLAGS.data_set_dir,
                 training_ratio=FLAGS.training_ratio,
                 dev_ratio=FLAGS.dev_ratio,
                 random_seed=FLAGS.random_seed,
                 balance_subsets=FLAGS.balance_subsets
                 )


if __name__ == '__main__':
  tf.app.run()
