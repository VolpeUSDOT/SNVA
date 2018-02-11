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
"""Provides utilities to preprocess images for the Inception networks."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from preprocessing import inception_preprocessing


# assumes std_dev to be > 0
def standardize(image, height, width, statistics, is_training=False, fast_mode=True):
  if image.dtype != tf.float32:
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
  tf.summary.image('original_image', tf.expand_dims(image, 0))

  if is_training:
    # Randomly distort the colors. There are 4 ways to do it.
    image = inception_preprocessing.apply_with_random_selector(
      image,
      lambda x, ordering: inception_preprocessing.distort_color(x, ordering, fast_mode),
      num_cases=4)
    tf.summary.image('color_distorted_image', tf.expand_dims(image, 0))

  # normalize image - statistics were computed on float32 images scaled to the range [0, 1]
  population_per_channel_mean = tf.constant(
    [[[statistics['r_mean'], statistics['g_mean'], statistics['b_mean']]]],
    dtype=tf.float32, shape=(3,))
  population_per_channel_std_dev = tf.constant(
    [[[statistics['r_std_dev'], statistics['g_std_dev'], statistics['b_std_dev']]]],
    dtype=tf.float32, shape=(3,))
  # population_per_channel_max = tf.constant(
  #   [[[statistics['r_max'], statistics['g_max'], statistics['b_max']]]],
  #   dtype=tf.float32, shape=(3,))
  # population_per_channel_min = tf.constant(
  #   [[[statistics['r_min'], statistics['g_min'], statistics['b_min']]]],
  #   dtype=tf.float32, shape=(3,))

  image = tf.subtract(image, population_per_channel_mean)
  tf.summary.image('mean_subtracted_image', tf.expand_dims(image, 0))

  image = tf.divide(image, population_per_channel_std_dev)
  tf.summary.image('std_dev_scaled_image', tf.expand_dims(image, 0))

  # want want to bound the values of the unseen test data to the min and max values seen
  # in the training data. Option #1 is to clip any values outside of the training set range.
  # Option #2 is to rescale the data into the desired range. My hope is that option #2 will
  # serve as a 'cheat' that makes the test data 'look more like' the training data and make
  # the network more likely to assign the proper class.
  #
  # # Option 1
  # clipped_image = tf.clip_by_value(image, population_per_channel_min, population_per_channel_max)
  # tf.summary.image('clipped_image', tf.expand_dims(clipped_image, 0))

  # # Option 2
  # # First, rescale image to [0, 1]
  # sample_per_channel_min = tf.reduce_min(image, axis=[0, 1])
  # sample_per_channel_max = tf.reduce_max(image, axis=[0, 1])
  #
  # image = tf.divide(tf.subtract(image, sample_per_channel_min),
  #                   tf.subtract(sample_per_channel_max, sample_per_channel_min))
  # tf.summary.image('rescaled_to_zero_one_image', tf.expand_dims(image, 0))
  #
  # # Then, rescale image to [population_per_channel_min, population_per_channel_max]
  # image = tf.add(tf.multiply(image, tf.subtract(
  #   population_per_channel_max, population_per_channel_min)), population_per_channel_min)
  # tf.summary.image('rescaled_to_min_max_image', tf.expand_dims(image, 0))

  # Option 3
  # First, rescale image to [0, 1]
  sample_per_channel_min = tf.reduce_min(image, axis=[0, 1])
  sample_per_channel_max = tf.reduce_max(image, axis=[0, 1])

  image = tf.divide(tf.subtract(image, sample_per_channel_min),
                    tf.subtract(sample_per_channel_max, sample_per_channel_min))
  tf.summary.image('rescaled_to_zero_one_image', tf.expand_dims(image, 0))

  # Then, rescale image to [-1, 1]
  image = tf.multiply(tf.subtract(image, 0.5), 2.0)
  tf.summary.image('rescaled_to_negative_one_positive_one_image', tf.expand_dims(image, 0))

  if height and width:
    # Resize the image to the specified height and width.
    image = tf.expand_dims(image, 0)
    image = tf.image.resize_bilinear(image, [height, width], align_corners=False)
    image = tf.squeeze(image, [0])
    tf.summary.image('resized_image', tf.expand_dims(image, 0))

  image = tf.clip_by_value(image, -1.0, 1.0)
  tf.summary.image('clipped_image', tf.expand_dims(image, 0))

  return image


def preprocess_image(image, height, width,
                     is_training=False,
                     statistics=None):
  """Pre-process one image for training or evaluation.

  Args:
    image: 3-D Tensor [height, width, channels] with the image. If dtype is
      tf.float32 then the range should be [0, 1], otherwise it would converted
      to tf.float32 assuming that the range is [0, MAX], where MAX is largest
      positive representable number for int(8/16/32) data type (see
      `tf.image.convert_image_dtype` for details).
    height: integer, image expected height.
    width: integer, image expected width.
    is_training: Boolean. If true it would transform an image for train,
      otherwise it would transform it for evaluation.
    bbox: 3-D float Tensor of bounding boxes arranged [1, num_boxes, coords]
      where each coordinate is [0, 1) and the coordinates are arranged as
      [ymin, xmin, ymax, xmax].
    fast_mode: Optional boolean, if True avoids slower transformations.

  Returns:
    3-D float Tensor containing an appropriately scaled image

  Raises:
    ValueError: if user does not provide bounding box
  """
  return standardize(image, height, width, statistics, is_training)
