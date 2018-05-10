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
r"""Converts washington_street data to TFRecords of TF-Example protos.

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import random
import shutil
import sys

import tensorflow as tf

from datasets import dataset_utils

path = os.path
slim = tf.contrib.slim

tf.logging.set_verbosity(tf.logging.INFO)


class ImageReader(object):
  """Helper class that provides TensorFlow image coding utilities."""

  def __init__(self):
    # Initializes function that decodes RGB JPEG data.
    self._decode_jpeg_data = tf.placeholder(dtype=tf.string)
    self._decode_jpeg = tf.image.decode_jpeg(self._decode_jpeg_data, channels=3)

  def read_image_dims(self, sess, image_data):
    image = self.decode_jpeg(sess, image_data)
    return image.shape[0], image.shape[1]

  def decode_jpeg(self, sess, image_data):
    image = sess.run(self._decode_jpeg,
                     feed_dict={self._decode_jpeg_data: image_data})
    assert len(image.shape) == 3
    assert image.shape[2] == 3
    return image


def _get_classes(data_subset_dir_path):
  class_names = []

  for file_name in os.listdir(data_subset_dir_path):
    if path.isdir(path.join(data_subset_dir_path, file_name)):
      class_names.append(file_name)

  return sorted(class_names)


def _get_filepaths(data_subset_dir_path, split_name):
  """Returns a list of filepaths and inferred class names.

  Args:
    dataset_dir: A directory containing a set of subdirectories representing
      class names. Each subdirectory should contain PNG or JPG encoded images.

  Returns:
    A list of image file paths, relative to `dataset_dir` and the list of
    subdirectories, representing class names.
  """
  image_file_paths = []

  # if not eval split, sort before shuffling for repeatability given a random seed
  # if eval split, sort anyway to preserve original frame ordering.
  class_names = os.listdir(data_subset_dir_path)

  # although the os.listdir() returns a random permutation of the contents of
  # data_subset_class_path, in the interest of reproduceability, sort the image
  # paths and then randomly permute them using the given random seed
  for class_name in class_names:
    data_subset_class_dir_path = os.path.join(data_subset_dir_path, class_name)
    if os.path.isdir(data_subset_class_dir_path):
      for image_name in sorted(os.listdir(data_subset_class_dir_path)):
        image_file_path = path.join(data_subset_class_dir_path, image_name)
        image_file_paths.append(image_file_path)

  #TODO: Explain why we dont need to shuffle the frames here
  # if creating an eval subset, leave the frames sorted so that predictions can
  # be sequentially compared to images (that appear ordered in the file system)
  # if split_name != 'eval':
  #   random.shuffle(image_file_paths)

  return image_file_paths


def _get_dataset_file_name(tfrecords_dir_path, dataset_name, split_name, shard_id, num_shards):
  output_file_name = dataset_name + '_%s_%05d-of-%05d.tfrecord' % (
    split_name, shard_id, num_shards)
  return path.join(tfrecords_dir_path, output_file_name)


def _convert_dataset(dataset_name, split_name, filepaths, class_names_to_ids, tfrecords_dir_path,
                     batch_size, num_shards, session_config=None, device_name=None):
  """Converts the given filepaths to a TFRecord dataset.

  Args:
    split_name: The name of the data subset; either 'training', 'dev', 'test' or 'eval'.
    filepaths: A list of absolute paths to png or jpg images.
    class_names_to_ids: A dictionary from class names (strings) to ids
      (integers).
    tfrecords_dir: The directory where the converted datasets are stored.
  """
  if not path.exists(path.join(path.join(tfrecords_dir_path, '..'), split_name)):
    raise AssertionError()

  filepaths_len = len(filepaths)

  with tf.Graph().as_default(), tf.device(device_name):
    image_reader = ImageReader()

    with tf.Session(config=session_config) as sess:
      for shard_id in range(num_shards):
        output_file_name = _get_dataset_file_name(
          tfrecords_dir_path, dataset_name, split_name, shard_id, num_shards)

        with tf.python_io.TFRecordWriter(output_file_name) as tfrecord_writer:
          start_ndx = shard_id * batch_size
          end_ndx = min((shard_id + 1) * batch_size, filepaths_len)
          for i in range(start_ndx, end_ndx):
            sys.stdout.write('\r>> Converting image %d/%d shard %d' % (i + 1, filepaths_len, shard_id))
            sys.stdout.flush()

            # Read the file_name:
            print(filepaths[i])
            image_data = tf.gfile.FastGFile(filepaths[i], 'rb').read()
            height, width = image_reader.read_image_dims(sess, image_data)

            class_name = path.basename(path.dirname(filepaths[i]))
            class_id = class_names_to_ids[class_name]

            example = dataset_utils.image_to_tfexample(image_data, b'jpg', height, width, class_id)
            tfrecord_writer.write(example.SerializeToString())

  sys.stdout.write('\n')
  sys.stdout.flush()


def _dataset_exists(dataset_name, tfrecords_dir_path, splits_to_shards):
  """Returns false if a named file does not exist or if the number of
  shards to be written is not equal to the number of shards that exists.

  Args:
    dataset_name: The name of the dataset.
    tfrecords_dir: The full path to the directory containing TFRecord shards.
    splits_to_shards: a map from split names (e.g. 'training') to a number of shards
  """
  for split_name, num_shards in splits_to_shards.items():
    for shard_id in range(num_shards):
      output_file_name = _get_dataset_file_name(
        tfrecords_dir_path, dataset_name, split_name, shard_id, num_shards)

      if not tf.gfile.Exists(output_file_name):
        return False
  return True


def get_split(dataset_name, split_name, datasets_root_dir_path, file_pattern=None, reader=None):
  """Gets a dataset tuple with instructions for reading construction.

  Args:
    split_name: A training/dev split name.
    dataset_dir: The base directory of the dataset sources.
    file_pattern: The file pattern to use when matching the dataset sources.
      It is assumed that the pattern contains a '%s' string so that the split
      name can be inserted.
    reader: The TensorFlow reader type.

  Returns:
    A `Dataset` namedtuple.

  Raises:
    ValueError: if `split_name` is not a valid training/dev split.
  """
  dataset_dir_path = path.join(datasets_root_dir_path, dataset_name)
  tfrecords_dir_path = path.join(dataset_dir_path, 'tfrecords')

  splits_file_name = dataset_name + '_splits.txt'

  if dataset_utils.has_splits(tfrecords_dir_path, splits_file_name):
    splits_to_sizes = dataset_utils.read_split_file(tfrecords_dir_path, splits_file_name)
  else:
    raise ValueError(path.join(tfrecords_dir_path, splits_file_name) + ' does not exist')

  if split_name not in splits_to_sizes:
    raise ValueError('split name %s was not recognized.' % split_name)

  if not file_pattern:
    file_pattern = dataset_name + '_%s_*.tfrecord'
  file_pattern = path.join(tfrecords_dir_path, file_pattern % split_name)

  # Allowing None in the signature so that dataset_factory can use the default.
  if reader is None:
    reader = tf.TFRecordReader

  keys_to_features = {
    'image/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
    'image/format': tf.FixedLenFeature((), tf.string, default_value='jpg'),
    'image/class/label': tf.FixedLenFeature(
      [], tf.int64, default_value=tf.zeros([], dtype=tf.int64))
  }

  items_to_handlers = {
    'image': slim.tfexample_decoder.Image(),
    'label': slim.tfexample_decoder.Tensor('image/class/label')
  }

  decoder = slim.tfexample_decoder.TFExampleDecoder(
    keys_to_features, items_to_handlers)

  labels_file_name = dataset_name + '_labels.txt'
  labels_to_names = None
  if dataset_utils.has_labels(tfrecords_dir_path, labels_file_name):
    labels_to_names = dataset_utils.read_label_file(tfrecords_dir_path, labels_file_name)

  descriptions_file_name = dataset_name + '_descriptions.txt'
  items_to_descriptions = None
  if dataset_utils.has_descriptions(tfrecords_dir_path, descriptions_file_name):
    items_to_descriptions = dataset_utils.read_description_file(tfrecords_dir_path, descriptions_file_name)

  statistics_file_name = dataset_name + '_statistics.txt'
  names_to_statistics = None
  if dataset_utils.has_statistics(tfrecords_dir_path, statistics_file_name):
    names_to_statistics = dataset_utils.read_statistics_file(tfrecords_dir_path, statistics_file_name)

  return slim.dataset.Dataset(
    data_sources=file_pattern,
    reader=reader,
    decoder=decoder,
    num_samples=splits_to_sizes[split_name],
    items_to_descriptions=items_to_descriptions,
    num_classes=len(labels_to_names),
    labels_to_names=labels_to_names,
    names_to_statistics=names_to_statistics)


def convert(datasets_root_dir_path, dataset_name, split_names, batch_size, random_seed,
            compute_statistics, session_config=None, device_name=None):
  """Runs the download and conversion operation.

  Args:
    datasets_root_dir: The directory where all datasets are stored.
    dataset_name: The the subfolder where the named dataset's TFRecords are stored.
    batch_size: The number of shards per batch of TFRecords.
    random_seed: The random seed used to instantiate the pseudo-random number generator
    that shuffles non-eval samples before creating TFRecord shards
    convert_eval_subset: If True, assume an subdir named 'eval' exists in datasets_root_dir
    and create TFRecords for the samples in that directory.
  """

  dataset_dir_path = path.join(datasets_root_dir_path, dataset_name)

  if not tf.gfile.Exists(dataset_dir_path):
    raise ValueError('The dataset ' + dataset_name + ' either does not exist or is misnamed')

  random.seed(random_seed)

  splits_to_filepaths = {}
  splits_to_sizes = {}
  splits_to_shards = {}

  class_path = path.join(dataset_dir_path, split_names[0])

  if not path.exists(class_path):
    os.mkdir(class_path)

  class_names = _get_classes(class_path)

  for split_name in split_names:
    split_dir_path = path.join(dataset_dir_path, split_name)
    image_filepaths = _get_filepaths(split_dir_path, split_name)
    num_samples = len(image_filepaths)

    splits_to_filepaths[split_name] = image_filepaths
    splits_to_sizes[split_name] = num_samples
    splits_to_shards[split_name] = int(math.ceil(num_samples / batch_size))

  tfrecords_dir_path = path.join(dataset_dir_path, 'tfrecords')

  if tf.gfile.Exists(tfrecords_dir_path):
    if _dataset_exists(dataset_name, tfrecords_dir_path, splits_to_shards):
      print('Dataset files already exist. Exiting without re-creating them.')
      return
    else:
      for file in os.listdir(tfrecords_dir_path):
        os.remove(path.join(tfrecords_dir_path, file))
  else:
    tf.gfile.MakeDirs(tfrecords_dir_path)

  # Write the statistics file:
  if compute_statistics:
    means, std_devs, mins, maxs = _compute_statistics(
      splits_to_filepaths['training'], session_config=session_config, device_name=device_name)
    statistics_file_name = dataset_name + '_statistics.txt'
    names_to_statistics = {
      'r_mean': means[0], 'g_mean': means[1], 'b_mean': means[2],
      'r_std_dev': std_devs[0], 'g_std_dev': std_devs[1], 'b_std_dev': std_devs[2],
      'r_min': mins[0], 'g_min': mins[1], 'b_min': mins[2],
      'r_max': maxs[0], 'g_max': maxs[1], 'b_max': maxs[2]
    }
    dataset_utils.write_statistics_file(
      names_to_statistics, tfrecords_dir_path, statistics_file_name)

  class_name_enum = [class_name for class_name in enumerate(class_names)]

  class_names_to_ids = {class_name: ndx for (ndx, class_name) in class_name_enum}

  # First, convert the data subsets.
  for split_name in split_names:
    _convert_dataset(
      dataset_name, split_name, splits_to_filepaths[split_name],
      class_names_to_ids, tfrecords_dir_path, batch_size, splits_to_shards[split_name])

  # Then, write the labels file:
  labels_file_name = dataset_name + '_labels.txt'
  labels_to_class_names = {ndx: class_name for (ndx, class_name) in class_name_enum}
  dataset_utils.write_label_file(labels_to_class_names, tfrecords_dir_path, labels_file_name)

  # Then, write the splits file:
  splits_file_name = dataset_name + '_splits.txt'
  # splits_to_sizes = {'training': num_traininging_samples, 'dev': num_dev_samples}
  dataset_utils.write_split_file(splits_to_sizes, tfrecords_dir_path, splits_file_name)

  # Finaly, write the descriptions file:
  descriptions_file_name = dataset_name + '_descriptions.txt'
  items_to_descriptions = {'image': 'A color image of varying size.',
                           'label': 'A single integer between 0 and 1'}
  dataset_utils.write_description_file(items_to_descriptions, tfrecords_dir_path, descriptions_file_name)

  print('\nFinished converting the ' + dataset_name + ' dataset!')


def _compute_statistics(
    source_file_paths, epsilon=1e-3, session_config=None, device_name=None):
  """Compute the min, max, mean, standard deviation of the training subset,
  then write the results to a file.

  Args:
    source_file_paths: A list of paths to the images in a training subset.
    epsilon: Value to use in place of the standard deviation in case
    it is too small to use as a divisor.
  """
  tf.logging.info('Computing image subset statistics.')

  with tf.Graph().as_default(), tf.device(device_name):
    image_reader = ImageReader()

    # track the number of samples processed so far
    training_set_size = tf.Variable(initial_value=0.0, trainable=False,
                                    name='training_set_size', dtype=tf.float32)

    # aggregate per-image mean values over the entire training_set
    training_set_per_channel_mean = tf.Variable(
      initial_value=[0.0, 0.0, 0.0], trainable=False,
      name='training_set_per_channel_mean', dtype=tf.float32)

    # aggregate per-image variance values over the entire training_set
    training_set_per_channel_squared_distance = tf.Variable(
      initial_value=[0.0, 0.0, 0.0], trainable=False,
      name='training_set_per_channel_squared_distance', dtype=tf.float32)

    image_placeholder = tf.placeholder(dtype=tf.uint8)

    image = tf.image.convert_image_dtype(image_placeholder, dtype=tf.float32)

    # compute per-channel means and variances for the given image
    training_sample_per_channel_mean, training_sample_per_channel_variance = tf.nn.moments(
      image, axes=[0, 1], name='moments')

    # compute one iteration of Welford's online algorithm for training_set mean and variance
    training_set_size = training_set_size.assign_add(1.0)

    delta = tf.subtract(training_sample_per_channel_mean, training_set_per_channel_mean)
    delta_ratio = tf.divide(delta, training_set_size)
    training_set_per_channel_mean = training_set_per_channel_mean.assign_add(delta_ratio)

    delta2 = tf.subtract(training_sample_per_channel_mean, training_set_per_channel_mean)
    delta_product = tf.multiply(delta, delta2)
    training_set_per_channel_squared_distance = \
      training_set_per_channel_squared_distance.assign_add(delta_product)

    decoded_images = []
    statistics = None

    with tf.Session(config=session_config) as sess:
      sess.run(tf.global_variables_initializer())

      for source_file_path in source_file_paths:
        # TODO: Replace with check for image file type
        if os.path.isfile(source_file_path):
          image_data = tf.gfile.FastGFile(source_file_path, 'rb').read()
          image_data = image_reader.decode_jpeg(sess, image_data)

          # TODO: Explain the purpose of aggregating all images read from disk
          decoded_images.append(image_data)

          # Each call to sess.run() will perform a single iteration of the Welford algorithm.
          # Following the final loop iteration, the statistics variable will hold the
          # training_set_size, training_set_per_channel_mean, and the sum of the squared
          # distances from the mean (to be later divided by the training_set_size to yield
          # the training set standard deviation
          statistics = sess.run([training_set_size,
                                 training_set_per_channel_mean,
                                 training_set_per_channel_squared_distance],
                                feed_dict={image_placeholder: image_data})

    # compute training set per-channel min and max after standardizing the
    standardized_image_placeholder = tf.placeholder(dtype=tf.uint8)
    training_set_per_channel_mean_placeholder = tf.placeholder(dtype=tf.float32)
    training_set_per_channel_std_dev_placeholder = tf.placeholder(dtype=tf.float32)

    # track the per-channel maximum and minimum pixel values for use in rescaling
    training_set_per_channel_min = tf.Variable(initial_value=[0.0, 0.0, 0.0],
                                               trainable=False,
                                               name='training_set_per_channel_min',
                                               dtype=tf.float32)

    training_set_per_channel_max = tf.Variable(initial_value=[0.0, 0.0, 0.0],
                                               trainable=False,
                                               name='training_set_per_channel_max',
                                               dtype=tf.float32)

    standardized_image = tf.image.convert_image_dtype(standardized_image_placeholder,
                                                      dtype=tf.float32)
    # standardize image
    standardized_image = tf.divide(
      tf.subtract(standardized_image, training_set_per_channel_mean_placeholder),
      training_set_per_channel_std_dev_placeholder)

    training_sample_per_channel_max = tf.reduce_max(standardized_image, axis=[0, 1])

    training_set_per_channel_max = training_set_per_channel_max.assign(
      tf.maximum(training_set_per_channel_max, training_sample_per_channel_max))

    training_sample_per_channel_min = tf.reduce_min(standardized_image, axis=[0, 1])

    training_set_per_channel_min = training_set_per_channel_min.assign(
      tf.minimum(training_set_per_channel_min, training_sample_per_channel_min))

    with tf.Session(config=session_config) as sess:
      sess.run(tf.global_variables_initializer())

      training_set_per_channel_mean = statistics[1]
      training_set_per_channel_variance = tf.divide(statistics[2],
                                                    tf.subtract(statistics[0], 1))
      training_set_per_channel_std_dev = sess.run(
        tf.maximum(tf.sqrt(training_set_per_channel_variance), epsilon))
      # identify the global per-channel minimum and maximum across the entire training set
      # after standardizing each image (allowing test-time images to be scaled down
      # if they exceed the training set max/min values
      for decoded_image in decoded_images:
        statistics = sess.run(
          [training_set_per_channel_min, training_set_per_channel_max], feed_dict={
            standardized_image_placeholder: decoded_image,
            training_set_per_channel_mean_placeholder: training_set_per_channel_mean,
            training_set_per_channel_std_dev_placeholder: training_set_per_channel_std_dev})

      training_set_per_channel_min = statistics[0]
      training_set_per_channel_max = statistics[1]

    return training_set_per_channel_mean, \
           training_set_per_channel_std_dev, \
           training_set_per_channel_min, \
           training_set_per_channel_max


def compute_statistics(datasets_root_dir_path, dataset_name, session_config=None, device_name=None):
  """Computes the mean, standard deviation, min and max of the entire
  training subset of the named data set, and then stores the values in
  a formatted text file within the data set's tfrecords sub-directory.
  Max and min are computed after standardizing the images so that test-time
  image pixel values can be downscaled if they exceed the max/min bounds.

  Args:
    datasets_root_dir: The directory where all datasets are stored.
    dataset_name: The the subfolder where the TFRecords of the dataset
    under consideration are stored.
  """

  dataset_path = path.join(datasets_root_dir_path, dataset_name)

  if not tf.gfile.Exists(dataset_path):
    raise ValueError('The dataset ' + dataset_name + ' either does not exist or is misnamed')

  tfrecords_dir_path = path.join(dataset_path, 'tfrecords')

  if not tf.gfile.Exists(tfrecords_dir_path):
    tf.gfile.MakeDirs(tfrecords_dir_path)

  split_path = path.join(dataset_path, 'training')
  training_image_filepaths = _get_filepaths(split_path, 'training')

  means, std_devs, mins, maxs = _compute_statistics(
    training_image_filepaths, session_config=session_config, device_name=device_name)
  statistics_file_name = dataset_name + '_statistics.txt'
  names_to_statistics = {
    'r_mean': means[0], 'g_mean': means[1], 'b_mean': means[2],
    'r_std_dev': std_devs[0], 'g_std_dev': std_devs[1], 'b_std_dev': std_devs[2],
    'r_min': mins[0], 'g_min': mins[1], 'b_min': mins[2],
    'r_max': maxs[0], 'g_max': maxs[1], 'b_max': maxs[2]
  }
  dataset_utils.write_statistics_file(
    names_to_statistics, tfrecords_dir_path, statistics_file_name)

  print('\nFinished computing ' + dataset_name + ' statistics!')


def _create_data_set_paths(data_set_dir_path, class_dir_names, create_standard_subsets,
                           create_eval_subset):
  """Creates the subfolder structure within a dataset directory.

    Args:
      data_set_dir_path: The full path to the directory containing the dataset being created.
      class_dir_names: A list of class labels to assign as names of the sub-folders of each data subset
      create_standard_subsets: A bool determining whether folders for the training, dev and test subsets are to be created
      create_eval_subset: A bool determining whether a folder for the eval subset is to be created
    """
  # create training, dev, and test sub-directories of dataset_dest_path
  # each will contain one sub-folder per class
  class_dir_paths = {}
  subset_dir_names = []

  if create_standard_subsets:
    subset_dir_names.extend(['training', 'dev', 'test'])

  if create_eval_subset:
    subset_dir_names.append('eval')

  if not path.exists(data_set_dir_path):
    os.mkdir(data_set_dir_path)

  for subset_dir_name in subset_dir_names:
    subset_dir_path = path.join(data_set_dir_path, subset_dir_name)

    if not path.exists(subset_dir_path):
      os.mkdir(subset_dir_path)

    for class_dir_name in class_dir_names:
      class_dir_path = path.join(subset_dir_path, class_dir_name)

      if not path.exists(class_dir_path):
        os.mkdir(class_dir_path)

      class_dir_paths[subset_dir_name + '_' + class_dir_name] = class_dir_path

  return class_dir_paths


def _populate_data_set_paths(
    class_dir_paths, class_dir_name, class_sub_dir_path, split_name, frame):
  dest_video_frame_path = path.join(
    class_dir_paths[split_name + '_' + class_dir_name], frame)
  if not path.exists(dest_video_frame_path):
    source_video_frame_path = path.join(class_sub_dir_path, frame)
    if path.islink(source_video_frame_path):
      print('Copying frame at source path: {}\nto destination path: {}'
            .format(source_video_frame_path, dest_video_frame_path))
      shutil.copy(source_video_frame_path, dest_video_frame_path, follow_symlinks=False)
    else:
      print('Creating symbolic link from source path: {}\nto destination path: {}'
            .format(source_video_frame_path, dest_video_frame_path))
      os.symlink(source_video_frame_path, dest_video_frame_path)


def create(class_names, create_standard_subsets, create_eval_subset, data_source_dir_path,
           data_set_dir_path, training_ratio, dev_ratio, random_seed, balance_subsets):
  '''Creates one destination folder for each class-subset pair (e.g. training_class_0_dir_path or
  dev_class_1_dir_path). For each subfolder (containing the frames of a single video) of the
  datasource_dir_path (containing many subfolders for many videos), randomly samples
  training_ratio %, dev_ratio % and test_percent % of subfolder contents and then moves
  all training, dev, and test sample frames into training, dev, and test folders,
  respectively. This method fits into the pipeline between frame extraction and tfrecord
  creation, with dataset standardization to eventually be placed between this function and
  tfrecord creation when implemented'''

  random.seed(random_seed)

  class_dir_paths = _create_data_set_paths(
    data_set_dir_path, class_names, create_standard_subsets, create_eval_subset)

  video_frame_dir_names = sorted(set(os.listdir(data_source_dir_path)) -
                            {'training', 'dev', 'test', 'eval', 'tfrecords'})

  # for each folder of frames in the data dir
  for video_frame_dir_name in video_frame_dir_names:
    video_frame_dir_path = path.join(data_source_dir_path, video_frame_dir_name)

    if path.isdir(video_frame_dir_path):
      class_sub_dir_paths = {}
      class_sub_dir_frame_sets = {}
      class_sub_dir_frame_counts = []

      for class_sub_dir_name in class_names:
        if class_sub_dir_name in os.listdir(video_frame_dir_path):
          class_sub_dir_path = path.join(video_frame_dir_path, class_sub_dir_name)
          class_sub_dir_paths[class_sub_dir_name] = class_sub_dir_path

          # sort in case we want to reproduce results using a given random seed
          class_sub_dir_frame_set = sorted(set(os.listdir(class_sub_dir_path)))
          class_sub_dir_frame_sets[class_sub_dir_name] = class_sub_dir_frame_set

          class_sub_dir_frame_counts.append(len(class_sub_dir_frame_set))

      class_sub_dir_frame_counts.sort()

      for class_sub_dir_name, class_sub_dir_path in class_sub_dir_paths.items():
        tf.logging.info('class_sub_dir_name: {}'.format(class_sub_dir_name))
        frame_set = class_sub_dir_frame_sets[class_sub_dir_name]
        n_frames = len(frame_set)
        tf.logging.info('initial n_frames: {}'.format(n_frames))
        # For multi-class classification, not all classes should have equal
        # representation because the least populous class may be very small. Instead,
        # balance the dataset by making the two most populous classes equal in size.
        # In practice, this means the 'background' class will be downsized, which is
        # helpful because it is usually ~50-90% larger than the next largest class, and
        # most samples are probably redundant in their information content
        if balance_subsets and n_frames == class_sub_dir_frame_counts[-1]:
          n_frames = class_sub_dir_frame_counts[-2]
          tf.logging.info('balanced n_frames: {}'.format(n_frames))
          frame_set = random.sample(frame_set, n_frames)
          tf.logging.info('balanced frame_set len: {}'.format(len(frame_set)))
        else:
          random.shuffle(frame_set)

        if create_eval_subset:
          # create the eval subset before shuffling frames so that
          # eval probabilities can be viewed in sequential order.
          for index in range(n_frames):
            _populate_data_set_paths(class_dir_paths, class_sub_dir_name, class_sub_dir_path,
                                     'eval', frame_set[index])

        if create_standard_subsets:
          n_training_frames = int(training_ratio * n_frames)
          n_dev_frames = int(dev_ratio * n_frames)
          n_test_frames = n_frames - (n_training_frames + n_dev_frames)

          for _ in range(n_training_frames):
            _populate_data_set_paths(class_dir_paths, class_sub_dir_name, class_sub_dir_path,
                                     'training', frame_set.pop(0))

          for _ in range(n_dev_frames):
            _populate_data_set_paths(class_dir_paths, class_sub_dir_name, class_sub_dir_path,
                                     'dev', frame_set.pop(0))

          for _ in range(n_test_frames):
            _populate_data_set_paths(class_dir_paths, class_sub_dir_name, class_sub_dir_path,
                                     'test', frame_set.pop(0))

          assert len(frame_set) == 0
