import csv
import numpy as np
import os
import tensorflow as tf
import time

path = os.path

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('data_set_dir', None,
                           'A path to a directory of images or to the parent of multiple directories of images.')

tf.app.flags.DEFINE_string(
  'raw_data_dir', None,
  'A path to a directory of images or to the parent of multiple directories of images.')

tf.app.flags.DEFINE_string('report_dir_path', None, 'A path to a folder containing CSV files.')

tf.app.flags.DEFINE_string(
  'label_predictions_dir', None,
  'A path to a directory where a report of image analysis will be stored.')

tf.app.flags.DEFINE_integer(
  'num_classes', 2, 'The number of classes that the model was trained to predict.')

tf.app.flags.DEFINE_boolean(
  'skip_priors', False,
  'Unsafely skip symlink creation for videos aready represented in label_predictions_dir')


def print_processing_duration(start_time, msg):
  end_time = time.time() - start_time
  minutes, seconds = divmod(end_time, 60)
  hours, minutes = divmod(minutes, 60)
  print('{:s}: {:02d}:{:02d}:{:02d}\n'.format(msg, int(hours), int(minutes), int(seconds)))


def add_example_to_label_predictions(
    raw_data_path, label_predictions_dir_path, video_file_name, predicted_class_dir_name):
  label_predictions_subdir_path = path.join(label_predictions_dir_path, video_file_name)

  if FLAGS.skip_priors and path.exists(label_predictions_subdir_path):
    return

  # for convenience, create paths at the per-video level and the per-round level
  predicted_class_dir_path = path.join(label_predictions_subdir_path, predicted_class_dir_name)

  if not path.exists(predicted_class_dir_path):
    os.makedirs(predicted_class_dir_path)

  example_file_name = path.basename(raw_data_path)
  label_predictions_example_file_path = path.join(predicted_class_dir_path, example_file_name)

  if path.exists(label_predictions_example_file_path):
    return

  os.symlink(raw_data_path, label_predictions_example_file_path)


def output_label_predictions(
    video_file_name, class_prob_map, class_name_map, label_predictions_dir):
  for example_path, example_probs in class_prob_map.items():
    class_id = np.argmax(example_probs)
    confidence = example_probs[class_id]

    if confidence > 0.9:
      predicted_class_dir_name = '90_to_100_percent_likely_' + class_name_map[class_id]
    elif confidence > 0.8:
      predicted_class_dir_name = '80_to_90_percent_likely_' + class_name_map[class_id]
    elif confidence > 0.7:
      predicted_class_dir_name = '70_to_80_percent_likely_' + class_name_map[class_id]
    elif confidence > 0.6:
      predicted_class_dir_name = '60_to_70_percent_likely_' + class_name_map[class_id]
    elif confidence > 0.5:
      predicted_class_dir_name = '50_to_60_percent_likely_' + class_name_map[class_id]
    else:
      predicted_class_dir_name = 'undecided'

    add_example_to_label_predictions(
      example_path, label_predictions_dir, video_file_name, predicted_class_dir_name)


def get_previously_incorporated_example_file_name_set(data_set_dir):
  standard_subset_names = ['training', 'dev', 'test']

  previously_incorporated_example_file_name_set = set()

  data_subset_dir_names = [name for name in os.listdir(data_set_dir)
                           if name in standard_subset_names]

  for data_subset_dir_name in data_subset_dir_names:
    data_subset_dir_path = path.join(data_set_dir, data_subset_dir_name)
    class_dir_names = os.listdir(data_subset_dir_path)

    for class_dir_name in class_dir_names:
      class_dir_path = path.join(data_subset_dir_path, class_dir_name)
      example_file_names = os.listdir(class_dir_path)
      previously_incorporated_example_file_name_set.update(example_file_names)

  return previously_incorporated_example_file_name_set


def main():
  # the three directories should already exist as a result of the creation of the
  # seed set and development of the seed model
  if not path.isdir(FLAGS.raw_data_dir):
    raise ValueError('The directory {} does not exist.'.format(FLAGS.raw_data_dir))

  if not path.isdir(FLAGS.label_predictions_dir):
    raise ValueError('The directory {} does not exist.'.format(FLAGS.label_predictions_dir))

  if not path.isdir(FLAGS.data_set_dir):
    raise ValueError('The directory {} does not exist.'.format(FLAGS.data_set_dir))

  tf.logging.set_verbosity(tf.logging.INFO)

  global_start = time.time()

  print('Identifying previously incorporated examples in {}.'.format(FLAGS.data_set_dir))
  start = time.time()
  incorporated_example_file_name_set = get_previously_incorporated_example_file_name_set(FLAGS.data_set_dir)
  print_processing_duration(start, 'Elapsed time')

  for report_file_name in os.listdir(FLAGS.report_dir_path):
    report_file_path = path.join(FLAGS.report_dir_path, report_file_name)

    print('Reading CSV for report {}.'.format(report_file_name))
    start = time.time()
    with open(report_file_path, newline='') as report_file:
      report_reader = csv.reader(report_file)
      header = next(report_reader)
      class_name_list = [name.rstrip('_probability') for name in header[2:2 + FLAGS.num_classes]]
      class_name_map = {i: class_name_list[i] for i in range(len(class_name_list))}
      class_prob_map = {row[1]: [float(prob) for prob in row[2:2 + FLAGS.num_classes]]
                        for row in report_reader}
    print_processing_duration(start, 'Elapsed time')

    report_name, _ = path.splitext(report_file_name)

    print('Identifying previously incorporated examples in {}.'.format(report_name))
    start = time.time()
    image_ext = '.jpg'
    example_file_name_set = set([report_name + '_{:07d}'.format(i) + image_ext
                                 for i in range(1, len(class_prob_map) + 1)])
    unincorporated_example_file_name_set = example_file_name_set - incorporated_example_file_name_set
    unincorporated_class_prob_map = {}
    for unincorporated_example_file_name in unincorporated_example_file_name_set:
      frame_dir_path = path.join(FLAGS.raw_data_dir, report_name)
      unincorporated_example_file_path = path.join(frame_dir_path,
                                                   unincorporated_example_file_name)
      frame_number = unincorporated_example_file_name[len(report_name) + 1:-len(image_ext)]
      unincorporated_class_prob_map[unincorporated_example_file_path] = class_prob_map[frame_number]
    print_processing_duration(start, 'Elapsed time')

    print('Writing class label predictions for {}.'.format(report_name))
    start = time.time()
    output_label_predictions(report_name, unincorporated_class_prob_map, class_name_map, FLAGS.label_predictions_dir)
    print_processing_duration(start, 'Elapsed time')

  print_processing_duration(global_start, 'Total elapsed time')

if __name__ == '__main__':
  main()