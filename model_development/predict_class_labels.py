import os
import signal
import sys
import time
import uuid

import numpy as np
import tensorflow as tf

from model_development.nets import inception
from preprocessing import inception_preprocessing

path = os.path
FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('data_set_dir', None,
  'A path to a directory of images or to the parent of multiple directories of images.')

tf.app.flags.DEFINE_string('raw_data_dir', None,
  'A path to a directory of images or to the parent of multiple directories of images.')

tf.app.flags.DEFINE_string('model_path', None,
  'A path to a protobuf file containing the model that will analyze the images in data_dir.')

tf.app.flags.DEFINE_string('labels_path', None,
  'A path to a text file containing a mapping from class ids to class names.')

tf.app.flags.DEFINE_string('io_tensor_names_path', None,
  'A path to a text file containing the names of the model''s input and output tensors.')

tf.app.flags.DEFINE_string('label_predictions_dir', None,
  'A path to a directory where a report of image analysis will be stored.')

tf.app.flags.DEFINE_integer('batch_size', 32, 'The number of images analyzed concurrently.')

tf.app.flags.DEFINE_float('gpu_memory_fraction', 0.9, 'The ratio of total memory across all '
  'available GPUs to use with this process. Defaults to a suggested max of 0.9.')

tf.app.flags.DEFINE_string('gpu_device_num', None, 'The device number of a single GPU to use for'
  ' evaluation on a multi-GPU system. Defaults to zero.')

tf.app.flags.DEFINE_boolean('cpu_only', False, 'Explicitly assign all evaluation ops to the '
  'CPU on a GPU-enabled system. Defaults to False.')

tf.app.flags.DEFINE_string('video_names_path', None, 'Path to a file containing a comma-separated '
'list of names of videos to include when predicting frame labels. Use this flag when the raw_data_'
'dir contains videos that should be ignored.')


def load_model(model_path):
  with tf.gfile.GFile(model_path, 'rb') as model:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(model.read())
    tensor_id = str(uuid.uuid4())
  return tf.import_graph_def(graph_def, name=tensor_id), tensor_id


def load_tensor_names(io_tensor_names_path):
  meta_map = read_meta_file(io_tensor_names_path)
  return {key: value + ':0' for key, value in meta_map.items()}


def load_labels(labels_path):
  meta_map = read_meta_file(labels_path)
  return {int(key): value for key, value in meta_map.items()}


def read_meta_file(file_path):
  meta_lines = [line.rstrip().split(":") for line in tf.gfile.GFile(file_path)]
  return {line[0]: line[1] for line in meta_lines}


def print_processing_duration(start_time, msg):
  end_time = time.time() - start_time
  minutes, seconds = divmod(end_time, 60)
  hours, minutes = divmod(minutes, 60)
  print('{:s}: {:02d}:{:02d}:{:02d}\n'.format(msg, int(hours), int(minutes), int(seconds)))


def add_example_to_label_predictions(
    raw_data_path, label_predictions_dir_path, video_file_name, predicted_class_dir_name):
  label_predictions_subdir_path = path.join(label_predictions_dir_path, video_file_name)
  # for convenience, create paths at the per-video level and the per-round level
  predicted_class_dir_path = path.join(label_predictions_subdir_path, predicted_class_dir_name)

  if not path.exists(predicted_class_dir_path):
    os.makedirs(predicted_class_dir_path)

  example_file_name = path.basename(raw_data_path)
  label_predictions_example_file_path = path.join(predicted_class_dir_path, example_file_name)

  if path.exists(label_predictions_example_file_path):
    os.remove(label_predictions_example_file_path)

  os.symlink(raw_data_path, label_predictions_example_file_path)


def output_label_predictions(
    video_file_name, example_path_list, prediction_list, label_map, label_predictions_dir):
  for i in range(len(prediction_list)):
    prediction = prediction_list[i]
    class_id = np.argmax(prediction)
    confidence = prediction[class_id]
    
    if confidence > 0.9:
      predicted_class_dir_name = '90_to_100_percent_likely_' + label_map[class_id]
    elif confidence > 0.8:
      predicted_class_dir_name = '80_to_90_percent_likely_' + label_map[class_id]
    elif confidence > 0.7:
      predicted_class_dir_name = '70_to_80_percent_likely_' + label_map[class_id]
    elif confidence > 0.6:
      predicted_class_dir_name = '60_to_70_percent_likely_' + label_map[class_id]
    elif confidence > 0.5:
      predicted_class_dir_name = '50_to_60_percent_likely_' + label_map[class_id]
    else:
      predicted_class_dir_name = 'undecided'

    add_example_to_label_predictions(
      example_path_list[i], label_predictions_dir, video_file_name, predicted_class_dir_name)


def interrupt_handler(signal_number, _):
    tf.logging.info(
        'Received interrupt signal (%d). Unsetting CUDA_VISIBLE_DEVICES environment variable.', signal_number)
    os.unsetenv('CUDA_VISIBLE_DEVICES')
    sys.exit(0)


def draw_progress_bar(percent, bar_len=20):
  sys.stdout.write("\r")
  progress = ""
  for i in range(bar_len):
    if i < int(bar_len * percent):
      progress += "="
    else:
      progress += " "
  sys.stdout.write("[ %s ] %.2f%%" % (progress, percent * 100))
  sys.stdout.flush()


def percentage(part, whole):
  return 100 * (float(part) / float(whole))


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


def decode_and_preprocess_for_eval(image):
  image = tf.image.decode_jpeg(image, channels=3)
  image_size = inception.inception_v3.default_image_size
  return inception_preprocessing.preprocess_for_eval(
    image, image_size, image_size, central_fraction=None)


def predict_class_labels(
    image_list, num_labels, placeholder_map, image_size, batch_size, tf_session):
  num_frames = len(image_list)
  num_batches = int(np.ceil(np.divide(num_frames, batch_size)))
  
  # populating a pre-allocated array is faster than creating a new list every batch
  prediction_array = np.ndarray(dtype=np.float32, shape=(num_frames, num_labels))

  image_list_placeholder = tf.placeholder(dtype=tf.string)

  images = tf.data.Dataset.from_tensor_slices(image_list_placeholder)
  images = images.map(decode_and_preprocess_for_eval, batch_size)
  image_batches = images.batch(batch_size)

  batch_iterator = image_batches.make_initializable_iterator()
  next_batch = batch_iterator.get_next()
  
  tf_session.run(batch_iterator.initializer, feed_dict={image_list_placeholder: image_list})

  for batch_num in range(num_batches):
    try:
      image_batch = next_batch.eval(session=tf_session)
      
      lower_index = batch_num * batch_size
      upper_index = min((batch_num + 1) * batch_size, num_frames)
      
      prediction_array[lower_index:upper_index] = tf_session.run(
        placeholder_map['softmax_tensor'],
        {placeholder_map['image_batch_placeholder']: image_batch,
         placeholder_map['image_size_placeholder']: image_size})

      draw_progress_bar(percentage(upper_index, num_frames) / 100, 40)
    except tf.errors.OutOfRangeError:
      break
  print('\n')
  return prediction_array


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

  if FLAGS.cpu_only:
    device_name = '/cpu:0'
    tf.logging.info('Setting CUDA_VISIBLE_DEVICES environment variable to None.')
    os.putenv('CUDA_VISIBLE_DEVICES', '')
    session_config = None
  else:
    gpu_options = tf.GPUOptions(allow_growth=True,
                                per_process_gpu_memory_fraction=FLAGS.gpu_memory_fraction)
    session_config = tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options)

    if FLAGS.gpu_device_num:
      device_name = '/gpu:' + FLAGS.gpu_device_num
      tf.logging.info('Setting CUDA_VISIBLE_DEVICES environment variable to {}.'.format(
        FLAGS.gpu_device_num))
      os.putenv('CUDA_VISIBLE_DEVICES', FLAGS.gpu_device_num)
    else:
      device_name = None

  signal.signal(signal.SIGINT, interrupt_handler)

  model_graph, graph_name = load_model(FLAGS.model_path)

  labels_path = '{}/tfrecords/{}_labels.txt'.format(FLAGS.data_set_dir,
                                                    path.basename(FLAGS.data_set_dir))
  label_map = load_labels(labels_path)
  num_labels = len(label_map)

  io_tensor_names_path = FLAGS.io_tensor_names_path if FLAGS.io_tensor_names_path else '{}/io_tensor_names.txt'.format(
    path.dirname(FLAGS.model_path))
  io_tensor_name_map = load_tensor_names(io_tensor_names_path)

  io_tensor_placeholder_map = {}

  with tf.Session(graph=model_graph) as sess, tf.device(device_name):
    io_tensor_placeholder_map['image_batch_placeholder'] = sess.graph.get_tensor_by_name(
      '{}/{}'.format(graph_name, io_tensor_name_map['input_tensor_name']))
    io_tensor_placeholder_map['softmax_tensor'] = sess.graph.get_tensor_by_name(
      '{}/{}'.format(graph_name, io_tensor_name_map['output_tensor_name']))
    io_tensor_placeholder_map['image_size_placeholder'] = sess.graph.get_tensor_by_name(
      '{}/{}'.format(graph_name, io_tensor_name_map['image_size_tensor_name']))

  # iterate over every subdirectory of the raw_data_dir
  if path.isfile(FLAGS.video_names_path):
    with open(FLAGS.video_names_path, 'r') as video_names:
      raw_data_subdirs = sorted(video_names.readline().rstrip().split(','))
  else:
    raw_data_subdirs = sorted(os.listdir(FLAGS.raw_data_dir))

  # identify and store the file names of examples already included in
  # the previously created data set named FLAGS.data_set_name
  print('Identifying previously incorporated examples in {}.'.format(FLAGS.data_set_dir))
  start = time.time()
  incorporated_example_file_name_set = get_previously_incorporated_example_file_name_set(
    FLAGS.data_set_dir)
  print_processing_duration(start, 'Elapsed time')

  for subdir in raw_data_subdirs:
    subdir_path = path.join(FLAGS.raw_data_dir, subdir)

    print('Identifying previously unincorporated examples in {}'.format(subdir))
    start = time.time()
    unincorporated_example_file_path_list = [
      path.join(subdir_path, name) for name in sorted(os.listdir(subdir_path))
      if name not in incorporated_example_file_name_set]
    print_processing_duration(start, 'Elapsed time')

    if not all([path.isfile(file_path) and file_path[-4:] == '.jpg'
                for file_path in unincorporated_example_file_path_list]):
      raise ValueError('The directory {} is expected to only contain image files.'
                       .format(subdir_path))

    print('Loading previously unincorporated examples from {}'.format(subdir))
    start = time.time()
    image_list = np.array([tf.gfile.GFile(file_path, 'rb').read()
                           for file_path in unincorporated_example_file_path_list
                           if os.stat(file_path).st_size > 0])
    print_processing_duration(start, 'Elapsed time')

    with tf.Session(graph=model_graph, config=session_config) as sess, tf.device(device_name):
      print('Processing {} examples from {}.'.format(len(image_list), subdir))
      start = time.time()
      prediction_array = predict_class_labels(image_list, num_labels, io_tensor_placeholder_map,
                                              inception.inception_v3.default_image_size, FLAGS.batch_size, sess)
      print_processing_duration(start, 'Elapsed time')

      print('Writing class label predictions for {}.'.format(subdir))
      start = time.time()
      output_label_predictions(subdir, unincorporated_example_file_path_list,
                               prediction_array, label_map, FLAGS.label_predictions_dir)
      print_processing_duration(start, 'Elapsed time')


# TODO: Make batch size a function of available system memory.
if __name__ == '__main__':
  main()
