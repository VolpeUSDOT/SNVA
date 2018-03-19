import numpy as np
import operator
import os
import signal
import sys
import tensorflow as tf
import timeit
import uuid

from preprocessing import inception_preprocessing

path = os.path
FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('data_dir', None,
  'A path to a directory of images or to the parent of multiple directories of images.')

tf.app.flags.DEFINE_string('model_path', None,
  'A path to a protobuf file containing the model that will analyze the images in data_dir.')

tf.app.flags.DEFINE_string('labels_path', None,
  'A path to a text file containing a mapping from class ids to class names.')

tf.app.flags.DEFINE_string('io_tensor_path', None,
  'A path to a text file containing the names of the model''s input and output tensors.')

tf.app.flags.DEFINE_string('report_dir', None,
  'A path to a directory where a report of image analysis will be stored.')

tf.app.flags.DEFINE_integer('batch_size', 32, 'The number of images analyzed concurrently.')

tf.app.flags.DEFINE_float('gpu_memory_fraction', 0.9, 'The ratio of total memory across all '
  'available GPUs to use with this process. Defaults to a suggested max of 0.9.')

tf.app.flags.DEFINE_integer('gpu_device_num', 0, 'The device number of a single GPU to use for'
  ' evaluation on a multi-GPU system. Defaults to zero.')

tf.app.flags.DEFINE_boolean('cpu_only', False, 'Explicitly assign all evaluation ops to the '
  'CPU on a GPU-enabled system. Defaults to False.')


def load_model(model_path):
  with tf.gfile.GFile(model_path, 'rb') as model:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(model.read())
    tensor_id = str(uuid.uuid4())
  return tf.import_graph_def(graph_def, name=tensor_id), tensor_id


def load_tensor_names(io_tensor_path):
  tensor_name_map = read_meta_file(io_tensor_path)
  for key, value in tensor_name_map.items():
    tensor_name_map[key] = value + ':0'
  return tensor_name_map


def load_labels(labels_path):
  meta_map = read_meta_file(labels_path)
  label_map = {int(key): value for key, value in meta_map.items()}
  return label_map


def read_meta_file(file_path):
  lines = [line.rstrip().split(":") for line in tf.gfile.GFile(file_path)]
  meta_map = {line[0]: line[1] for line in lines}
  return meta_map


def print_processing_duration(start_time, stop_time, msg):
  print(' ')
  total_time = stop_time - start_time
  mins, secs = divmod(total_time, 60)
  hours, mins = divmod(mins, 60)
  sys.stdout.write(msg + ": %d:%d:%d.\n\n" % (hours, mins, secs))


def decode_jpeg(image):
  return tf.image.decode_jpeg(image, channels=3)


# Collect the array of class probability distributions. perform the argmax operation on each
# distribution to yield an array of indices, then count the number of occurrences of each index.
def classify_and_count(model_graph, graph_name, io_tensor_map, label_map, images,
                       image_size, batch_size, session_config, device_name):

  def decode_and_preprocess_for_eval(image):
    image = decode_jpeg(image)
    return inception_preprocessing.preprocess_for_eval(
      image, image_size, image_size, central_fraction=None)

  num_frames = len(images)
  num_batches = int(np.ceil(np.divide(num_frames, batch_size)))
  num_labels = len(label_map)

  print('Analyzing ' + str(num_frames) + ' video frames')

  class_id_counts = {class_id: 0 for class_id in range(num_labels)}

  start = timeit.default_timer()
  
  with tf.Session(graph=model_graph, config=session_config) as sess, tf.device(device_name):
    input_placeholder = sess.graph.get_tensor_by_name(
      '{}/{}'.format(graph_name, io_tensor_map['input_tensor_name']))
    image_size_placeholder = sess.graph.get_tensor_by_name(
      '{}/{}'.format(graph_name, io_tensor_map['image_size_tensor_name']))
    predictions = sess.graph.get_tensor_by_name(
      '{}/{}'.format(graph_name, io_tensor_map['output_tensor_name']))

    # perform non-maximum suppression to extract only the most likely class to be counted
    classifications = tf.argmax(predictions, axis=1)
    # identify unique occurrences of class_ids and count the number of each occurrence.
    unique_classifications, indexes, unique_counts = tf.unique_with_counts(classifications)
    
    images_placeholder = tf.placeholder(dtype=tf.string)

    image_slices = tf.data.Dataset.from_tensor_slices(images_placeholder)
    image_slices = image_slices.map(decode_and_preprocess_for_eval, batch_size)
    image_batches = image_slices.batch(batch_size)

    batch_iterator = image_batches.make_initializable_iterator()
    sess.run(batch_iterator.initializer, feed_dict={images_placeholder: images})

    next_batch = batch_iterator.get_next()

    for batch_num in range(num_batches):
      try:
        numpy_array = next_batch.eval(session=sess)
        unique_class_ids, _, counts = sess.run(
          [unique_classifications, indexes, unique_counts],
          {input_placeholder: numpy_array, image_size_placeholder: image_size})

        # use the identified unique class ids rather than the given list of class ids 
        # in case one or more classes is not ever predicted to be the most likely class
        # after all, we are processing only 'batch_size' images at a time
        for class_id in unique_class_ids:
          # squeeze because argwhere returns a 2D (or N+1 D) array, but we want a scalar
          # and we know that there will exist exactly one occurrence of each 'unique' id.
          class_id_count = np.squeeze(np.argwhere(unique_class_ids == class_id))
          class_id_counts[class_id] += counts[class_id_count]
      except tf.errors.OutOfRangeError:
        break

  stop = timeit.default_timer()
  print_processing_duration(start, stop, 'Video analysis time')

  return class_id_counts


def interrupt_handler(signal_number, _):
  tf.logging.info(
    'Received interrupt signal (%d). Unsetting CUDA_VISIBLE_DEVICES environment variable.', signal_number)
  os.unsetenv('CUDA_VISIBLE_DEVICES')
  sys.exit(0)


def main():
  tf.logging.set_verbosity(tf.logging.INFO)

  if FLAGS.cpu_only:
    device_name = '/cpu:0'

    tf.logging.info('Setting CUDA_VISIBLE_DEVICES environment variable to None.')
    os.putenv('CUDA_VISIBLE_DEVICES', '')

    session_config = None
  else:
    device_name = '/gpu:' + str(FLAGS.gpu_device_num)

    gpu_options = tf.GPUOptions(allow_growth=True,
                                per_process_gpu_memory_fraction=FLAGS.gpu_memory_fraction)
    session_config = tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options)

    tf.logging.info('Setting CUDA_VISIBLE_DEVICES environment variable to {}.'.format(
      FLAGS.gpu_device_num))
    os.putenv('CUDA_VISIBLE_DEVICES', str(FLAGS.gpu_device_num))

  signal.signal(signal.SIGINT, interrupt_handler)

  start = timeit.default_timer()

  class_count_map = {}

  labels = load_labels(FLAGS.labels_path)
  io_tensors = load_tensor_names(FLAGS.io_tensor_path)
  model_graph, graph_name = load_model(FLAGS.model_path)
  image_size = 299  # TODO: replace with fetch from inception module

  data_dir_subpaths = sorted([path.join(FLAGS.data_dir, data_subdir)
                       for data_subdir in os.listdir(FLAGS.data_dir)])

  if all([path.isfile(subpath) for subpath in data_dir_subpaths]):
    images = np.array([tf.gfile.GFile(subpath, 'rb').read()
                       for subpath in data_dir_subpaths])
    
    class_counts = classify_and_count(
      model_graph, graph_name, io_tensors, labels, images, image_size, FLAGS.batch_size,
      session_config, device_name)

    print('class_counts: ' + str(class_counts))

    ratio = float(len(images) - class_counts[0]) / len(images)

    print(path.basename(data_dir_subpaths[0][:-12]) + ' work_zone ratio: ' + str(ratio))
  elif all([path.isdir(subpath) for subpath in data_dir_subpaths]):
    for subpath in data_dir_subpaths:
      data_subdir_filepaths = sorted([path.join(subpath, file)
                                  for file in os.listdir(subpath)])

      if all([path.isfile(filepath) for filepath in data_subdir_filepaths]):
        images = np.array([tf.gfile.GFile(filepath, 'rb').read()
                           for filepath in data_subdir_filepaths])

        class_counts = classify_and_count(
          model_graph, graph_name, io_tensors, labels, images, image_size, FLAGS.batch_size,
          session_config, device_name)
    
        print('class_counts: ' + str(class_counts))
    
        ratio = float(len(images) - class_counts[0]) / len(images)
        video_name = path.basename(data_subdir_filepaths[0][:-12])
        class_count_map[video_name] = ratio
        print(video_name + ' work_zone ratio: ' + str(ratio))
      
  else:
    pass

  stop = timeit.default_timer()
  print_processing_duration(start, stop, 'Total analysis time')

  sorted_class_count_map = sorted(class_count_map.items(), key=operator.itemgetter(1))
  print(str(sorted_class_count_map))

  output_file_path = path.join(FLAGS.report_dir, 'work_zone_ratios_sorted_in_ascending_order.csv')
  with open(output_file_path, 'w') as output_file:
    output_file.write('Video Name,Work Zone Ratio\n')
    for (video_name, work_zone_ratio) in sorted_class_count_map:
      output_file.write('{},{}\n'.format(video_name, work_zone_ratio))

if __name__ == '__main__':
  main()