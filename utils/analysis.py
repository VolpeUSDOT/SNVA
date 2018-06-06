import logging
import os
import tensorflow as tf
from time import time
from utils.io import IO


def load_model(
    model_path, io_node_names_path, device_type, gpu_memory_fraction):
  graph_def = tf.GraphDef()

  with open(model_path, 'rb') as file:
    graph_def.ParseFromString(file.read())

  node_names_map = IO.read_node_names(io_node_names_path)

  input_node, output_node = tf.import_graph_def(
    graph_def, return_elements=[node_names_map['input_node_name'],
                                node_names_map['output_node_name']])

  if device_type == 'gpu':
    gpu_options = tf.GPUOptions(
      allow_growth=True, per_process_gpu_memory_fraction=gpu_memory_fraction)
    session_config = tf.ConfigProto(
      allow_soft_placement=True, gpu_options=gpu_options)
  else:
    session_config = None

  return {'session_config': session_config,
          'input_node': input_node,
          'output_node': output_node}


def preprocess_for_inception(image, image_size):
  if image.dtype != tf.float32:
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
  if image.shape[0] != image_size or image.shape[1] != image_size:
    image = tf.expand_dims(image, 0)
    image = tf.image.resize_bilinear(
      image, [image_size, image_size], align_corners=False)
    image = tf.squeeze(image, [0])
  image = tf.subtract(image, 0.5)
  image = tf.multiply(image, 2.0)

  return image


def analyze_video(video_frame_generator, video_frame_shape, batch_size,
                  session_config, input_node, output_node, preprocessing_fn,
                  prob_array, device_type, device_count):
  video_frame_dataset = tf.data.Dataset.from_generator(
    video_frame_generator, tf.uint8, tf.TensorShape(list(video_frame_shape)))

  video_frame_dataset = video_frame_dataset.map(
    preprocessing_fn, num_parallel_calls=int(os.cpu_count() / device_count))

  video_frame_dataset = video_frame_dataset.batch(batch_size)

  video_frame_dataset = video_frame_dataset.prefetch(batch_size)

  next_batch = video_frame_dataset.make_one_shot_iterator().get_next()

  logging.debug('constructed image dataset pipeline')

  attempts = 0

  while attempts < 3:
    try:
      logging.info('started inference.')

      start = time()

      with tf.device('/cpu:0') if device_type == 'cpu' else tf.device(None):
        with tf.Session(config=session_config) as session:
          total_num_probs = 0

          while True:
            try:
              video_frame_batch = session.run(next_batch)
              probs = session.run(output_node, {input_node: video_frame_batch})
              num_probs = probs.shape[0]
              prob_array[total_num_probs:total_num_probs + num_probs] = probs
              total_num_probs += num_probs
            except tf.errors.OutOfRangeError:
              logging.info('completed inference.')
              break

      end = time() - start

      processing_duration = IO.get_processing_duration(
        end, 'processed {} frames in'.format(total_num_probs))
      logging.info(processing_duration)

      return total_num_probs
    # TODO: permanently update batch size in main so future runs aren't delayed.
    # in the limit, we should only detect OOM once and update a shared batch
    # size variable to benefit all future videos within the current app run.
    except tf.errors.ResourceExhaustedError as ree:
      logging.warning('encountered a resource exhausted error.')
      logging.warning(ree)
      attempts += 1

      # If an error occurs, retry up to two times
      if attempts < 3:
        batch_size = int(batch_size / 2)
        logging.debug('will re-attempt inference with a new batch size of '
                      '{}'.format(batch_size))
      else:
        logging.error('will not re-attempt inference.')
        break