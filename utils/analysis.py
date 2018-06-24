import logging
from multiprocessing import Process
import numpy as np
from subprocess import PIPE, Popen
import tensorflow as tf


class VideoAnalyzer(Process):
  def __init__(
      self, frame_shape, num_frames, num_classes, batch_size, model_input_size,
      model_path, device_type, device_count, cpu_count, node_names_map,
      gpu_memory_fraction, extract_timestamps, timestamp_x, timestamp_y,
      timestamp_height, timestamp_max_width, crop, crop_x, crop_y,
      ffmpeg_command, child_interrupt_queue, result_queue, name):
    super(VideoAnalyzer, self).__init__(name=name)

    graph_def = tf.GraphDef()

    with open(model_path, 'rb') as file:
      graph_def.ParseFromString(file.read())

    self.input_node, self.output_node = tf.import_graph_def(
      graph_def, return_elements=[node_names_map['input_node_name'],
                                  node_names_map['output_node_name']])

    self.device_type = device_type

    if self.device_type == 'gpu':
      gpu_options = tf.GPUOptions(
        allow_growth=True,
        per_process_gpu_memory_fraction=gpu_memory_fraction)
      
      self.session_config = tf.ConfigProto(allow_soft_placement=True,
                                           gpu_options=gpu_options)
    else:
      self.session_config = None

    self.frame_shape = frame_shape

    frame_dataset = tf.data.Dataset.from_generator(
      self._generate_frames, tf.uint8, tf.TensorShape(self.frame_shape))

    self.model_input_size = model_input_size

    self.device_count = device_count

    self.cpu_count = cpu_count

    frame_dataset = frame_dataset.map(self._preprocess_frames,
                                      self._get_num_parallel_calls())

    self.batch_size = batch_size

    frame_dataset = frame_dataset.batch(self.batch_size)

    frame_dataset = frame_dataset.prefetch(self.batch_size)

    self.next_batch = frame_dataset.make_one_shot_iterator().get_next()

    self.prob_array = np.ndarray((num_frames, num_classes), dtype=np.float32)
    
    #### frame generator variables ####
    self.extract_timestamps = extract_timestamps

    if self.extract_timestamps:
      self.ti = 0

      self.tx = timestamp_x
      self.ty = timestamp_y
      self.th = timestamp_height
      self.tw = timestamp_max_width

      if crop:
        self.tx -= crop_x
        self.ty -= crop_y

      self.timestamp_array = np.ndarray(
        (self.th * num_frames, self.tw, self.frame_shape[-1]), dtype=np.uint8)
    else:
      self.timestamp_array = None

    self.ffmpeg_command = ffmpeg_command

    self.child_interrupt_queue = child_interrupt_queue

    self.result_queue = result_queue

  # feed the tf.data input pipeline one image at a time and, while we're at it,
  # extract timestamp overlay crops for later mapping to strings.
  def _generate_frames(self):
    logging.debug('opening video frame pipe')

    frame_string_len = 1
    
    for dim in self.frame_shape:
      frame_string_len *= dim

    base_two_exp = 2

    while base_two_exp < frame_string_len:
      base_two_exp *= 2

    frame_pipe = Popen(self.ffmpeg_command, stdout=PIPE, stderr=PIPE,
                             bufsize=self.batch_size * base_two_exp)

    logging.debug('video frame pipe opened with pid: {}'.format(
      frame_pipe.pid))

    while True:
      try:
        try:
          _ = self.child_interrupt_queue.get_nowait()
          logging.warning('closing video frame pipe following interrupt signal')
          frame_pipe.stdout.close()
          frame_pipe.stderr.close()
          frame_pipe.terminate()
          frame_pipe = None
          return
        except:
          pass

        frame_string = frame_pipe.stdout.read(
          frame_string_len)

        if not frame_string:
          logging.debug('closing video frame pipe following end of stream')
          frame_pipe.stdout.close()
          frame_pipe.stderr.close()
          frame_pipe.terminate()
          frame_pipe = None
          return

        frame_array = np.fromstring(frame_string, dtype=np.uint8)
        frame_array = np.reshape(frame_array, 
                                       self.frame_shape)

        if self.extract_timestamps:
          self.timestamp_array[self.th * self.ti:self.th * (self.ti + 1)] = \
            frame_array[self.ty:self.ty + self.th, 
            self.tx:self.tx + self.tw]
          self.ti += 1

        yield frame_array
      except Exception as e:
        logging.error(
          'met an unexpected error after processing {} frames.'.format(self.ti))
        logging.error(e)
        logging.error(
          'ffmpeg reported:\n{}'.format(frame_pipe.stderr.readlines()))
        logging.debug('closing video frame pipe following raised exception')
        frame_pipe.stdout.close()
        frame_pipe.stderr.close()
        frame_pipe.terminate()
        logging.debug('raising exception to caller.')
        raise e

  def _preprocess_frames(self, image):
    if image.dtype != tf.float32:
      image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    if image.shape[0] != self.model_input_size or \
            image.shape[1] != self.model_input_size:
      image = tf.expand_dims(image, 0)
      image = tf.image.resize_bilinear(
        image, [self.model_input_size, self.model_input_size], 
        align_corners=False)
      image = tf.squeeze(image, [0])
    image = tf.subtract(image, 0.5)
    image = tf.multiply(image, 2.0)

    return image

  # assumed user specification of numperdeviceprocesses has been validated,
  # to be <= cpu cores when in --cpuonly mode
  def _get_num_parallel_calls(self):
    if self.device_type == 'gpu':
      return int(self.cpu_count / self.device_count)
    else:
      if self.device_count == 1:
        return self.cpu_count - 1
      elif self.device_count == self.cpu_count:
        return self.cpu_count
      else:
        return int((self.cpu_count - self.device_count) / self.device_count)

  def run(self):
    try:
      logging.info('started inference.')
      
      count = 0

      with tf.device('/cpu:0') if self.device_type == 'cpu' else \
          tf.device(None):
        with tf.Session(config=self.session_config) as session:

          while True:
            try:
              frame_batch = session.run(self.next_batch)
              probs = session.run(self.output_node,
                                  {self.input_node: frame_batch})
              self.prob_array[count:count + probs.shape[0]] = probs
              count += probs.shape[0]
            except tf.errors.OutOfRangeError:
              logging.info('completed inference.')
              break

      self.result_queue.put((count, self.prob_array, self.timestamp_array))
      self.result_queue.close()

    # We depend on the used to empirically determine the max batch size
    # supported by their GPU given the model.
    except tf.errors.ResourceExhaustedError as ree:
      logging.warning('encountered a resource exhausted error.')
      logging.warning(ree)
      logging.error('will not re-attempt inference.')