import logging
from multiprocessing import Process
import numpy as np
from subprocess import PIPE, Popen
import tensorflow as tf

# Script uses tf2.0 compatability library
# Need to move to 2.0 proper, guide is here: https://www.tensorflow.org/guide/migrate

# enable eager execution throughout
#tf.compat.v1.enable_eager_execution()

class VideoAnalyzer(Process):
  def __init__(
      self, frame_shape, num_frames, num_classes, batch_size, model_input_size,
      model_path, device_type, num_processes_per_device, cpu_count,
      node_names_map, gpu_memory_fraction, extract_timestamps, timestamp_x,
      timestamp_y, timestamp_height, timestamp_max_width, crop, crop_x, crop_y,
      crop_width, crop_height, ffmpeg_command, child_interrupt_queue,
      result_queue, name):
    super(VideoAnalyzer, self).__init__(name=name)
    
    #### TF session variables ####
    graph_def = tf.compat.v1.GraphDef()

    with open(model_path, 'rb') as file:
      graph_def.ParseFromString(file.read())

    self.input_node, self.output_node = tf.import_graph_def(
      graph_def, return_elements=[node_names_map['input_node_name'],
                                  node_names_map['output_node_name']])

    self.device_type = device_type
    
    if self.device_type == 'gpu':
      gpu_options = tf.compat.v1.GPUOptions(
        allow_growth=True,
        per_process_gpu_memory_fraction=gpu_memory_fraction)

      self.session_config = tf.compat.v1.ConfigProto(allow_soft_placement=True,
                                           gpu_options=gpu_options)
    else:
      self.session_config = None

    #### frame generator variables ####
    self.frame_shape = frame_shape
    self.crop = crop

    if self.crop:
      self.crop_x = crop_x
      self.crop_y = crop_y
      self.crop_width = crop_width
      self.crop_height = crop_height

      self.tensor_shape = [
        self.crop_height, self.crop_width, self.frame_shape[-1]]
    else:
      self.tensor_shape = self.frame_shape

    self.extract_timestamps = extract_timestamps

    if self.extract_timestamps:
      self.ti = 0

      self.tx = timestamp_x
      self.ty = timestamp_y
      self.th = timestamp_height
      self.tw = timestamp_max_width

      self.timestamp_array = np.ndarray(
        (self.th * num_frames, self.tw, self.frame_shape[-1]), dtype=np.uint8)
    else:
      self.timestamp_array = None

    self.model_input_size = model_input_size
    self.num_processes_per_device = num_processes_per_device
    self.cpu_count = cpu_count
    self.batch_size = batch_size
    self.ffmpeg_command = ffmpeg_command

    self.prob_array = np.ndarray((num_frames, num_classes), dtype=np.float32)

    self.child_interrupt_queue = child_interrupt_queue
    self.result_queue = result_queue

    logging.debug('opening video frame pipe')

    self.frame_string_len = 1

    for dim in self.frame_shape:
      self.frame_string_len *= dim

    buffer_scale = 2

    while buffer_scale < self.frame_string_len:
      buffer_scale *= 2

    self.frame_pipe = Popen(self.ffmpeg_command, stdout=PIPE, stderr=PIPE,
                            bufsize=2 * self.batch_size * buffer_scale)

    logging.debug('video frame pipe created with pid: {}'.format(
      self.frame_pipe.pid))

  # feed the tf.data input pipeline one image at a time and, while we're at it,
  # extract timestamp overlay crops for later mapping to strings.
  def generate_frames(self):
    while True:
      try:
        try:
          _ = self.child_interrupt_queue.get_nowait()
          logging.warning('closing video frame pipe following interrupt signal')
          self.frame_pipe.stdout.close()
          self.frame_pipe.stderr.close()
          self.frame_pipe.terminate()
          return
        except:
          pass

        frame_string = self.frame_pipe.stdout.read(self.frame_string_len)

        if not frame_string:
          logging.debug('closing video frame pipe following end of stream')
          self.frame_pipe.stdout.close()
          self.frame_pipe.stderr.close()
          self.frame_pipe.terminate()
          return

        frame_array = np.fromstring(frame_string, dtype=np.uint8)
        frame_array = np.reshape(frame_array, self.frame_shape)

        if self.extract_timestamps:
          self.timestamp_array[self.th * self.ti:self.th * (self.ti + 1)] = \
            frame_array[self.ty:self.ty + self.th, self.tx:self.tx + self.tw]
          self.ti += 1

        if self.crop:
          frame_array = frame_array[self.crop_y:self.crop_y + self.crop_height,
                                    self.crop_x:self.crop_x + self.crop_width]

        yield frame_array
      except Exception as e:
        logging.error(
          'met an unexpected error after processing {} frames.'.format(self.ti))
        logging.error(e)
        logging.error(
          'ffmpeg reported:\n{}'.format(self.frame_pipe.stderr.readlines()))
        logging.debug('closing video frame pipe following raised exception')
        self.frame_pipe.stdout.close()
        self.frame_pipe.stderr.close()
        self.frame_pipe.terminate()
        logging.debug('raising exception to caller.')
        raise e

  def _preprocess_frames(self, image):
    if image.dtype != tf.float32:
      image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    if image.shape[0] != self.model_input_size or \
            image.shape[1] != self.model_input_size:
      image = tf.expand_dims(image, 0)
      image = tf.compat.v1.image.resize_bilinear(
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
      return int(self.cpu_count / self.num_processes_per_device)
    else:
      if self.num_processes_per_device == 1:
        return self.cpu_count - 1
      elif self.num_processes_per_device == self.cpu_count:
        return self.cpu_count
      else:
        return int((self.cpu_count - self.num_processes_per_device) / 
                   self.num_processes_per_device)

  # tf function decorator seems to be necessary for eager evaluation
  #@tf.function
  def run(self):
    logging.info('started inference.')
    logging.debug('TF input frame shape == {}'.format(self.tensor_shape))
    #builder = tf.compat.v1.profiler.ProfileOptionBuilder
    #opts = builder(builder.time_and_memory()).order_by('micros').with_timeline_output("/home/bsumner/Documents/foo.timeline").build()
    count = 0
    with tf.device('/cpu:0') if self.device_type == 'cpu' else \
        tf.device(None):
      with tf.compat.v1.Session(config=self.session_config) as session:
        #profiler = tf.compat.v1.profiler.Profiler(session.graph)
        frame_dataset = tf.data.Dataset.from_generator(
          self.generate_frames, tf.uint8, tf.TensorShape(self.tensor_shape))
        frame_dataset = frame_dataset.map(self._preprocess_frames,
                                          self._get_num_parallel_calls())
        frame_dataset = frame_dataset.batch(self.batch_size)
        frame_dataset = frame_dataset.prefetch(self.batch_size)
        next_batch = tf.compat.v1.data.make_one_shot_iterator(frame_dataset).get_next()

        # TODO use of this iterator should be replaced by eager evaluation loop through the dataset
        #for batch in frame_dataset:
        while True:
          try:
            frame_batch = session.run(next_batch)
            probs = session.run(self.output_node,
                                {self.input_node: frame_batch})
            #profiler.profile_graph(options=opts)
            self.prob_array[count:count + probs.shape[0]] = probs
            count += probs.shape[0]
          except tf.errors.OutOfRangeError:
            logging.info('completed inference.')
            break

    self.result_queue.put((count, self.prob_array, self.timestamp_array))
    self.result_queue.close()

  def __del__(self):
    if self.frame_pipe.returncode is None:
      logging.debug(
        'video frame pipe with pid {} remained alive after being instructed to '
        'temrinate and had to be killed'.format(self.frame_pipe.pid))
      self.frame_pipe.kill()
