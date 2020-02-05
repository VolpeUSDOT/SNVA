import json
import logging
from multiprocessing import Process
import numpy as np
import requests
from skimage import img_as_float32
from skimage.transform import resize
from subprocess import PIPE, Popen


class VideoAnalyzer(Process):
  def __init__(
      self, frame_shape, num_frames, num_classes, batch_size, model_input_size,
      model_path, device_type, num_processes_per_device, cpu_count,
      node_names_map, gpu_memory_fraction, extract_timestamps, timestamp_x,
      timestamp_y, timestamp_height, timestamp_max_width, crop, crop_x, crop_y,
      crop_width, crop_height, ffmpeg_command, child_interrupt_queue,
      result_queue, name):
    super(VideoAnalyzer, self).__init__(name=name)

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
  def generate_frame_batches(self):
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

        frame_string = self.frame_pipe.stdout.read(
          self.frame_string_len * self.batch_size)

        if not frame_string:
          logging.debug('closing video frame pipe following end of stream')
          self.frame_pipe.stdout.close()
          self.frame_pipe.stderr.close()
          self.frame_pipe.terminate()
          return

        frame_array = np.fromstring(frame_string, dtype=np.uint8)
        frame_array = np.reshape(frame_array, [-1] + self.frame_shape)

        if self.extract_timestamps:
          self.timestamp_array[
          self.th * self.ti:self.th * (self.ti + len(frame_array))] = \
            frame_array[:, self.ty:self.ty + self.th, self.tx:self.tx + self.tw]
          self.ti += len(frame_array)

        if self.crop:
          frame_array = frame_array[:, self.crop_y:self.crop_y + self.crop_height,
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

  def _preprocess_frame(self, image):
    image = img_as_float32(image)
    image = resize(image, (self.model_input_size, self.model_input_size))
    image -= .5
    image *= 2.
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

  def run(self):
    logging.info('started inference.')
    logging.debug('TF input frame shape == {}'.format(self.tensor_shape))

    count = 0

    headers = {'content-type': 'application/json'}
    served_model_fn = 'http://localhost:8501/v1/models/mobilenet_v2:predict'
    signature_name = 'serving_default'

    print('len: ', len(self.prob_array))
    for frame_batch in self.generate_frame_batches():
      # preprocess for CNN
      temp = np.ndarray(
        (len(frame_batch), self.model_input_size, self.model_input_size, 3))
      for i in range(len(frame_batch)):
        temp[i] = self._preprocess_frame(frame_batch[i])

      frame_batch = temp

      try:
        data = json.dumps({'signature_name': signature_name,
                           'instances': frame_batch.tolist()})
        json_response = requests.post(
          served_model_fn, data=data, headers=headers)
        response = json.loads(json_response.text)
        probs = [pred['probabilities'] for pred in response['predictions']]
        self.prob_array[count:count + len(probs)] = probs
        count += len(probs)
      except StopIteration:
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