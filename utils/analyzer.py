# import asyncio
import json
import logging
from threading import Thread
import numpy as np
import requests
from skimage import img_as_float32
from skimage.transform import resize
from subprocess import PIPE, Popen


class VideoAnalyzer:
  def __init__(
      self, frame_shape, num_frames, num_classes, batch_size, model_input_size,
      num_processes_per_device, cpu_count, extract_timestamps, timestamp_x,
      timestamp_y, timestamp_height, timestamp_max_width, crop, crop_x, crop_y,
      crop_width, crop_height, ffmpeg_command):
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

    self.headers = {'content-type': 'application/json'}
    self.served_model_fn = 'http://localhost:8501/v1/models/mobilenet_v2:predict'
    self.signature_name = 'serving_default'

  # feed the tf.data input pipeline one image at a time and, while we're at it,
  # extract timestamp overlay crops for later mapping to strings.
  def _generate_frame_batches(self):
    while True:
      try:
        frame_batch_string = self.frame_pipe.stdout.read(
          self.frame_string_len * self.batch_size)

        if not frame_batch_string:
          logging.debug('closing video frame pipe following end of stream')
          self.frame_pipe.stdout.close()
          self.frame_pipe.stderr.close()
          self.frame_pipe.terminate()
          return

        yield frame_batch_string
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

  # feed the tf.data input pipeline one image at a time and, while we're at it,
  # extract timestamp overlay crops for later mapping to strings.
  def _generate_preprocessed_frame(self):
    while True:
      try:
        frame_string = self.frame_pipe.stdout.read(
          self.frame_string_len)

        if not frame_string:
          logging.debug('closing video frame pipe following end of stream')
          self.frame_pipe.stdout.close()
          self.frame_pipe.stderr.close()
          self.frame_pipe.terminate()
          return

        frame_array = np.fromstring(frame_string, dtype=np.uint8)
        frame_array = np.reshape(frame_array, self.frame_shape)

        if self.extract_timestamps:
          self.timestamp_array[self.th * self.ti:self.th * self.ti] = \
            frame_array[self.ty:self.ty + self.th,
            self.tx:self.tx + self.tw]
          self.ti += 1

        if self.crop:
          frame_array = frame_array[self.crop_y:self.crop_y + self.crop_height,
                        self.crop_x:self.crop_x + self.crop_width]

        frame_array = self._preprocess_frame(frame_array)

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

  def _preprocess_frame_batch(self, frame_batch_string):
    frame_batch_array = np.fromstring(frame_batch_string, dtype=np.uint8)
    frame_batch_array = np.reshape(frame_batch_array, [-1] + self.frame_shape)

    if self.extract_timestamps:
      self.timestamp_array[
      self.th * self.ti:self.th * (self.ti + len(frame_batch_array))] = \
        frame_batch_array[:, self.ty:self.ty + self.th, self.tx:self.tx + self.tw]
      self.ti += len(frame_batch_array)

    if self.crop:
      frame_batch_array = frame_batch_array[:, self.crop_y:self.crop_y + self.crop_height,
                    self.crop_x:self.crop_x + self.crop_width]
    temp_array = np.ndarray(
      (len(frame_batch_array), self.model_input_size, self.model_input_size, 3))

    for i in range(len(frame_batch_array)):
      temp_array[i] = self._preprocess_frame(frame_batch_array[i])

    return temp_array

  def _get_probs_from_tf_serving(self, frame_list):
    data = json.dumps({'signature_name': self.signature_name,
                       'instances': frame_list})
    response = requests.post(self.served_model_fn, data=data,
                             headers=self.headers)
    response = json.loads(response.text)

    # if more than one output (e.g. probabilities and logits) are available,
    # probabilities will have to be fetched from a nested dictionary
    # probs = [pred['probabilities'] for pred in response['predictions']]
    # self.prob_array[count:count + len(probs)] = probs

    # if probabilities are the only output, they will be mapped directly to
    # 'predictions'
    return response['predictions']

  def run(self):
    logging.info('started inference.')
    logging.debug('TF input frame shape == {}'.format(self.tensor_shape))

    count = 0

    print('len: ', len(self.prob_array))

    for preprocessed_frame in self._generate_preprocessed_frame():
      try:
        # nest the single frame in a list to simulate a batch
        probs = self._get_probs_from_tf_serving([preprocessed_frame.tolist()])
        self.prob_array[count:count + len(probs)] = probs
        count += len(probs)
      except StopIteration:
        logging.info('completed inference.')
        break

    return count, self.prob_array, self.timestamp_array

  def __del__(self):
    if self.frame_pipe.returncode is None:
      logging.debug(
        'video frame pipe with pid {} remained alive after being instructed to '
        'temrinate and had to be killed'.format(self.frame_pipe.pid))
      self.frame_pipe.kill()