from concurrent import futures
from grpc import insecure_channel
import logging
import numpy as np
from skimage import img_as_float32
from skimage.transform import resize
from subprocess import PIPE, Popen
from tensorboard._vendor.tensorflow_serving.apis.predict_pb2 \
  import PredictRequest  #TODO or not todo, find an alternative source of TF serving api
from tensorboard._vendor.tensorflow_serving.apis.prediction_service_pb2_grpc \
  import PredictionServiceStub
import tensorflow as tf


class VideoAnalyzer:
  def __init__(
      self, frame_shape, num_frames, num_classes, batch_size, model_input_size,
      max_num_threads, should_extract_timestamps, timestamp_x, timestamp_y,
      timestamp_height, timestamp_max_width, should_crop, crop_x, crop_y,
      crop_width, crop_height, ffmpeg_command, model_name, signature_name,
      host):
    #### frame generator variables ####
    self.frame_shape = frame_shape
    self.should_crop = should_crop

    if self.should_crop:
      self.crop_x = crop_x
      self.crop_y = crop_y
      self.crop_width = crop_width
      self.crop_height = crop_height

    self.should_extract_timestamps = should_extract_timestamps

    if self.should_extract_timestamps:
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
    self.max_num_threads = max_num_threads
    self.batch_size = batch_size
    self.ffmpeg_command = ffmpeg_command
    self.num_classes = num_classes
    self.prob_array = np.ndarray(
      (num_frames, self.num_classes), dtype=np.float32)
    self.num_frames_processed = 0

    self.model_name = model_name
    self.signature_name = signature_name
    self.service_stub = PredictionServiceStub(insecure_channel(host))

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

  def _preprocess_frame(self, frame):
    frame = img_as_float32(frame)
    frame = resize(frame, (self.model_input_size, self.model_input_size))
    frame -= .5
    frame *= 2.
    return frame
  
  def _preprocess_frame_batch(self, frame_batch):
    temp = np.ndarray(
      (len(frame_batch), self.model_input_size, self.model_input_size, 3))
    for i in range(len(frame_batch)):
      temp[i] = self._preprocess_frame(frame_batch[i])
    return temp

  def _produce_grpc_request(self):
    num_processed = 0

    while True:
      try:
        frame = self.frame_pipe.stdout.read(self.frame_string_len)

        if not frame:
          logging.debug('closing video frame pipe following end of stream')
          self.frame_pipe.stdout.close()
          self.frame_pipe.stderr.close()
          self.frame_pipe.terminate()
          return

        frame = np.fromstring(frame, dtype=np.uint8)

        frame = np.reshape(frame, self.frame_shape)

        if self.should_extract_timestamps:
          self.timestamp_array[self.th * self.ti:self.th * (self.ti + 1)] = \
            frame[self.ty:self.ty + self.th,
            self.tx:self.tx + self.tw]
          self.ti += 1

        if self.should_crop:
          frame = frame[self.crop_y:self.crop_y + self.crop_height,
                  self.crop_x:self.crop_x + self.crop_width]

        frame = self._preprocess_frame(frame)

        frame = np.expand_dims(frame, axis=0)  # batchify single frame

        request = PredictRequest()
        request.model_spec.name = self.model_name
        request.model_spec.signature_name = self.signature_name
        request.inputs['input'].CopyFrom(
          tf.make_tensor_proto(frame, shape=frame.shape))

        num_processed += 1

        yield request, num_processed - 1
      except Exception as e:
        logging.error(
          'met an unexpected error after processing {} frames.'.format(num_processed))
        logging.error(e)
        logging.error(
          'ffmpeg reported:\n{}'.format(self.frame_pipe.stderr.readlines()))
        logging.debug('closing video frame pipe following raised exception')
        self.frame_pipe.stdout.close()
        self.frame_pipe.stderr.close()
        self.frame_pipe.terminate()
        logging.debug('raising exception to caller.')
        raise e

  def _consume_grpc_request(self, request, index):
    #TODO: validate the response
    response = self.service_stub.Predict(request)
    self.prob_array[index] = response.outputs['probabilities'].float_val[:]
    return 1  # report one additional frame processed to caller

  def _produce_batch_grpc_request(self):
    num_processed = 0

    while True:
      try:
        frame = self.frame_pipe.stdout.read(
          self.frame_string_len * self.batch_size)

        if not frame:
          logging.debug('closing video frame pipe following end of stream')
          self.frame_pipe.stdout.close()
          self.frame_pipe.stderr.close()
          self.frame_pipe.terminate()
          return

        frame = np.fromstring(frame, dtype=np.uint8)
        frame = np.reshape(frame, [-1] + self.frame_shape)

        if self.should_extract_timestamps:
          self.timestamp_array[self.th * self.ti:self.th * (
            self.ti + frame.shape[0])] = \
            np.reshape(frame[:, self.ty:self.ty + self.th,
            self.tx:self.tx + self.tw], (-1,) + self.timestamp_array.shape[1:])
          self.ti += frame.shape[0]

        if self.should_crop:
          frame = frame[:, self.crop_y:self.crop_y + self.crop_height,
                  self.crop_x:self.crop_x + self.crop_width]

        frame = self._preprocess_frame_batch(frame)

        request = PredictRequest()
        request.model_spec.name = self.model_name
        request.model_spec.signature_name = self.signature_name
        request.inputs['input'].CopyFrom(
          tf.make_tensor_proto(frame, shape=frame.shape, dtype=tf.float32))

        num_processed += frame.shape[0]

        yield request, num_processed - frame.shape[0]  # index of prob_array
      except Exception as e:
        logging.error(
          'met an unexpected error after processing {} frames.'.format(num_processed))
        logging.error(e)
        logging.error(
          'ffmpeg reported:\n{}'.format(self.frame_pipe.stderr.readlines()))
        logging.debug('closing video frame pipe following raised exception')
        self.frame_pipe.stdout.close()
        self.frame_pipe.stderr.close()
        self.frame_pipe.terminate()
        logging.debug('raising exception to caller.')
        raise e

  def _consume_batch_grpc_request(self, request, index):
    #TODO: validate the response
    response = self.service_stub.Predict(request)
    response = response.outputs['probabilities'].float_val[:]
    response = np.array(response, dtype=np.float32)
    response = np.reshape(response, (-1, self.num_classes))

    self.prob_array[index:index + response.shape[0]] = response

    return response.shape[0]  # report num frames processed to caller

  def run(self):
    logging.info('started inference on {} frames'.format(
      self.prob_array.shape[0]))

    with futures.ThreadPoolExecutor(
        max_workers=self.max_num_threads) as executor:
      future_probs = [executor.submit(
        self._consume_batch_grpc_request, request, index)
        for request, index in self._produce_batch_grpc_request()]

      for future in futures.as_completed(future_probs):
        num_frames_processed = future.result()
        self.num_frames_processed += num_frames_processed

    logging.info('completed inference on {} frames.'.format(
      self.num_frames_processed))

    return self.num_frames_processed, self.prob_array, self.timestamp_array

  def __del__(self):
    if self.frame_pipe.returncode is None:
      logging.debug(
        'video frame pipe with pid {} remained alive after being instructed to '
        'temrinate and had to be killed'.format(self.frame_pipe.pid))
      self.frame_pipe.kill()