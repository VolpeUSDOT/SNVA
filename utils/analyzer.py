import concurrent.futures as futures
import grpc
import json
import logging
import numpy as np
import requests
from skimage import img_as_float32
from skimage.transform import resize
from subprocess import PIPE, Popen
from tensorboard._vendor.tensorflow_serving.apis import predict_pb2
from tensorboard._vendor.tensorflow_serving.apis import prediction_service_pb2_grpc
import tensorflow as tf


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

    # TODO: parameterize tf serving info
    self.headers = {'content-type': 'application/json'}
    # self.host = 'localhost:8501'
    self.host = '0.0.0.0:8500'
    self.model_name = 'mobilenet_v2'
    self.served_model_fn = 'http://' + self.host + '/v1/models/' + self.model_name + ':predict'
    self.signature_name = 'serving_default'
    self.channel = grpc.insecure_channel(self.host)
    self.stub = prediction_service_pb2_grpc.PredictionServiceStub(self.channel)

  # feed the tf.data input pipeline one image at a time and, while we're at
  # it, extract timestamp overlay crops for later mapping to strings.
  def _generate_frames(self):
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

        yield frame_string
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

  # feed the tf.data input pipeline one image at a time and, while we're at
  # it, extract timestamp overlay crops for later mapping to strings.
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

  def _get_frame_probs_from_tf_serving(self, frame_string):
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
    frame_array =  np.expand_dims(frame_array, 0)

    data = json.dumps({'signature_name': self.signature_name,
                       'instances': [frame_array.tolist()]})
    response = requests.post(self.served_model_fn, data=data,
                             headers=self.headers)
    response = json.loads(response.text)

    # if more than one output (e.g. probabilities and logits) are available,
    # probabilities will have to be fetched from a nested dictionary
    # return [pred['probabilities'] for pred in response['predictions']]

    # if probabilities are the only output, they will be mapped directly to
    # 'predictions'
    return response['predictions']

  def _get_frame_probs_from_tf_serving_grpc(self, frame_string):
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
    frame_array =  frame_array[np.newaxis, :]  # batchify single frame

    request = predict_pb2.PredictRequest()
    request.model_spec.name = self.model_name
    request.model_spec.signature_name = self.signature_name
    request.inputs['input'].CopyFrom(tf.make_tensor_proto(
      frame_array, shape=frame_array.shape))

    # returns a PredictResponse object
    response = self.stub.Predict(request)

    #return a list constructed from the RepeatedScalarContainer float_val
    return response.outputs['probabilities'].float_val[:]

  def _get_frame_batch_probs_from_tf_serving(self, frame_batch_string):
    frame_batch_array = np.fromstring(frame_batch_string, dtype=np.uint8)
    frame_batch_array = np.reshape(frame_batch_array, [-1] + self.frame_shape)

    if self.extract_timestamps:
      self.timestamp_array[self.th * self.ti:self.th * self.ti] = \
        frame_batch_array[:, self.ty:self.ty + self.th,
        self.tx:self.tx + self.tw]
      self.ti += 1

    if self.crop:
      frame_batch_array = frame_batch_array[
                          self.crop_y:self.crop_y + self.crop_height,
                          self.crop_x:self.crop_x + self.crop_width]

    frame_batch_array = self._preprocess_frame_batch(frame_batch_array)

    data = json.dumps({'signature_name': self.signature_name,
                       'instances': frame_batch_array.tolist()})
    response = requests.post(self.served_model_fn, data=data,
                             headers=self.headers)
    response = json.loads(response.text)

    return response['predictions']

  def run(self):
    logging.info('started inference.')
    logging.debug('TF input frame shape == {}'.format(self.tensor_shape))

    count = 0

    logging.info('processing {} frames'.format(len(self.prob_array)))

    with futures.ThreadPoolExecutor(max_workers=10) as executor:
      future_probs = [executor.submit(
        self._get_frame_probs_from_tf_serving_grpc, frame)
        for frame in self._generate_frames()]

      for future in futures.as_completed(future_probs):
        probs = future.result()
        # if we are batching, our returned list will be 2D and we can count
        # the number of rows using len()
        self.prob_array[count:count + len(probs) if self.batch_size > 1 else 1] = probs

        count += len(probs) if self.batch_size > 1 else 1

    logging.info('completed inference with count {} and num_probs {}.'.format(count, self.prob_array.shape))

    return count, self.prob_array, self.timestamp_array

  def __del__(self):
    if self.frame_pipe.returncode is None:
      logging.debug(
        'video frame pipe with pid {} remained alive after being instructed to '
        'temrinate and had to be killed'.format(self.frame_pipe.pid))
      self.frame_pipe.kill()