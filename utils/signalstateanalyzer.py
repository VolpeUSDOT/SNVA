from concurrent import futures
from grpc import insecure_channel
import logging
import numpy as np
from skimage import img_as_float32
from skimage.transform import resize
from subprocess import PIPE, Popen
from tensorflow_serving.apis import predict_pb2, prediction_service_pb2_grpc
import tensorflow as tf


class SignalVideoAnalyzer:
  def __init__(
      self, frame_shape, num_frames, num_classes, batch_size, model_name,
      model_signature_name, model_server_host, model_input_size,
      should_extract_timestamps, timestamp_x, timestamp_y, timestamp_height,
      timestamp_max_width, should_crop, crop_x, crop_y, crop_width,
      crop_height, ffmpeg_command, max_num_threads):
    #### frame generator variables ####
    self.frame_shape = frame_shape
    self.should_crop = should_crop
    self.printOneTime = True

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
    self.signal_maps = []
    self.num_frames_processed = 0

    self.model_name = model_name
    self.signature_name = model_signature_name
    max_msg_length = 100* 1024 * 1024
    options = [('grpc.max_message_length', max_msg_length), ('grpc.max_receive_message_length', max_msg_length)]
    channel = insecure_channel(model_server_host, options=options)
    self.service_stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)

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

        request = predict_pb2.PredictRequest()
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
    counts = response.outputs['num_detections'].float_val[:]
    counts = np.array(counts, dtype=np.float32)
    classes = tf.make_ndarray(response.outputs['detection_classes'])
    scores = tf.make_ndarray(response.outputs['detection_scores'])
    boxes = tf.make_ndarray(response.outputs['detection_boxes'])
    num_detections = int(counts[0])
    frame_scores = scores[0]
    frame_scores = frame_scores[:num_detections]
    frame_classes = classes[0]
    frame_classes = frame_classes[:num_detections]
    frame_boxes = boxes[0]
    frame_boxes = frame_boxes[:num_detections]
    frame_map = {'num_detections': num_detections, 'detection_classes': frame_classes, 'detection_scores': frame_scores, 'detection_boxes': frame_boxes }
    self.signal_maps.insert(index, frame_map)
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

        request = predict_pb2.PredictRequest()
        request.model_spec.name = self.model_name
        request.model_spec.signature_name = self.signature_name
        request.inputs['inputs'].CopyFrom(
          tf.make_tensor_proto(frame, shape=frame.shape, dtype=tf.uint8))

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
    counts = response.outputs['num_detections'].float_val[:]
    counts = np.array(counts, dtype=np.float32)
    classes = tf.make_ndarray(response.outputs['detection_classes'])
    scores = tf.make_ndarray(response.outputs['detection_scores'])
    boxes = tf.make_ndarray(response.outputs['detection_boxes'])
    for i in range(counts.shape[0]):
      num_detections = int(counts[i])
      frame_scores = scores[i]
      frame_scores = frame_scores[:num_detections]
      frame_classes = classes[i]
      frame_classes = frame_classes[:num_detections]
      frame_boxes = boxes[i]
      frame_boxes = frame_boxes[:num_detections]
      frame_map = {'num_detections': num_detections, 'detection_classes': frame_classes, 'detection_scores': frame_scores, 'detection_boxes': frame_boxes }
      self.signal_maps.insert(index + i, frame_map)

    return counts.shape[0]  # report num frames processed to caller

  def run(self):
    #logging.info('started inference on {} frames'.format(
    #  self.prob_array.shape[0]))

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

    return self.num_frames_processed, self.signal_maps, self.timestamp_array

  def __del__(self):
    if self.frame_pipe.returncode is None:
      logging.debug(
        'video frame pipe with pid {} remained alive after being instructed to '
        'temrinate and had to be killed'.format(self.frame_pipe.pid))
      self.frame_pipe.kill()