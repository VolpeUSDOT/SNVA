import logging
from logging.handlers import QueueHandler
from multiprocessing import Queue
import numpy as np
import os
import signal
from time import time
from utils.analyzer import VideoAnalyzer
from utils.signalstateanalyzer import SignalVideoAnalyzer
from utils.event import Trip
from utils.io import IO
from utils.timestamp import Timestamp

path = os.path


def configure_logger(log_level, log_queue):
  root_logger = logging.getLogger(__name__)
  if root_logger.hasHandlers():  # Clear any handlers to avoid duplicate entries
    root_logger.handlers.clear()
  root_logger.setLevel(log_level)
  queue_handler = QueueHandler(log_queue)
  root_logger.addHandler(queue_handler)


def should_crop(frame_width, frame_height, do_crop, crop_width, crop_height,
                crop_x, crop_y):
  if do_crop:
    if all(
        [frame_width >= crop_width > 0, frame_height >= crop_height > 0,
         frame_width > crop_x >= 0, frame_height > crop_y >= 0]):
      logging.info(
        'video frames will be cropped from [w={}:h={}:x={}:y={}]'.format(
          crop_width, crop_height, crop_x, crop_y))
      return True
    else:
      raise ValueError(
        'video frames cannot be cropped from [w={}:h={}:x={}:y={}] because the '
        'video dimensions are [w={}:h={}]'.format(
          crop_width, crop_height, crop_x, crop_y, frame_width, frame_height))
  else:
    logging.debug('video frames will not be cropped')
    return False


def should_extract_timestamps(
    frame_width, frame_height, do_extract_timestamps, timestamp_max_width,
    timestamp_height, timestamp_x, timestamp_y):
  if do_extract_timestamps:
    if all([frame_width >= timestamp_x + timestamp_max_width > 0,
            frame_height >= timestamp_y + timestamp_height > 0]):
      logging.info(
        'timestamps will be extracted from [w={}:h={}:x={}:y={}]'.format(
          timestamp_max_width, timestamp_height, timestamp_x, timestamp_y))
      return True
    else:
      raise ValueError(
        'timestamps cannot be extracted from [w={}:h={}:x={}:y={}] because the '
        'video dimensions are [w={}:h={}]'.format(
          timestamp_max_width, timestamp_height, timestamp_x,
          timestamp_y, frame_width, frame_height))
  else:
    logging.debug('timestamps will not be extracted')
    return False
  
  
def process_video(
    video_file_path, output_dir_path, class_name_map, model_name,
    model_signature_name, model_server_host, model_input_size,
    return_code_queue, log_queue, log_level, ffmpeg_path, ffprobe_path,
    do_crop, crop_width, crop_height, crop_x, crop_y, do_extract_timestamps,
    timestamp_max_width, timestamp_height, timestamp_x, timestamp_y,
    do_deinterlace, num_channels, batch_size, do_smooth_probs,
    smoothing_factor, do_binarize_probs, do_write_inference_reports,
    do_write_event_reports, max_threads, processor_mode):
  configure_logger(log_level, log_queue)

  interrupt_queue = Queue()
  
  def interrupt_handler(signal_number, _):
    logging.warning('received interrupt signal {}.'.format(signal_number))

    interrupt_queue.put_nowait('_')

    # TODO: cancel timestamp/report generation when an interrupt is signalled
    # logging.debug('instructing inference pipeline to halt.')
    # child_interrupt_queue.put_nowait('_')

  signal.signal(signal.SIGINT, interrupt_handler)

  video_file_name = path.basename(video_file_path)
  video_file_name, _ = path.splitext(video_file_name)

  logging.info('preparing to analyze {}'.format(video_file_path))

  try:
    start = time()

    frame_width, frame_height, num_frames, _ = IO.get_video_dimensions(
      video_file_path, ffprobe_path)

    end = time() - start

    processing_duration = IO.get_processing_duration(
      end, 'read video dimensions in')

    logging.info(processing_duration)
  except Exception as e:
    logging.error('encountered an unexpected error while fetching video '
                  'dimensions')
    logging.error(e)

    logging.debug(
      'will exit with code: exception and value get_video_dimensions')
    log_queue.put(None)
    log_queue.close()

    return_code_queue.put(
      {'return_code': 'exception', 'return_value': 'get_video_dimensions'})
    return_code_queue.close()

    return

  try:
    do_crop = should_crop(frame_width, frame_height, do_crop, crop_width,
                          crop_height, crop_x, crop_y)
  except Exception as e:
    logging.error(e)

    logging.debug('will exit with code: exception and value should_crop')
    log_queue.put(None)
    log_queue.close()

    return_code_queue.put(
      {'return_code': 'exception', 'return_value': 'should_crop'})
    return_code_queue.close()

    return

  logging.debug('Constructing ffmpeg command')

  ffmpeg_command = [ffmpeg_path, '-i', video_file_path]

  if do_deinterlace:
    ffmpeg_command.append('-deinterlace')

  ffmpeg_command.extend(
    ['-vcodec', 'rawvideo', '-pix_fmt', 'rgb24', '-vsync', 'vfr',
     '-hide_banner', '-loglevel', '0', '-f', 'image2pipe', 'pipe:1'])

  try:
    do_extract_timestamps = should_extract_timestamps(
      frame_width, frame_height, do_extract_timestamps, timestamp_max_width,
      timestamp_height, timestamp_x, timestamp_y)
  except Exception as e:
    logging.error(e)

    logging.debug(
      'will exit with code: exception and value should_extract_timestamps')
    log_queue.put(None)
    log_queue.close()

    return_code_queue.put(
      {'return_code': 'exception', 'return_value': 'should_extract_timestamps'})
    return_code_queue.close()

    return

  frame_shape = [frame_height, frame_width, num_channels]

  logging.debug('FFmpeg output frame shape == {}'.format(frame_shape))
    #TODO parameterize tf serving values
  analyzer = VideoAnalyzer(
    frame_shape, num_frames, len(class_name_map), batch_size, model_name,
    model_signature_name, model_server_host, model_input_size,
    do_extract_timestamps, timestamp_x, timestamp_y, timestamp_height,
    timestamp_max_width, do_crop, crop_x, crop_y, crop_width, crop_height,
    ffmpeg_command, max_threads)

  try:
    start = time()

    num_analyzed_frames, probability_array, timestamp_array = analyzer.run()

    end = time()

    analysis_duration = end - start

    processing_duration = IO.get_processing_duration(
      analysis_duration, 'processed {} frames in'.format(num_analyzed_frames))
    logging.info(processing_duration)

    if num_analyzed_frames != num_frames:
      if interrupt_queue.empty():
        raise AssertionError('num_analyzed_frames ({}) != num_frames '
                             '({})'.format(num_analyzed_frames, num_frames))
      else:
        raise InterruptedError('num_analyzed_frames ({}) != num_frames '
                               '({})'.format(num_analyzed_frames, num_frames))
  except InterruptedError as ae:
    logging.error(ae)

    logging.debug('will exit with code: interrupt and value: analyze_video')
    log_queue.put(None)
    log_queue.close()

    return_code_queue.put({'return_code': 'interrupt',
                           'return_value': 'analyze_video'})
    return_code_queue.close()

    return
  except AssertionError as ae:
    logging.error(ae)

    logging.debug(
      'will exit with code: assertion error and value: analyze_video')
    log_queue.put(None)
    log_queue.close()

    return_code_queue.put({'return_code': 'assertion error',
                           'return_value': 'analyze_video'})
    return_code_queue.close()

    return
  except Exception as e:
    logging.error('encountered an unexpected error while analyzing {}'.format(
      video_file_name))
    logging.error(e)

    logging.debug(
      'will exit with code: exception and value: analyze_video')
    log_queue.put(None)
    log_queue.close()

    return_code_queue.put({'return_code': 'exception',
                           'return_value': 'analyze_video'})
    return_code_queue.close()

    return

  logging.debug('converting timestamp images to strings')

  if do_extract_timestamps:
    try:
      start = time()

      timestamp_object = Timestamp(timestamp_height, timestamp_max_width)
      timestamp_strings, qa_flags = \
        timestamp_object.stringify_timestamps(timestamp_array)

      end = time() - start

      processing_duration = IO.get_processing_duration(
        end, 'timestamp strings converted in')

      logging.info(processing_duration)
    except Exception as e:
      logging.error('encountered an unexpected error while converting '
                    'timestamp image crops to strings'.format(os.getpid()))
      logging.error(e)

      logging.debug(
        'will exit with code: exception and value: stringify_timestamps')
      log_queue.put(None)
      log_queue.close()

      return_code_queue.put({'return_code': 'exception',
                             'return_value': 'stringify_timestamps'})
      return_code_queue.close()

      return
  else:
    timestamp_strings = None
    qa_flags = None

  logging.debug('attempting to generate reports')

  if do_write_inference_reports:
    try:
      start = time()

      IO.write_inference_report(
        video_file_name, output_dir_path, analyzer.prob_array, class_name_map,
        timestamp_strings, qa_flags, do_smooth_probs, smoothing_factor,
        do_binarize_probs)

      end = time() - start

      processing_duration = IO.get_processing_duration(
        end, 'generated inference reports in')
      logging.info(processing_duration)
    except Exception as e:
      logging.error(
        'encountered an unexpected error while generating inference report.')
      logging.error(e)

      logging.debug(
        'will exit with code: exception and value: write_inference_report')
      log_queue.put(None)
      log_queue.close()

      return_code_queue.put({'return_code': 'exception',
                             'return_value': 'write_inference_report'})
      return_code_queue.close()

      return

  try:
    start = time()

    if do_smooth_probs:
      probability_array = IO.smooth_probs(
        probability_array, smoothing_factor)

    frame_numbers = list(range(1, len(probability_array) + 1))

    if timestamp_strings is not None:
      timestamp_strings = timestamp_strings.astype(np.int32)
    trip = Trip(frame_numbers, timestamp_strings, qa_flags, probability_array,
                class_name_map)

    if processor_mode == "weather":
      if len(trip.feature_sequence) > 0:
        logging.info('{} weather events were found in {}'.format(
          len(trip.feature_sequence), video_file_name))
        if do_write_event_reports:
          IO.write_weather_report(video_file_name, output_dir_path, trip.feature_sequence)
    else:
      events = trip.find_work_zone_events()

      if len(events) > 0:
        logging.info('{} work zone events were found in {}'.format(
          len(events), video_file_name))

        if do_write_event_reports:
          IO.write_event_report(video_file_name, output_dir_path, events)
      else:
        logging.info(
          'No work zone events were found in {}'.format(video_file_name))

    end = time() - start

    processing_duration = IO.get_processing_duration(
      end, 'generated event reports in')
    logging.info(processing_duration)
  except Exception as e:
    logging.error(
      'encountered an unexpected error while generating event report.')
    logging.error(e)

    logging.debug(
      'will exit with code: exception and value: write_event_report')
    log_queue.put(None)
    log_queue.close()

    return_code_queue.put({'return_code': 'exception',
                           'return_value': 'write_event_report'})
    return_code_queue.close()

    return

  logging.debug(
    'will exit with code: success and value: {}'.format(num_analyzed_frames))
  log_queue.put(None)
  log_queue.close()

  return_code_queue.put({'return_code': 'success',
                         'return_value': num_analyzed_frames,
                         'analysis_duration': analysis_duration})
  return_code_queue.close()

def process_video_signalstate(
    video_file_path, output_dir_path, class_name_map, model_name,
    model_signature_name, model_server_host, model_input_size,
    return_code_queue, log_queue, log_level, ffmpeg_path, ffprobe_path,
    do_crop, crop_width, crop_height, crop_x, crop_y, do_extract_timestamps,
    timestamp_max_width, timestamp_height, timestamp_x, timestamp_y,
    do_deinterlace, num_channels, batch_size, do_smooth_probs,
    smoothing_factor, do_binarize_probs, do_write_inference_reports,
    do_write_event_reports, max_threads, processor_mode):
  configure_logger(log_level, log_queue)

  interrupt_queue = Queue()
  
  def interrupt_handler(signal_number, _):
    logging.warning('received interrupt signal {}.'.format(signal_number))

    interrupt_queue.put_nowait('_')

    # TODO: cancel timestamp/report generation when an interrupt is signalled
    # logging.debug('instructing inference pipeline to halt.')
    # child_interrupt_queue.put_nowait('_')

  signal.signal(signal.SIGINT, interrupt_handler)

  video_file_name = path.basename(video_file_path)
  video_file_name, _ = path.splitext(video_file_name)

  logging.info('preparing to signalstate analyze {}'.format(video_file_path))

  try:
    start = time()

    # For signal state, we use duration as num_frames, as we will only grab one frame per second
    frame_width, frame_height, num_frames, duration = IO.get_video_dimensions(
      video_file_path, ffprobe_path)
    num_frames = duration
    end = time() - start

    processing_duration = IO.get_processing_duration(
      end, 'read video dimensions in')

    logging.info(processing_duration)
  except Exception as e:
    logging.error('encountered an unexpected error while fetching video '
                  'dimensions')
    logging.error(e)

    logging.debug(
      'will exit with code: exception and value get_video_dimensions')
    log_queue.put(None)
    log_queue.close()

    return_code_queue.put(
      {'return_code': 'exception', 'return_value': 'get_video_dimensions'})
    return_code_queue.close()

    return

  try:
    do_crop = should_crop(frame_width, frame_height, do_crop, crop_width,
                          crop_height, crop_x, crop_y)
  except Exception as e:
    logging.error(e)

    logging.debug('will exit with code: exception and value should_crop')
    log_queue.put(None)
    log_queue.close()

    return_code_queue.put(
      {'return_code': 'exception', 'return_value': 'should_crop'})
    return_code_queue.close()

    return

  logging.debug('Constructing ffmpeg command')

  ffmpeg_command = [ffmpeg_path, '-i', video_file_path]

  if do_deinterlace:
    ffmpeg_command.append('-deinterlace')

  ffmpeg_command.extend(
    ['-vcodec', 'rawvideo', '-pix_fmt', 'rgb24', '-vsync', 'vfr',
     '-hide_banner', '-loglevel', '0', '-r', '1', '-f', 'image2pipe', 'pipe:1'])

  try:
    do_extract_timestamps = should_extract_timestamps(
      frame_width, frame_height, do_extract_timestamps, timestamp_max_width,
      timestamp_height, timestamp_x, timestamp_y)
  except Exception as e:
    logging.error(e)

    logging.debug(
      'will exit with code: exception and value should_extract_timestamps')
    log_queue.put(None)
    log_queue.close()

    return_code_queue.put(
      {'return_code': 'exception', 'return_value': 'should_extract_timestamps'})
    return_code_queue.close()

    return

  frame_shape = [frame_height, frame_width, num_channels]

  logging.debug('FFmpeg output frame shape == {}'.format(frame_shape))

  analyzer = SignalVideoAnalyzer(
  frame_shape, num_frames, len(class_name_map), batch_size, model_name,
  model_signature_name, model_server_host, model_input_size,
  do_extract_timestamps, timestamp_x, timestamp_y, timestamp_height,
  timestamp_max_width, do_crop, crop_x, crop_y, crop_width, crop_height,
  ffmpeg_command, max_threads)

  try:
    start = time()

    num_analyzed_frames, frame_map_array, timestamp_array = analyzer.run()

    end = time()

    analysis_duration = end - start

    processing_duration = IO.get_processing_duration(
      analysis_duration, 'processed {} frames in'.format(num_analyzed_frames))
    logging.info(processing_duration)
    
  except InterruptedError as ae:
    logging.error(ae)

    logging.debug('will exit with code: interrupt and value: analyze_video')
    log_queue.put(None)
    log_queue.close()

    return_code_queue.put({'return_code': 'interrupt',
                           'return_value': 'analyze_video'})
    return_code_queue.close()

    return
  except AssertionError as ae:
    logging.error(ae)

    logging.debug(
      'will exit with code: assertion error and value: analyze_video')
    log_queue.put(None)
    log_queue.close()

    return_code_queue.put({'return_code': 'assertion error',
                           'return_value': 'analyze_video'})
    return_code_queue.close()

    return
  except Exception as e:
    logging.error('encountered an unexpected error while analyzing {}'.format(
      video_file_name))
    logging.error(e)

    logging.debug(
      'will exit with code: exception and value: analyze_video')
    log_queue.put(None)
    log_queue.close()

    return_code_queue.put({'return_code': 'exception',
                           'return_value': 'analyze_video'})
    return_code_queue.close()

    return

  logging.debug('converting timestamp images to strings')

  if do_extract_timestamps:
    try:
      start = time()

      timestamp_object = Timestamp(timestamp_height, timestamp_max_width)
      timestamp_strings, qa_flags = \
        timestamp_object.stringify_timestamps(timestamp_array)

      end = time() - start

      processing_duration = IO.get_processing_duration(
        end, 'timestamp strings converted in')

      logging.info(processing_duration)
    except Exception as e:
      logging.error('encountered an unexpected error while converting '
                    'timestamp image crops to strings'.format(os.getpid()))
      logging.error(e)

      logging.debug(
        'will exit with code: exception and value: stringify_timestamps')
      log_queue.put(None)
      log_queue.close()

      return_code_queue.put({'return_code': 'exception',
                             'return_value': 'stringify_timestamps'})
      return_code_queue.close()

      return
  else:
    timestamp_strings = None
    qa_flags = None

  logging.debug('attempting to generate reports')

  #if do_write_inference_reports:
    # TODO Write inference reports

  try:
    start = time()

    frame_numbers = list(range(1, len(frame_map_array) + 1))

    if timestamp_strings is not None:
      timestamp_strings = timestamp_strings.astype(np.int32)

    # Process our raw predictions into a list of bounding boxes and frame data
    detections = []
    for frame_num, frame_map in enumerate(frame_map_array, start=0):
      if timestamp_strings is not None:
        timestamp = timestamp_strings[frame_num]
      else:
        timestamp = None
      for i in range(0, frame_map['num_detections']):
        class_name = class_name_map[frame_map['detection_classes'][i]]
        bbox = frame_map['detection_boxes'][i]
        xtl = bbox[1] * frame_width
        ytl = bbox[0] * frame_height
        xbr = bbox[3] * frame_width
        ybr = bbox[2] * frame_height
        detections.append({'frame_num': frame_num, 'timestamp': timestamp, 'classification': class_name, \
                          'xtl': xtl, 'ytl': ytl, 'xbr': xbr, 'ybr': ybr})
    if len(detections) > 0:
      logging.info('{} signal state detections were found in {}'.format(
        len(detections), video_file_name))

      if do_write_event_reports:
        IO.write_signalstate_report(video_file_name, output_dir_path, detections)
    else:
      logging.info(
        'No signal state events were found in {}'.format(video_file_name))

    end = time() - start

    processing_duration = IO.get_processing_duration(
      end, 'generated event reports in')
    logging.info(processing_duration)
  except Exception as e:
    logging.error(
      'encountered an unexpected error while generating event report.')
    logging.error(e)

    logging.debug(
      'will exit with code: exception and value: write_event_report')
    log_queue.put(None)
    log_queue.close()

    return_code_queue.put({'return_code': 'exception',
                           'return_value': 'write_event_report'})
    return_code_queue.close()

    return

  logging.debug(
    'will exit with code: success and value: {}'.format(num_analyzed_frames))
  log_queue.put(None)
  log_queue.close()

  return_code_queue.put({'return_code': 'success',
                         'return_value': num_analyzed_frames,
                         'analysis_duration': analysis_duration})
  return_code_queue.close()