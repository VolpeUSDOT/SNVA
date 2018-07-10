import argparse
import logging
from logging.handlers import QueueHandler, TimedRotatingFileHandler
from multiprocessing import Process, Queue
import numpy as np
import os
import platform
from queue import Empty
import signal
import socket
from threading import Thread
from time import sleep, time
from utils.analysis import VideoAnalyzer
from utils.event import Trip
from utils.io import IO
from utils.timestamp import Timestamp

path = os.path


# Logger thread: listens for updates to log queue and writes them as they arrive
# Terminates after we add None to the queue
def main_logger_fn(log_queue):
  while True:
    try:
      message = log_queue.get()
      if message is None:
        break
      logger = logging.getLogger(message.name)
      logger.handle(message)
    except Exception as e:
      logging.error(e)
      break


# Logger thread: listens for updates to log queue and writes them as they arrive
# Terminates after we add None to the queue
def child_logger_fn(main_log_queue, child_log_queue):
  while True:
    try:
      message = child_log_queue.get()
      if message is None:
        break
      main_log_queue.put(message)
    except Exception as e:
      logging.error(e)
      break


def configure_logger(log_level, log_queue):
  root_logger = logging.getLogger(__name__)
  if root_logger.hasHandlers():  # Clear any handlers to avoid duplicate entries
    root_logger.handlers.clear()
  root_logger.setLevel(log_level)
  queue_handler = QueueHandler(log_queue)
  root_logger.addHandler(queue_handler)


def stringify_command(arg_list):
  command_string = arg_list[0]
  for elem in arg_list[1:]:
    command_string += ' ' + elem
  return 'command string: {}'.format(command_string)


def get_valid_num_processes_per_device(device_type):
  valid_n_procs = {1, 2}
  if device_type == 'cpu':
    n_cpus = os.cpu_count()
    n_procs = 4
    while n_procs <= n_cpus:
      k = (n_cpus - n_procs) / n_procs
      if k == int(k):
        valid_n_procs.add(n_procs)
      n_procs += 2
  return valid_n_procs


def should_crop(frame_width, frame_height):
  if args.crop:
    if all(
        [frame_width >= args.cropwidth > 0, frame_height >= args.cropheight > 0,
         frame_width > args.cropx >= 0, frame_height > args.cropy >= 0]):
      logging.info(
        'video frames will be cropped from [w={}:h={}:x={}:y={}]'.format(
          args.cropwidth, args.cropheight, args.cropx, args.cropy))
      return True
    else:
      raise ValueError(
        'video frames cannot be cropped from [w={}:h={}:x={}:y={}] because the '
        'video dimensions are [w={}:h={}]'.format(
          args.cropwidth, args.cropheight, args.cropx,
          args.cropy, frame_width, frame_height))
  else:
    logging.debug('video frames will not be cropped')
    return False


def should_extract_timestamps(frame_width, frame_height):
  if args.extracttimestamps:
    if all([frame_width >= args.timestampx + args.timestampmaxwidth > 0,
            frame_height >= args.timestampy + args.timestampheight > 0]):
      logging.info(
        'timestamps will be extracted from [w={}:h={}:x={}:y={}]'.format(
          args.timestampmaxwidth, args.timestampheight,
          args.timestampx, args.timestampy))
      return True
    else:
      raise ValueError(
        'timestamps cannot be extracted from [w={}:h={}:x={}:y={}] because the '
        'video dimensions are [w={}:h={}]'.format(
          args.timestampmaxwidth, args.timestampheight, args.timestampx,
          args.timestampy, frame_width, frame_height))
  else:
    logging.debug('timestamps will not be extracted')
    return False


def process_video(
    video_file_path, output_dir_path, class_name_map, model_input_size,
    device_id_queue, return_code_queue, log_queue, log_level, device_type,
    logical_device_count, physical_device_count, ffmpeg_path, ffprobe_path,
    model_path, node_name_map, gpu_memory_fraction):
  configure_logger(log_level, log_queue)

  interrupt_queue = Queue()

  child_interrupt_queue = Queue()

  def interrupt_handler(signal_number, _):
    logging.warning('received interrupt signal {}.'.format(signal_number))

    interrupt_queue.put_nowait('_')

    # TODO: cancel timestamp/report generation when an interrupt is signalled
    logging.debug('instructing inference pipeline to halt.')
    child_interrupt_queue.put_nowait('_')

  signal.signal(signal.SIGINT, interrupt_handler)

  video_file_name = path.basename(video_file_path)
  video_file_name, _ = path.splitext(video_file_name)

  logging.info('preparing to analyze {}'.format(video_file_path))

  try:
    start = time()

    frame_width, frame_height, num_frames = IO.get_video_dimensions(
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
    crop = should_crop(frame_width, frame_height)
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

  if args.deinterlace:
    ffmpeg_command.append('-deinterlace')

  ffmpeg_command.extend(
    ['-vcodec', 'rawvideo', '-pix_fmt', 'rgb24', '-vsync', 'vfr',
     '-hide_banner', '-loglevel', '0', '-f', 'image2pipe', 'pipe:1'])

  try:
    extract_timestamps = should_extract_timestamps(frame_width, frame_height)
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

  frame_shape = [frame_height, frame_width, args.numchannels]

  logging.debug('FFmpeg output frame shape == {}'.format(frame_shape))

  def release_device_id(device_id, device_id_queue):
    try:
      logging.debug(
        'attempting to unset CUDA_VISIBLE_DEVICES environment variable.')
      os.environ.pop('CUDA_VISIBLE_DEVICES')
    except KeyError as ke:
      logging.warning(ke)

    logging.debug('released {} device with id {}'.format(
      device_type, device_id))

    device_id_queue.put(device_id)
    device_id_queue.close()

  result_queue = Queue(1)

  analyzer = VideoAnalyzer(
    frame_shape, num_frames, len(class_name_map), args.batchsize,
    model_input_size, model_path, device_type, logical_device_count, os.cpu_count(),
    node_name_map, gpu_memory_fraction, extract_timestamps, args.timestampx,
    args.timestampy, args.timestampheight, args.timestampmaxwidth, crop,
    args.cropx, args.cropy, args.cropwidth, args.cropheight, ffmpeg_command,
    child_interrupt_queue, result_queue, video_file_name)

  device_id = device_id_queue.get()

  logging.debug('acquired {} device with id {}'.format(device_type, device_id))

  try:
    _ = child_interrupt_queue.get_nowait()

    release_device_id(device_id, device_id_queue)

    logging.debug('will exit with code: interrupt and value: process_video')
    log_queue.put(None)
    log_queue.close()

    return_code_queue.put({'return_code': 'interrupt',
                           'return_value': 'process_video'})
    return_code_queue.close()

    return
  except:
    pass

  if device_type == 'gpu':
    mapped_device_id = str(int(device_id) % physical_device_count)
    logging.debug('mapped logical device_id {} to physical device_id {}'.format(
      device_id, mapped_device_id))

    logging.debug('setting CUDA_VISIBLE_DEVICES environment variable to '
                  '{}.'.format(mapped_device_id))
    os.environ['CUDA_VISIBLE_DEVICES'] = mapped_device_id
  else:
    logging.debug('Setting CUDA_VISIBLE_DEVICES environment variable to None.')
    os.environ['CUDA_VISIBLE_DEVICES'] = ''

  try:
    start = time()

    analyzer.start()

    num_analyzed_frames, probability_array, timestamp_array = result_queue.get()

    analyzer.terminate()

    result_queue.close()

    end = time()

    analysis_duration = end - start

    processing_duration = IO.get_processing_duration(
      analysis_duration, 'processed {} frames in'.format(num_analyzed_frames))
    logging.info(processing_duration)

    analyzer.join(timeout=15)

    try:
      os.kill(analyzer.pid, signal.SIGTERM)
      logging.debug('analyzer process {} remained alive following join timeout '
                    'and had to be killed'.format(analyzer.pid))
    except:
      pass

    release_device_id(device_id, device_id_queue)

    if num_analyzed_frames != num_frames:
      if interrupt_queue.empty():
        raise AssertionError('num_analyzed_frames ({}) != num_frames '
                             '({})'.format(num_analyzed_frames, num_frames))
      else:
        raise InterruptedError('num_analyzed_frames ({}) != num_frames '
                               '({})'.format(num_analyzed_frames, num_frames))
  except InterruptedError as ae:
    logging.error(ae)

    release_device_id(device_id, device_id_queue)

    logging.debug('will exit with code: interrupt and value: analyze_video')
    log_queue.put(None)
    log_queue.close()

    return_code_queue.put({'return_code': 'interrupt',
                           'return_value': 'analyze_video'})
    return_code_queue.close()

    return
  except AssertionError as ae:
    logging.error(ae)

    release_device_id(device_id, device_id_queue)

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

    release_device_id(device_id, device_id_queue)

    logging.debug(
      'will exit with code: exception and value: analyze_video')
    log_queue.put(None)
    log_queue.close()

    return_code_queue.put({'return_code': 'exception',
                           'return_value': 'analyze_video'})
    return_code_queue.close()

    return

  logging.debug('converting timestamp images to strings')

  if extract_timestamps:
    try:
      start = time()

      timestamp_object = Timestamp(args.timestampheight, args.timestampmaxwidth)
      timestamp_strings = timestamp_object.stringify_timestamps(timestamp_array)

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

  logging.debug('attempting to generate reports')

  try:
    start = time()

    IO.write_inference_report(
      video_file_name, output_dir_path, probability_array, class_name_map,
      timestamp_strings, args.smoothprobs, args.smoothingfactor,
      args.binarizeprobs)

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

    if args.smoothprobs:
      probability_array = IO.smooth_probs(
        probability_array, args.smoothingfactor)

    frame_numbers = [i + 1 for i in range(len(probability_array))]

    if timestamp_strings is not None:
      timestamp_strings = timestamp_strings.astype(np.int32)

    trip = Trip(
      frame_numbers, timestamp_strings, probability_array, class_name_map)

    work_zone_events = trip.find_work_zone_events()

    if len(work_zone_events) > 0:
      logging.info('{} work zone events were found in {}'.format(
        len(work_zone_events), video_file_name))

      IO.write_event_report(video_file_name, output_dir_path, work_zone_events)
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


def main():
  logging.info('entering snva {} main process'.format(snva_version_string))

  total_num_video_to_process = None

  def interrupt_handler(signal_number, _):
    logging.warning('Main process received interrupt signal '
                    '{}.'.format(signal_number))
    main_interrupt_queue.put_nowait('_')

    if total_num_video_to_process is None \
        or total_num_video_to_process == len(video_file_names):

      # Signal the logging thread to finish up
      logging.debug('signaling logger thread to end service.')

      log_queue.put_nowait(None)

      logger_thread.join()

      logging.shutdown()

  signal.signal(signal.SIGINT, interrupt_handler)

  try:
    ffmpeg_path = os.environ['FFMPEG_HOME']
  except KeyError:
    logging.warning('Environment variable FFMPEG_HOME not set. Attempting '
                    'to use default ffmpeg binary location.')
    if platform.system() == 'Windows':
      ffmpeg_path = 'ffmpeg.exe'
    else:
      ffmpeg_path = '/usr/local/bin/ffmpeg'

      if not path.exists(ffmpeg_path):
        ffmpeg_path = '/usr/bin/ffmpeg'

  logging.debug('FFMPEG path set to: {}'.format(ffmpeg_path))

  try:
    ffprobe_path = os.environ['FFPROBE_HOME']
  except KeyError:
    logging.warning('Environment variable FFPROBE_HOME not set. '
                    'Attempting to use default ffprobe binary location.')
    if platform.system() == 'Windows':
      ffprobe_path = 'ffprobe.exe'
    else:
      ffprobe_path = '/usr/local/bin/ffprobe'

      if not path.exists(ffprobe_path):
        ffprobe_path = '/usr/bin/ffprobe'

  logging.debug('FFPROBE path set to: {}'.format(ffprobe_path))

  if path.isdir(args.inputpath):
    video_dir_path = args.inputpath
    video_file_names = set(IO.read_video_file_names(video_dir_path))
  elif path.isfile(args.inputpath):
    video_dir_path, video_file_name = path.split(args.inputpath)
    video_file_names = {video_file_name}
  else:
    raise ValueError('The video file/folder specified at the path {} could '
                     'not be found.'.format(args.inputpath))

  models_root_dir_path = path.join(snva_home, args.modelsdirpath)

  models_dir_path = path.join(models_root_dir_path, args.modelname)

  logging.debug('models_dir_path set to {}'.format(models_dir_path))

  model_file_path = path.join(models_dir_path, args.protobuffilename)

  if not path.isfile(model_file_path):
    raise ValueError('The model specified at the path {} could not be '
                     'found.'.format(model_file_path))

  logging.debug('model_file_path set to {}'.format(model_file_path))

  model_input_size_file_path = path.join(models_dir_path, 'input_size.txt')

  if not path.isfile(model_input_size_file_path):
    raise ValueError('The model input size file specified at the path {} '
                     'could not be found.'.format(model_input_size_file_path))

  logging.debug('model_input_size_file_path set to {}'.format(
    model_input_size_file_path))

  with open(model_input_size_file_path) as file:
    model_input_size_string = file.readline().rstrip()

    valid_size_set = ['224', '299']

    if model_input_size_string not in valid_size_set:
      raise ValueError('The model input size is not in the set {}.'.format(
        valid_size_set))

    model_input_size = int(model_input_size_string)

  # if logpath is the default value, expand it using the SNVA_HOME prefix,
  # otherwise, use the value explicitly passed by the user
  if args.outputpath == 'reports':
    output_dir_path = path.join(snva_home, args.outputpath)
  else:
    output_dir_path = args.outputpath

  if not path.isdir(output_dir_path):
    os.makedirs(output_dir_path)

  if args.excludepreviouslyprocessed:
    inference_report_dir_path = path.join(output_dir_path, 'inference_reports')

    if args.writeinferencereports and path.isdir(inference_report_dir_path):
      inference_report_file_names = os.listdir(inference_report_dir_path)
      inference_report_file_names = [path.splitext(name)[0]
                                     for name in inference_report_file_names]
      print('previously generated inference reports: {}'.format(
        inference_report_file_names))
    else:
      inference_report_file_names = None
    event_report_dir_path = path.join(output_dir_path, 'event_reports')

    if args.writeeventreports and path.isdir(event_report_dir_path):
      event_report_file_names = os.listdir(event_report_dir_path)
      event_report_file_names = [path.splitext(name)[0]
                                 for name in event_report_file_names]
      print('previously generated event reports: {}'.format(
        event_report_file_names))
    else:
      event_report_file_names = None

    file_names_to_exclude = set()

    for video_file_name in video_file_names:
      truncated_file_name = path.splitext(video_file_name)[0]
      if (event_report_file_names and truncated_file_name
          in event_report_file_names) or (inference_report_file_names and
              truncated_file_name in inference_report_file_names):
        file_names_to_exclude.add(video_file_name)

    video_file_names -= file_names_to_exclude

  if args.ionodenamesfilepath is None \
      or not path.isfile(args.ionodenamesfilepath):
    io_node_names_path = path.join(models_dir_path, 'io_node_names.txt')
  else:
    io_node_names_path = args.ionodenamesfilepath
  logging.debug('io tensors path set to: {}'.format(io_node_names_path))

  node_name_map = IO.read_node_names(io_node_names_path)

  if args.classnamesfilepath is None \
      or not path.isfile(args.classnamesfilepath):
    class_names_path = path.join(models_root_dir_path, 'class_names.txt')
  else:
    class_names_path = args.classnamesfilepath
  logging.debug('labels path set to: {}'.format(class_names_path))

  if args.cpuonly:
    device_id_list = ['0']
    device_type = 'cpu'
  else:
    device_id_list = IO.get_device_ids()
    device_type = 'gpu'

  physical_device_count = len(device_id_list)

  logging.info('Found {} physical {} device(s).'.format(
    physical_device_count, device_type))

  valid_num_processes_list = get_valid_num_processes_per_device(device_type)

  if args.numprocessesperdevice not in valid_num_processes_list:
      raise ValueError(
        'The the number of processes to assign to each {} device is expected '
        'to be in the set {}.'.format(device_type, valid_num_processes_list))

  for i in range(physical_device_count,
                 physical_device_count * args.numprocessesperdevice):
    device_id_list.append(str(i))

  logical_device_count = len(device_id_list)

  logging.info('Generated an additional {} logical {} device(s).'.format(
    logical_device_count - physical_device_count, device_type))

  # child processes will dequeue and enqueue device names
  device_id_queue = Queue(logical_device_count)

  for device_id in device_id_list:
    device_id_queue.put(device_id)

  class_name_map = IO.read_class_names(class_names_path)

  logging.debug('loading model at path: {}'.format(model_file_path))

  return_code_queue_map = {}
  child_logger_thread_map = {}
  child_process_map = {}

  total_num_video_to_process = len(video_file_names)

  total_num_processed_videos = 0
  total_num_processed_frames = 0
  total_analysis_duration = 0

  logging.info('Processing {} videos in directory: {} using {}'.format(
    total_num_video_to_process, video_dir_path, args.modelname))

  def start_video_processor(video_file_name):
    # Before popping the next video off of the list and creating a process to
    # scan it, check to see if fewer than logical_device_count + 1 processes are
    # active. If not, Wait for a child process to release its semaphore
    # acquisition. If so, acquire the semaphore, pop the next video name,
    # create the next child process, and pass the semaphore to it
    video_file_path = path.join(video_dir_path, video_file_name)

    return_code_queue = Queue()

    return_code_queue_map[video_file_name] = return_code_queue

    logging.debug('creating new child process.')

    child_log_queue = Queue()

    child_logger_thread = Thread(target=child_logger_fn,
                                 args=(log_queue, child_log_queue))

    child_logger_thread.start()

    child_logger_thread_map[video_file_name] = child_logger_thread

    gpu_memory_fraction = args.gpumemoryfraction / args.numprocessesperdevice

    child_process = Process(
      target=process_video, name=path.splitext(video_file_name)[0],
      args=(video_file_path, output_dir_path, class_name_map, model_input_size,
            device_id_queue, return_code_queue, child_log_queue, log_level,
            device_type, logical_device_count, physical_device_count,
            ffmpeg_path, ffprobe_path, model_file_path, node_name_map,
            gpu_memory_fraction))

    logging.debug('starting child process.')

    child_process.start()

    child_process_map[video_file_name] = child_process

  def close_completed_video_processors(
      total_num_processed_videos, total_num_processed_frames,
      total_analysis_duration):
    for video_file_name in list(return_code_queue_map.keys()):
      return_code_queue = return_code_queue_map[video_file_name]

      try:
        return_code_map = return_code_queue.get_nowait()

        return_code = return_code_map['return_code']
        return_value = return_code_map['return_value']

        child_process = child_process_map[video_file_name]

        logging.debug(
          'child process {} returned with exit code {} and exit value '
          '{}'.format(child_process.pid, return_code, return_value))

        if return_code == 'success':
          total_num_processed_videos += 1
          total_num_processed_frames += return_value
          total_analysis_duration += return_code_map['analysis_duration']

        child_logger_thread = child_logger_thread_map[video_file_name]
        
        logging.debug('joining logger thread for child process {}'.format(
          child_process.pid))
        
        child_logger_thread.join(timeout=15)
        
        if child_logger_thread.is_alive():
          logging.warning(
            'logger thread for child process {} remained alive following join '
            'timeout'.format(child_process.pid))
        
        logging.debug('joining child process {}'.format(child_process.pid))
        
        child_process.join(timeout=15)

        # if the child process has not yet terminated, kill the child process at
        # the risk of losing any log message not yet buffered by the main logger
        try:
          os.kill(child_process.pid, signal.SIGTERM)
          logging.warning(
            'child process {} remained alive following join timeout and had to '
            'be killed'.format(child_process.pid))
        except:
          pass
        
        return_code_queue.close()
        
        return_code_queue_map.pop(video_file_name)
        child_logger_thread_map.pop(video_file_name)
        child_process_map.pop(video_file_name)
      except Empty:
        pass

    return total_num_processed_videos, total_num_processed_frames, \
           total_analysis_duration

  start = time()

  while len(video_file_names) > 0:
    # block if logical_device_count + 1 child processes are active
    while len(return_code_queue_map) > logical_device_count:
      total_num_processed_videos, total_num_processed_frames, \
      total_analysis_duration = close_completed_video_processors(
        total_num_processed_videos, total_num_processed_frames,
        total_analysis_duration)

    try:
      _ = main_interrupt_queue.get_nowait()
      logging.debug(
        'breaking out of child process generation following interrupt signal')
      break
    except:
      pass

    video_file_name = video_file_names.pop()

    try:
      start_video_processor(video_file_name)
    except Exception as e:
      logging.error('an unknown error has occured while processing '
                    '{}'.format(video_file_name))
      logging.error(e)

  while len(return_code_queue_map) > 0:
    logging.debug('waiting for the final {} child processes to '
                  'terminate'.format(len(return_code_queue_map)))

    total_num_processed_videos, total_num_processed_frames, \
    total_analysis_duration = close_completed_video_processors(
      total_num_processed_videos, total_num_processed_frames,
      total_analysis_duration)

    # by now, the last device_id_queue_len videos are being processed,
    # so we can afford to poll for their completion infrequently
    if len(return_code_queue_map) > 0:
      sleep_duration = 10
      logging.debug('sleeping for {} seconds'.format(sleep_duration))
      sleep(sleep_duration)

  end = time() - start

  processing_duration = IO.get_processing_duration(
    end, 'snva {} processed a total of {} videos and {} frames in:'.format(
      snva_version_string, total_num_processed_videos,
      total_num_processed_frames))
  logging.info(processing_duration)

  logging.info('Video analysis alone spanned a cumulative {:.02f} seconds'.format(
      total_analysis_duration))

  logging.info('exiting snva {} main process'.format(snva_version_string))

if __name__ == '__main__':
  parser = argparse.ArgumentParser(
    description='SHRP2 NDS Video Analytics built on TensorFlow')

  parser.add_argument('--batchsize', '-bs', type=int, default=32,
                      help='Number of concurrent neural net inputs')
  parser.add_argument('--binarizeprobs', '-b', action='store_true',
                      help='Round probs to zero or one. For distributions with '
                           ' two 0.5 values, both will be rounded up to 1.0')
  parser.add_argument('--classnamesfilepath', '-cnfp',
                      help='Path to the class ids/names text file.')
  parser.add_argument('--cpuonly', '-cpu', action='store_true', help='')
  parser.add_argument('--crop', '-c', action='store_true',
                      help='Crop video frames to [offsetheight, offsetwidth, '
                           'targetheight, targetwidth]')
  parser.add_argument('--cropheight', '-ch', type=int, default=320,
                      help='y-component of bottom-right corner of crop.')
  parser.add_argument('--cropwidth', '-cw', type=int, default=474,
                      help='x-component of bottom-right corner of crop.')
  parser.add_argument('--cropx', '-cx', type=int, default=2,
                      help='x-component of top-left corner of crop.')
  parser.add_argument('--cropy', '-cy', type=int, default=0,
                      help='y-component of top-left corner of crop.')
  parser.add_argument('--deinterlace', '-d', action='store_true',
                      help='Apply de-interlacing to video frames during '
                           'extraction.')
  parser.add_argument('--excludepreviouslyprocessed', '-epp',
                      action='store_true',
                      help='Skip processing of videos for which reports '
                           'already exist in outputpath.')
  parser.add_argument('--extracttimestamps', '-et', action='store_true',
                      help='Crop timestamps out of video frames and map them to'
                           ' strings for inclusion in the output CSV.')
  parser.add_argument('--gpumemoryfraction', '-gmf', type=float, default=0.9,
                      help='% of GPU memory available to this process.')
  parser.add_argument('--inputpath', '-ip', required=True,
                      help='Path to video file(s).')
  parser.add_argument('--ionodenamesfilepath', '-ifp',
                      help='Path to the io tensor names text file.')
  parser.add_argument('--loglevel', '-ll', default='info',
                      help='Defaults to \'info\'. Pass \'debug\' or \'error\' '
                           'for verbose or minimal logging, respectively.')
  parser.add_argument('--logmode', '-lm', default='verbose',
                      help='If verbose, log to file and console. If silent, '
                           'log to file only.')
  parser.add_argument('--logpath', '-l', default='logs',
                      help='Path to the directory where log files are stored.')
  parser.add_argument('--modelsdirpath', '-mdp',
                      default='models/work_zone_scene_detection',
                      help='Path to the parent directory of model directories.')
  parser.add_argument('--modelname', '-mn', required=True,
                      help='The square input dimensions of the neural net.')
  parser.add_argument('--numchannels', '-nc', type=int, default=3,
                      help='The fourth dimension of image batches.')
  parser.add_argument('--numprocessesperdevice', '-nppd', type=int, default=1,
                      help='The number of instances of inference to perform on '
                           'each device.')
  parser.add_argument('--protobuffilename', '-pbfn', default='model.pb',
                      help='Name of the model protobuf file.')
  parser.add_argument('--outputpath', '-op', default='reports',
                      help='Path to the directory where reports are stored.')
  parser.add_argument('--smoothprobs', '-sp', action='store_true',
                      help='Apply class-wise smoothing across video frame '
                           'class probability distributions.')
  parser.add_argument('--smoothingfactor', '-sf', type=int, default=16,
                      help='The class-wise probability smoothing factor.')
  parser.add_argument('--timestampheight', '-th', type=int, default=16,
                      help='The length of the y-dimension of the timestamp '
                           'overlay.')
  parser.add_argument('--timestampmaxwidth', '-tw', type=int, default=160,
                      help='The length of the x-dimension of the timestamp '
                           'overlay.')
  parser.add_argument('--timestampx', '-tx', type=int, default=25,
                      help='x-component of top-left corner of timestamp '
                           '(before cropping).')
  parser.add_argument('--timestampy', '-ty', type=int, default=340,
                      help='y-component of top-left corner of timestamp '
                           '(before cropping).')
  parser.add_argument('--writeeventreports', '-wer', type=bool,
                      default=True, help='Output a CVS file for each video containing one or more feature events')
  parser.add_argument('--writeinferencereports', '-wir', type=bool,
                      default=False, help='For every video, output a CSV file containing a probability distribution over class labels, a timestamp, and a frame number for each frame')

  args = parser.parse_args()

  snva_version_string = 'v0.1.1'

  os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

  try:
    snva_home = os.environ['SNVA_HOME']
  except KeyError:
    snva_home = '.'

  # Define our log level based on arguments
  if args.loglevel == 'error':
    log_level = logging.ERROR
  elif args.loglevel == 'debug':
    log_level = logging.DEBUG
  else:
    log_level = logging.INFO

  # if logpath is the default value, expand it using the SNVA_HOME prefix,
  # otherwise, use the value explicitly passed by the user
  if args.logpath == 'logs':
    logs_dir_path = path.join(snva_home, args.logpath)
  else:
    logs_dir_path = args.logpath

  # Configure our log in the main process to write to a file
  if path.exists(logs_dir_path):
    if path.isfile(logs_dir_path):
      raise ValueError('The specified logpath {} is expected to be a '
                       'directory, not a file.'.format(logs_dir_path))
  else:
    os.makedirs(logs_dir_path)

  try:
    log_file_name = 'snva_' + socket.getfqdn() + '.log'
  except:
    log_file_name = 'snva.log'

  log_file_path = path.join(logs_dir_path, log_file_name)

  log_handlers = [TimedRotatingFileHandler(
    filename=log_file_path, when='H', interval=4, encoding='utf-8')]

  valid_log_modes = ['verbose', 'silent']

  if args.logmode == 'verbose':
    log_handlers.append(logging.StreamHandler())
  elif not args.logmode == 'silent':
    raise ValueError(
      'The specified logmode is not in the set {}.'.format(valid_log_modes))

  log_format = '%(asctime)s:%(processName)s:%(process)d:%(levelname)s:' \
               '%(module)s:%(funcName)s:%(message)s'

  logging.basicConfig(level=log_level, format=log_format, handlers=log_handlers)

  log_queue = Queue()

  logger_thread = Thread(target=main_logger_fn, args=(log_queue,))

  logger_thread.start()

  logging.debug('SNVA_HOME set to {}'.format(snva_home))

  main_interrupt_queue = Queue()

  try:
    main()
  except Exception as e:
    logging.error(e)

  logging.debug('signaling logger thread to end service.')
  log_queue.put(None)

  logger_thread.join()

  logging.shutdown()
