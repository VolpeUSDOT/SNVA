import argparse
import logging
from logging.handlers import QueueHandler, TimedRotatingFileHandler
from multiprocessing import BoundedSemaphore, Process, Queue
import numpy as np
import os
import platform
from queue import Empty
import signal
import socket
import subprocess as sp
from threading import Thread
from time import sleep, time
from utils import analysis
from utils.io import IO
from utils.timestamp import Timestamp

path = os.path


# Logger thread: listens for updates to log queue and writes them as they arrive
# Terminates after we add None to the queue
def logger_fn(log_queue):
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
  logging.shutdown()


def configure_logger(log_level, log_queue):
  root_logger = logging.getLogger()
  if root_logger.hasHandlers():  # Clear any handlers to avoid duplicate entries
    root_logger.handlers.clear()
  root_logger.setLevel(log_level)
  queue_handler = QueueHandler(log_queue)
  root_logger.addHandler(queue_handler)


def get_ffmpeg_command(video_file_path, ffmpeg_path, frame_width, frame_height):
  logging.debug('Constructing ffmpeg command')

  command = [ffmpeg_path, '-i', video_file_path]

  if args.crop and all(
      [frame_width >= args.cropwidth > 0, frame_height >= args.cropheight > 0,
       frame_width > args.cropx >= 0, frame_height > args.cropy >= 0]
  ):
    command.extend(['-vf', 'crop=w={}:h={}:x={}:y={}'.format(
      args.cropwidth, args.cropheight, args.cropx, args.cropy)])

    frame_width = args.cropwidth
    frame_height = args.cropheight

  command.extend(
    ['-vcodec', 'rawvideo', '-pix_fmt', 'rgb24', '-vsync', 'vfr',
     '-hide_banner', '-loglevel', '0', '-f', 'image2pipe', 'pipe:1'])

  logging.debug(IO.command_as_string(command))

  return command, frame_width, frame_height


def process_video(
    video_file_path, class_names, model_map, model_input_size, device_id_queue,
    return_code_queue, child_process_semaphore, log_queue, log_level,
    device_type, device_count, ffmpeg_path, ffprobe_path):
  configure_logger(log_level, log_queue)

  child_interrupt_queue = Queue()

  def interrupt_handler(signal_number, _):
    logging.warning('received interrupt signal {}.'.format(signal_number))

    # TODO: cancel timestamp/report generation when an interrupt is signalled
    logging.debug('instructing inference pipeline to halt.')
    child_interrupt_queue.put('_')

  signal.signal(signal.SIGINT, interrupt_handler)

  video_file_name = path.basename(video_file_path)
  video_file_name, _ = path.splitext(video_file_name)

  logging.info('preparing to analyze {}'.format(video_file_path))

  try:
    frame_width, frame_height, num_frames = IO.get_video_dimensions(
      video_file_path, ffprobe_path)
  except Exception as e:
    logging.error('encountered an unexpected error while fetching video '
                  'dimensions')
    logging.error(e)

    logging.debug(
      'will exit with code: exception and value None')
    return_code_queue.put((os.getpid(), 'exception', None))

    logging.debug('released semaphore back to parent process '
                  '{}'.format(os.getppid()))
    child_process_semaphore.release()

    return

  ffmpeg_command, frame_width, frame_height = get_ffmpeg_command(
    video_file_path, ffmpeg_path, frame_width, frame_height)

  if not args.excludetimestamps:
    timestamp_array = np.ndarray((args.timestampheight * num_frames,
                                  args.timestampmaxwidth, args.numchannels),
                                 dtype='uint8')

  video_frame_shape = (frame_height, frame_width, args.numchannels)

  video_frame_string_len = frame_height * frame_width * args.numchannels

  # feed the tf.data input pipeline one image at a time and, while we're at it,
  # extract timestamp overlay crops for later mapping to strings.
  def video_frame_generator():
    if not args.excludetimestamps:
      i = 0

      tx = args.timestampx - args.cropx
      ty = args.timestampy - args.cropy
      th = args.timestampheight
      tw = args.timestampmaxwidth

    logging.debug('opening image pipe')

    base_two_exp = 2

    while base_two_exp < video_frame_string_len:
      base_two_exp *= 2

    video_frame_pipe = sp.Popen(ffmpeg_command, stdout=sp.PIPE, stderr=sp.PIPE,
                                bufsize=args.batchsize * base_two_exp)

    while True:
      try:
        try:
          _ = child_interrupt_queue.get_nowait()
          logging.warning('closing image pipe following interrupt signal')
          video_frame_pipe.stdout.close()
          video_frame_pipe.stderr.close()
          video_frame_pipe.terminate()
          video_frame_pipe = None
          return
        except:
          pass

        video_frame_string = video_frame_pipe.stdout.read(
          video_frame_string_len)

        if not video_frame_string:
          logging.debug('closing image pipe following end of stream')
          video_frame_pipe.stdout.close()
          video_frame_pipe.stderr.close()
          video_frame_pipe.terminate()
          video_frame_pipe = None
          return

        video_frame_array = np.fromstring(video_frame_string, dtype=np.uint8)
        video_frame_array = np.reshape(video_frame_array, video_frame_shape)

        if not args.excludetimestamps:
          timestamp_array[th * i:th * (i + 1)] = \
            video_frame_array[ty:ty + th, tx:tx + tw]
          i += 1

        yield video_frame_array
      except Exception as e:
        logging.error('met an unexpected error after processing {} '
                      'frames.'.format(i))
        logging.error(e)
        logging.error('ffmpeg reported:\n{}'.format(video_frame_pipe.stderr.readlines()))
        logging.debug('closing image pipe following raised exception')
        video_frame_pipe.stdout.close()
        video_frame_pipe.stderr.close()
        video_frame_pipe.terminate()
        logging.debug('raising exception to caller.')
        raise e

  def preprocessing_fn(image):
    return analysis.preprocess_for_inception(image, model_input_size)

  # pre-allocate memory for prediction storage
  num_classes = len(class_names)
  probability_array = np.ndarray((num_frames, num_classes), dtype=np.float32)

  device_id = device_id_queue.get()
  logging.debug('acquired {} device with id '
                '{}'.format(device_type, device_id))

  try:
    _ = child_interrupt_queue.get_nowait()
    logging.debug('will exit with code: interrupt and value: None')
    return_code_queue.put((os.getpid(), 'exception', None))
    return_code_queue.close()

    logging.debug('released device_id {}'.format(device_id))
    device_id_queue.put(device_id)
    device_id_queue.close()

    logging.debug(
      'released semaphore back to parent process {}'.format(os.getppid()))
    child_process_semaphore.release()

    return
  except:
    pass

  if device_type == 'gpu':
    mapped_device_id = str(int(device_id) % 2)

    logging.debug('mapped logical device_id {} to physical device_id {}'.format(
      device_id, mapped_device_id))

    logging.debug('setting CUDA_VISIBLE_DEVICES environment variable to '
                  '{}.'.format(mapped_device_id))

    os.environ['CUDA_VISIBLE_DEVICES'] = mapped_device_id
  else:
    logging.debug('Setting CUDA_VISIBLE_DEVICES environment variable to None.')
    os.environ['CUDA_VISIBLE_DEVICES'] = ''

  try:
    num_analyzed_frames = analysis.analyze_video(
      video_frame_generator, video_frame_shape, args.batchsize,
      model_map['session_config'], model_map['input_node'],
      model_map['output_node'], preprocessing_fn, probability_array,
      device_type, device_count)

    if num_analyzed_frames != num_frames:
      raise AssertionError('num_analyzed_frames ({}) != num_frames '
                           '({})'.format(num_analyzed_frames, num_frames))

    try:
      logging.debug(
        'attempting to unset CUDA_VISIBLE_DEVICES environment variable.')
      os.environ.pop('CUDA_VISIBLE_DEVICES')
    except KeyError as ke:
      logging.warning(ke)

    logging.debug('released device_id {}'.format(device_id))
    device_id_queue.put(device_id)
    device_id_queue.close()
    device_id = None
  except Exception as e:
    logging.error('encountered an unexpected error while analyzing '
                  '{}'.format(video_file_name))
    logging.error(e)

    try:
      logging.debug(
        'attempting to unset CUDA_VISIBLE_DEVICES environment variable.')
      os.environ.pop('CUDA_VISIBLE_DEVICES')
    except KeyError as ke:
      logging.warning(ke)

    logging.debug('released device_id {}'.format(device_id))
    device_id_queue.put(device_id)
    device_id_queue.close()

    logging.debug(
      'will exit with code: exception and value: None')
    return_code_queue.put((os.getpid(), 'exception', None))
    return_code_queue.close()

    logging.debug('released semaphore back to parent process '
                  '{}'.format(os.getppid()))
    child_process_semaphore.release()

    return

  logging.debug('converting timestamp images to strings')

  if args.excludetimestamps:
    timestamp_strings = None
  else:
    try:
      start = time()

      timestamp_object = Timestamp(args.timestampheight, args.timestampmaxwidth)
      timestamp_strings = timestamp_object.stringify_timestamps(timestamp_array)

      end = time() - start

      processing_duration = IO.get_processing_duration(
        end, 'timestamp strings converted in')
      logging.info(processing_duration)
    except Exception as e:
      logging.error('encountered an unexpected error while '
                    'converting timestamp image crops to strings'.format(
        os.getpid()))
      logging.error(e)

      logging.debug('will exit with code: exception and value: '
                    'None')
      return_code_queue.put((os.getpid(), 'exception', None))
      return_code_queue.close()

      logging.debug('released semaphore back to parent '
                    'process {}'.format(os.getppid()))
      child_process_semaphore.release()

      return

  logging.debug('attempting to generate reports strings')

  try:
    start = time()

    IO.write_report(video_file_name, args.reportpath, args.excludetimestamps,
                    timestamp_strings, probability_array, class_names,
                    args.smoothprobs, args.smoothingfactor, args.binarizeprobs)

    end = time() - start

    processing_duration = IO.get_processing_duration(
      end, 'generated reports in')
    logging.info(processing_duration)
  except Exception as e:
    logging.error('encountered an unexpected error while '
                  'generating report.')
    logging.error(e)

    logging.debug('will exit with code: exception and value: '
                  'None')
    return_code_queue.put((os.getpid(), 'exception', None))
    return_code_queue.close()

    logging.debug('released semaphore back to parent process '
                  '{}'.format(os.getppid()))
    child_process_semaphore.release()

    return

  logging.debug(
    'will exit with code: success and value: {}'.format(num_analyzed_frames))
  return_code_queue.put((os.getpid(), 'success', num_analyzed_frames))
  return_code_queue.close()

  logging.debug('released semaphore back to parent process '
                '{}'.format(os.getppid()))
  child_process_semaphore.release()


def main():
  logging.info('entering snva v0.1 main process')

  # TODO: manage muliple sequential interrupt signals
  def interrupt_handler(signal_number, _):
    logging.warning('Main process received interrupt signal '
                    '{}.'.format(signal_number))
    main_interrupt_queue.put('_')

    if len(video_file_names) == total_num_video_to_process:

      # Signal the logging thread to finish up
      logging.debug('signaling logger thread to end service.')
      log_queue.put(None)
      log_queue.close()

      logging.debug('joining logger thread.')
      logger_thread.join()

      exit()

  signal.signal(signal.SIGINT, interrupt_handler)

  try:
    ffmpeg_path = os.environ['FFMPEG_HOME']
  except KeyError:
    logging.warning('Environment variable FFMPEG_HOME not set. Attempting '
                    'to use default ffmpeg binary location.')
    if platform.system() == 'Windows':
      ffmpeg_path = 'ffmpeg.exe'
    else:
      ffmpeg_path = '/usr/local/bin/ffmpeg' \
        if path.exists('/usr/local/bin/ffmpeg') else '/usr/bin/ffmpeg'

  logging.debug('FFMPEG path set to: {}'.format(ffmpeg_path))

  try:
    ffprobe_path = os.environ['FFPROBE_HOME']
  except KeyError:
    logging.warning('Environment variable FFPROBE_HOME not set. '
                    'Attempting to use default ffprobe binary location.')
    if platform.system() == 'Windows':
      ffprobe_path = 'ffprobe.exe'
    else:
      ffprobe_path = '/usr/local/bin/ffprobe' \
        if path.exists('/usr/local/bin/ffprobe') else '/usr/bin/ffprobe'

  logging.debug('FFPROBE path set to: {}'.format(ffprobe_path))

  if path.isdir(args.videopath):
    video_dir_path = args.videopath
    video_file_names = IO.read_video_file_names(video_dir_path)
  elif path.isfile(args.videopath):
    video_dir_path, video_file_name = path.split(args.videopath)
    video_file_names = [video_file_name]
  else:
    raise ValueError('The video file/folder specified at the path {} could '
                     'not be found.'.format(args.videopath))

  if not path.isdir(args.modelsdirpath):
    raise ValueError('The model specified at the path {} could not be '
                     'found.'.format(args.modelsdirpath))

  model_dir_path = path.join(args.modelsdirpath, args.modelname)
  model_file_path = path.join(model_dir_path, args.protobuffilename)

  if not path.isfile(model_file_path):
    raise ValueError('The model specified at the path {} could not be '
                     'found.'.format(model_file_path))

  model_input_size_file_path = path.join(model_dir_path, 'input_size.txt')

  if not path.isfile(model_input_size_file_path):
    raise ValueError('The model input size file specified at the path {} '
                     'could not be found.'.format(model_input_size_file_path))

  with open(model_input_size_file_path) as file:
    model_input_size_string = file.readline().rstrip()

    valid_size_set = ['224', '299']

    if model_input_size_string not in valid_size_set:
      raise ValueError('The model input size is not in the set {}.'.format(
        valid_size_set))

    model_input_size = int(model_input_size_string)

  if args.excludepreviouslyprocessed and path.isdir(args.reportpath):
    report_file_names = os.listdir(args.reportpath)

    if len(report_file_names) > 0:
      video_ext = path.splitext(video_file_names[0])[1]
      report_ext = path.splitext(report_file_names[0])[1]

      previously_processed_video_file_names = [
        name.replace(report_ext, video_ext) for name in report_file_names]

      video_file_names = [name for name in video_file_names if name
                          not in previously_processed_video_file_names]

  if not path.isdir(args.reportpath):
    os.makedirs(args.reportpath)

  if args.ionodenamesfilepath is None \
      or not path.isfile(args.ionodenamesfilepath):
    io_node_names_path = path.join(model_dir_path, 'io_node_names.txt')
  else:
    io_node_names_path = args.ionodenamesfilepath
  logging.debug('io tensors path set to: {}'.format(io_node_names_path))

  if args.classnamesfilepath is None \
      or not path.isfile(args.classnamesfilepath):
    class_names_path = path.join(args.modelsdirpath, 'class_names.txt')
  else:
    class_names_path = args.classnamesfilepath
  logging.debug('labels path set to: {}'.format(class_names_path))

  if args.cpuonly:
    device_id_list = ['0']
    device_type = 'cpu'
  else:
    device_id_list = IO.get_device_ids()
    device_type = 'gpu'

  device_id_list_len = len(device_id_list)

  logging.info('Found {} physical {} device(s).'.format(
    device_id_list_len, device_type))

  if not args.numprocessespergpu > 0:
      raise ValueError(
        'The the number of processes to assign to each GPU is expected to be '
        'an integer greater than zero.'.format(valid_size_set))

  for i in range(device_id_list_len,
                 device_id_list_len * args.numprocessespergpu):
    device_id_list.append(str(i))
    device_id_list_len += 1

  logging.info('Generated an additional {} logical {} device(s).'.format(
    int(device_id_list_len - (device_id_list_len / args.numprocessespergpu)),
    device_type))

  # child processes will dequeue and enqueue device names
  device_id_queue = Queue(device_id_list_len)

  for device_id in device_id_list:
    device_id_queue.put(device_id)

  label_map = IO.read_class_names(class_names_path)

  class_name_list = list(label_map.values())

  logging.debug('loading model at path: {}'.format(model_file_path))

  model_map = analysis.load_model(
    model_file_path, io_node_names_path, device_type,
    args.gpumemoryfraction / args.numprocessespergpu)

  # The chief worker will allow at most device_count + 1 child processes to be
  # created since the greatest number of concurrent operations is the number
  # of compute devices plus one for IO
  child_process_semaphore = BoundedSemaphore(device_id_list_len + 1)

  return_code_map = {}

  total_num_video_to_process = len(video_file_names)

  logging.info('Processing {} videos in directory: {} using {}'.format(
    total_num_video_to_process, video_dir_path, args.modelname))

  def call_process_video(video_file_name, child_process_semaphore):
    # Before popping the next video off of the list and creating a process to
    # scan it, check to see if fewer than device_id_list_len + 1 processes are
    # active. If not, Wait for a child process to release its semaphore
    # acquisition. If so, acquire the semaphore, pop the next video name,
    # create the next child process, and pass the semaphore to it
    video_file_path = path.join(video_dir_path, video_file_name)

    return_code_queue = Queue()

    if device_id_list_len > 1:
      logging.debug('creating new child process.')

      child_process = Process(
        target=process_video,
        name='ChildProcess-{}'.format(path.splitext(video_file_name)[0]),
        args=(video_file_path, class_name_list, model_map, model_input_size,
              device_id_queue, return_code_queue, child_process_semaphore,
              log_queue, log_level, device_type, device_id_list_len,
              ffmpeg_path, ffprobe_path))

      logging.debug('starting child process.')

      child_process.start()
    else:
      logging.debug('invoking process_video() in main process because '
                    'device_type == {}'.format(device_type))

      process_video(
        video_file_path, class_name_list, model_map, model_input_size,
        device_id_queue, return_code_queue, child_process_semaphore,
        log_queue, log_level, device_type, device_id_list_len,
        ffmpeg_path, ffprobe_path)

    return_code_map[video_file_name] = return_code_queue

  total_num_videos_processed = 0
  total_num_frames_processed = 0

  start = time()

  while len(video_file_names) > 0:
    # block if device_id_list_len + 1 child processes are active
    child_process_semaphore.acquire()
    logging.debug('acquired child_process_semaphore')

    try:
      _ = main_interrupt_queue.get_nowait()
      child_process_semaphore.release()
      logging.debug(
        'released child_process_semaphore following interrupt signal')
      break
    except:
      pass

    video_file_name = video_file_names.pop()

    try:
      call_process_video(video_file_name, child_process_semaphore)
    except Exception as e:
      logging.error('an unknown error has occured while processing '
                    '{}'.format(video_file_name))
      logging.error(e)

    for video_file_name in list(return_code_map.keys()):
      return_code_queue = return_code_map[video_file_name]
      try:
        child_pid, exit_code, exit_value = return_code_queue.get_nowait()
        logging.debug('child process {} returned with exit code {} and exit '
                      'value {}'.format(child_pid, exit_code, exit_value))
        if exit_code == 'success':
          total_num_videos_processed += 1
          total_num_frames_processed += exit_value
        try:
          os.kill(child_pid, signal.SIGTERM)
          logging.debug('child process {} was still alive following return and '
                        'had to be killed'.format(child_pid))
        except:
          pass
        return_code_queue.close()
        return_code_map.pop(video_file_name)
      except Empty:
        pass

  logging.debug('waiting for every child process to terminate.')

  while len(return_code_map) > 0:
    logging.debug('return_code_map_len on entry == {}'.format(
      len(return_code_map)))

    for video_file_name in list(return_code_map.keys()):
      return_code_queue = return_code_map[video_file_name]

      try:
        child_pid, exit_code, exit_value = return_code_queue.get_nowait()

        logging.debug('video_file_name == {}'.format(video_file_name))

        logging.debug('child_pid, exit_code, exit_value == {}, {}, '
                      '{}'.format(child_pid, exit_code, exit_value))
        if exit_code == 'success':
          total_num_videos_processed += 1
          total_num_frames_processed += exit_value

        return_code_queue.close()
        return_code_map.pop(video_file_name)
      except Empty:
        logging.debug('Child process assigned to {} has not yet '
                      'returned.'.format(video_file_name))

    logging.debug('return_code_map_len on exit == {}'.format(
      len(return_code_map)))

    # by now, the last device_id_queue_len videos are being processed,
    # so we can afford to poll for their completion infrequently
    if len(return_code_map) > 0:
      sleep(10)

  end = time() - start

  processing_duration = IO.get_processing_duration(
    end, 'snva v0.1 processed a total of {} videos and {} frames in: '.format(
      total_num_videos_processed, total_num_frames_processed))
  logging.info(processing_duration)

  logging.info('exiting snva v0.1 main process')

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
  parser.add_argument('--cropheight', '-ch', type=int, default=356,
                      help='y-component of bottom-right corner of crop.')
  parser.add_argument('--cropwidth', '-cw', type=int, default=474,
                      help='x-component of bottom-right corner of crop.')
  parser.add_argument('--cropx', '-cx', type=int, default=2,
                      help='x-component of top-left corner of crop.')
  parser.add_argument('--cropy', '-cy', type=int, default=0,
                      help='y-component of top-left corner of crop.')
  parser.add_argument('--excludepreviouslyprocessed', '-epp',
                      action='store_true',
                      help='Skip processing of videos for which reports '
                           'already exist in reportpath.')
  parser.add_argument('--excludetimestamps', '-et', action='store_true',
                      help='Read timestamps off of video frames and include '
                           'them as strings in the output CSV.')
  parser.add_argument('--gpumemoryfraction', '-gmf', type=float, default=0.9,
                      help='% of GPU memory available to this process.')
  parser.add_argument('--ionodenamesfilepath', '-ifp',
                      help='Path to the io tensor names text file.')
  parser.add_argument('--modelsdirpath', '-mdp',
                      default='./models/work_zone_scene_detection',
                      help='Path to the parent directory of model directories.')
  parser.add_argument('--modelname', '-mn', required=True,
                      help='The square input dimensions of the neural net.')
  parser.add_argument('--loglevel', '-ll', default='info',
                      help='Defaults to \'info\'. Pass \'debug\' or \'error\' '
                           'for verbose or minimal logging, respectively.')
  parser.add_argument('--logmode', '-lm', default='standard',
                      help='Print debug info in logs. If verbose and debug'
                           ' are both not set, log level will default to error')
  parser.add_argument('--logpath', '-l', default='./logs',
                      help='Path to the directory where log files are stored.')
  parser.add_argument('--logsilently', '-ls', action='store_true',
                      help='Print logs to logfile only and not to console. '
                           'Use together with --logmode flags')
  parser.add_argument('--numchannels', '-nc', type=int, default=3,
                      help='The fourth dimension of image batches.')
  parser.add_argument('--numprocessespergpu', '-nppg', type=int, default=1,
                      help='The number of instances of interence to perform'
                           'on each GPU.')
  parser.add_argument('--protobuffilename', '-pbfn', default='model.pb',
                      help='Name of the model protobuf file.')
  parser.add_argument('--reportpath', '-rp', default='./reports',
                      help='Path to the directory where reports are stored.')
  parser.add_argument('--smoothprobs', '-sm', action='store_true',
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
  parser.add_argument('--videopath', '-v', required=True,
                      help='Path to video file(s).')
  parser.add_argument('--nfs', '-nfs', action='store_true',
                      help='Indicates videopath is an nfs share')
  parser.add_argument('--nfs_username', '-nu',
                      help='Username for videopath nfs share')
  parser.add_argument('--nfs_password', '-np',
                      help='Password for videopath nfs share')

  args = parser.parse_args()

  os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

  # Define our log level based on arguments
  if args.loglevel == 'error':
    log_level = logging.ERROR
  elif args.loglevel == 'debug':
    log_level = logging.DEBUG
  else:
    log_level = logging.INFO

  # Configure our log in the main process to write to a file
  if path.exists(args.logpath):
    if path.isfile(args.logpath):
      raise ValueError('The specified logpath {} is expected to be a '
                       'directory, not a file.'.format(args.logpath))
  else:
    logging.debug("Creating log directory {}".format(args.logpath))

    os.makedirs(args.logpath)

  try:
    log_file_name = 'snva_' + socket.getfqdn() + '.log'
  except:
    log_file_name = 'snva.log'

  log_file_path = path.join(args.logpath, log_file_name)

  log_handlers = [TimedRotatingFileHandler(
    filename=log_file_path, when='midnight', encoding='utf-8')]

  if not args.logsilently:
    log_handlers.append(logging.StreamHandler())

  log_format = '%(asctime)s:%(processName)s:%(process)d:' \
               '%(levelname)s:%(module)s:%(funcName)s:%(message)s'

  logging.basicConfig(level=log_level, format=log_format, handlers=log_handlers)

  # Create a queue to handle log requests from multiple processes
  log_queue = Queue()

  # Start our listener process (use of threads creates deadlock issues)
  logger_thread = Thread(target=logger_fn, args=(log_queue,))

  logger_thread.start()

  main_interrupt_queue = Queue()

  main()

  # Signal the logging thread to finish up
  logging.debug('signaling logger thread to end service.')
  log_queue.put(None)
  log_queue.close()

  logging.debug('joining logger thread.')
  logger_thread.join()
