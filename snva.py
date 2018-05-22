import argparse
from datetime import datetime as dt
import logging
import logging.handlers
from multiprocessing import BoundedSemaphore, Process, Queue
import numpy as np
import os
import platform
# import psutil
import signal
from snva_utils.io import IO
from subprocess import Popen, PIPE
import sys
import tensorflow as tf
from threading import Thread
import time
from snva_utils.timestamp import Timestamp

path = os.path

parser = argparse.ArgumentParser(description='SHRP2 NDS Video Analytics built on TensorFlow')

parser.add_argument('--batchsize', '-bs', type=int, default=32,
                    help='The number of images fed into the neural net at a time')
parser.add_argument('--binarizeprobs', '-b', action='store_true',
                    help='Round probabilities to zero or one. For distributions'
                         'with two 0.5 values, both will be rounded up to 1.0')
parser.add_argument('--cpuonly', '-cpu', action='store_true', help='')
parser.add_argument('--crop', '-c', action='store_true',
                    help='Crop video frames to [offsetheight, offsetwidth, targetheight, targetwidth]')
parser.add_argument('--cropheight', '-ch', type=int, default=356, help='y-component of bottom-right corner of crop.')
parser.add_argument('--cropwidth', '-cw', type=int, default=474, help='x-component of bottom-right corner of crop.')
parser.add_argument('--cropx', '-cx', type=int, default=2, help='x-component of top-left corner of crop.')
parser.add_argument('--cropy', '-cy', type=int, default=0, help='y-component of top-left corner of crop.')
parser.add_argument('--gpumemoryfraction', '-gmf', type=float, default=0.9,
                    help='% of GPU memory available to this process.')
parser.add_argument('--iotensornamespath', '-itp', default=None, help='Path to the io tensor names text file.')
parser.add_argument('--classnamespath', '-cnp', default=None, help='Path to the class ids/names text file.')
parser.add_argument('--modelinputsize', '-mis', type=int, default=299,
                    help='The square input dimensions of the neural net.')
parser.add_argument('--logpath', '-l', default='./logs', help='Path to the directory where log files are stored.')
parser.add_argument('--numchannels', '-nc', type=int, default=3, help='The fourth dimension of image batches.')
parser.add_argument('--modelpath', '-mp', required=True, help='Path to the tensorflow protobuf model file.')
parser.add_argument('--reportpath', '-rp', default='./results', help='Path to the directory where results are stored.')
parser.add_argument('--smoothingfactor', '-sf', type=int, default=16,
                    help='The class-wise probability smoothing factor.')
parser.add_argument('--smoothprobs', '-sm', action='store_true',
                    help='Apply class-wise smoothing across video frame class probability distributions.')
parser.add_argument('--timestampheight', '-th', type=int, default=16,
                    help='The length of the y-dimension of the timestamp overlay.')
parser.add_argument('--timestampmaxwidth', '-tw', type=int, default=160,
                    help='The length of the x-dimension of the timestamp overlay.')
parser.add_argument('--timestampx', '-tx', type=int, default=25,
                    help='x-component of top-left corner of timestamp (before cropping).')
parser.add_argument('--timestampy', '-ty', type=int, default=340,
                    help='y-component of top-left corner of timestamp (before cropping).')
parser.add_argument('--videopath', '-v', required=True, help='Path to video file(s).')
parser.add_argument('--verbose', '-vb', action='store_true', help='Print additional information in logs')
parser.add_argument('--debug', '-d', action='store_true', help='Print debug information in logs')

args = parser.parse_args()

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

FFMPEG_PATH = os.environ['FFMPEG_HOME']
print(FFMPEG_PATH)
if not FFMPEG_PATH:
  if platform.system() == 'Windows':
    FFMPEG_PATH = 'ffmpeg.exe'
  else:
    FFMPEG_PATH = '/usr/local/bin/ffmpeg' if path.exists('/usr/local/bin/ffmpeg') else '/usr/bin/ffmpeg'

FFPROBE_PATH = os.environ['FFPROBE_HOME']
print(FFPROBE_PATH)
if not FFPROBE_PATH:
  if platform.system() == 'Windows':
    FFPROBE_PATH = 'ffprobe.exe'
  else:
    FFPROBE_PATH = '/usr/local/bin/ffprobe' if path.exists('/usr/local/bin/ffprobe') else '/usr/bin/ffprobe'

# Define our log level based on arguments
loglevel = logging.WARNING
if args.verbose:
  loglevel = logging.INFO
if args.debug:
  loglevel = logging.DEBUG


# Logger thread: listens for updates to our log queue and writes them as they come in
# Terminates after we add None to the queue
def logger_thread(q):
  while True:
    record = q.get()
    if record is None:
      logging.debug('Terminating log thread')
      break
    logger = logging.getLogger(record.name)
    logger.handle(record)


def preprocess_for_inception(image):
  logging.debug('Preprocessing image...')
  if image.dtype != tf.float32:
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
  if image.shape[0] != args.modelinputsize \
      or image.shape[1] != args.modelinputsize:
    image = tf.expand_dims(image, 0)
    image = tf.image.resize_bilinear(
      image, [args.modelinputsize, args.modelinputsize], align_corners=False)
    image = tf.squeeze(image, [0])
  image = tf.subtract(image, 0.5)
  image = tf.multiply(image, 2.0)
  return image


def infer_class_names(
    video_file_path, video_file_name, output_width, output_height, num_frames, session_config, session_graph,
    input_tensor_name, output_tensor_name, image_size_tensor_name, batch_size, num_classes, device_type, device_count):
  process_id = os.getpid()
  ############################
  # construct ffmpeg command #
  ############################
  logging.debug('Constructing ffmpeg command')
  command = [FFMPEG_PATH, '-i', video_file_path]

  if args.crop and all([output_width >= args.cropwidth > 0, output_height >= args.cropheight > 0,
                        output_width > args.cropx >= 0, output_height > args.cropy >= 0]):
    command.extend(['-vf', 'crop=w={}:h={}:x={}:y={}'.format(
      args.cropwidth, args.cropheight, args.cropx, args.cropy)])

    output_width = args.cropwidth
    output_height = args.cropheight

  command.extend(['-vcodec', 'rawvideo', '-pix_fmt', 'rgb24', '-vsync', 'vfr',
                  '-hide_banner', '-loglevel', '0', '-f', 'image2pipe', 'pipe:1'])

  # log the constructed command string if debug
  if loglevel == logging.DEBUG:
    IO.print_subprocess_command(command)

  #####################################
  # prepare neural net input pipeline #
  #####################################
  logging.info('Child process {} is opening image pipe for {}'.format(process_id, video_file_name))
  image_string_len = output_width * output_height * args.numchannels

  # TODO: set buffsize equal to the smallest multiple of a power of two >= batch_size * image_size_in_bytes
  image_pipe = Popen(command, stdout=PIPE, bufsize=batch_size * 2 * 512 ** 2)

  timestamp_array = np.ndarray(
    (args.timestampheight * num_frames, args.timestampmaxwidth, args.numchannels), dtype='uint8')

  num_channels = args.numchannels

  # feed the tf.data input pipeline one image at a time and, while we're at it,
  # extract timestamp overlay crops for later mapping to strings.
  def image_array_generator():
    i = 0

    tx = args.timestampx - args.cropx
    ty = args.timestampy - args.cropy
    th = args.timestampheight
    tw = args.timestampmaxwidth

    while True:
      try:
        image_string = image_pipe.stdout.read(image_string_len)
        if not image_string:
          logging.info('Child process {} is closing image pipe for {}'.format(process_id, video_file_name))
          image_pipe.stdout.close()
          image_pipe.terminate()
          return

        image_array = np.fromstring(image_string, dtype=np.uint8)
        image_array = np.reshape(image_array, (output_height, output_width, num_channels))
        timestamp_array[th * i:th * (i + 1)] = image_array[ty:ty + th, tx:tx + tw]
        i += 1
        yield image_array
      except Exception as e:
        logging.error('An unexpected error occured after processing {} frames from {}.'
                      .format(num_processed_frames, video_file_name))
        logging.error(e)
        logging.error('Child process {} is closing image pipe for {}'.format(
          process_id, video_file_name))
        image_pipe.stdout.close()
        image_pipe.terminate()
        logging.error('Child process {} is raising exception to caller.'.format(process_id))
        raise e

  logging.debug('Child process {} is constructing image dataset pipeline'.format(process_id))
  image_dataset = tf.data.Dataset.from_generator(
    image_array_generator, tf.uint8, tf.TensorShape([output_height, output_width, args.numchannels]))

  num_physical_cpu_cores = int(os.cpu_count() / device_count)

  image_dataset = image_dataset.map(preprocess_for_inception,
                                    num_parallel_calls=num_physical_cpu_cores)

  image_dataset = image_dataset.batch(batch_size)

  image_dataset = image_dataset.prefetch(batch_size)

  next_batch = image_dataset.make_one_shot_iterator().get_next()

  # pre-allocate memory for prediction storage
  prob_array = np.ndarray((num_frames, num_classes), dtype=np.float32)

  #################
  # run inference #
  #################
  logging.debug('Child process {} is starting inference on video at path {}'.format(
    process_id, video_file_path))
  if loglevel == logging.INFO or loglevel == logging.DEBUG:
    logging.debug('Child process {} is starting inference on video at path {}'.format(
      process_id, video_file_path))

  with tf.device('/cpu:0') if device_type == 'cpu' else tf.device(None):
    with tf.Session(graph=session_graph, config=session_config) as session:
      input_tensor = session.graph.get_tensor_by_name(input_tensor_name)
      output_tensor = session.graph.get_tensor_by_name(output_tensor_name)
      image_size_tensor = session.graph.get_tensor_by_name(image_size_tensor_name)

      num_processed_frames = 0

      while True:
        try:
          image_batch = session.run(next_batch)
          probs = session.run(output_tensor, {input_tensor: image_batch,
                                              image_size_tensor: args.modelinputsize})
          num_probs = probs.shape[0]
          prob_array[num_processed_frames:num_processed_frames + num_probs] = probs
          num_processed_frames += num_probs
        except tf.errors.OutOfRangeError:
          logging.info('Child process {} has completed inference on video at path {}'.format(
            process_id, video_file_path))
          if loglevel == logging.INFO or loglevel == logging.DEBUG:
            print('Child process {} has completed inference on video at path {}'.format(
              process_id, video_file_path))
          break

      assert num_processed_frames == num_frames

  return prob_array, timestamp_array


def multi_process_video(
    video_file_path, tensor_name_map, class_names, model_map, device_id_queue,
    child_process_semaphore, logqueue, device_type, device_count):
  # Configure logging for this process
  qh = logging.handlers.QueueHandler(logqueue)
  root = logging.getLogger()

  # Clear any handlers to avoid duplicate entries
  if root.hasHandlers():
    root.handlers.clear()
  root.setLevel(loglevel)
  root.addHandler(qh)

  # Should this be set to match our command line arg, or should we always output this level of detail?
  tf.logging.set_verbosity(tf.logging.INFO)

  def interrupt_handler(signal_number, _):
    logging.warning('Received interrupt signal (%d).', signal_number)
    try:
      logging.info('Unsetting CUDA_VISIBLE_DEVICES environment variable.')
      os.environ.pop('CUDA_VISIBLE_DEVICES')
    except KeyError as ke:
      tf.logging.error(ke)
    logging.warning('Signaling logger to terminate.')
    logqueue.put(None)
    sys.exit(0)

  signal.signal(signal.SIGINT, interrupt_handler)

  process_id = os.getpid()

  video_file_name = path.basename(video_file_path)
  video_file_name, _ = path.splitext(video_file_name)

  logging.info('Child process {} is preparing to process {}'.format(process_id, video_file_name))

  try:
    video_meta_map = IO.read_video_metadata(video_file_path, FFPROBE_PATH)
  except Exception as e:
    logging.debug('Process {} received ffprobe error: {}'.format(
      process_id, e))

    logging.debug('Child process {} released child_process_semaphore back to parent process {}'
                  .format(process_id, os.getppid()))
    child_process_semaphore.release()

    logging.debug('Child process {} terminated itself'.format(process_id))
    exit()

  except Exception as e:
    tf.logging.error('An unexpected error occured while probing {} for metadata.'.format(
      video_file_name))

    tf.logging.error('Throwing exception to main process.')

    raise e

  device_id = device_id_queue.get()
  logging.debug('Child process {} acquired {} device_id {}'.format(
    process_id, device_type, device_id))

  if device_type == 'gpu':
    logging.info('Child process {} is setting CUDA_VISIBLE_DEVICES environment variable to {}.'
                 .format(process_id, device_id))
    os.environ['CUDA_VISIBLE_DEVICES'] = device_id
  else:
    logging.info('Setting CUDA_VISIBLE_DEVICES environment variable to None.')
    os.environ['CUDA_VISIBLE_DEVICES'] = ''

  session_name = model_map['session_name']

  batch_size = args.batchsize

  attempts = 0

  while attempts < 3:
    try:
      start = time.time()

      class_name_probs, timestamps = infer_class_names(
        video_file_path=video_file_path,
        video_file_name=video_file_name,
        output_width=video_meta_map['width'],
        output_height=video_meta_map['height'],
        num_frames=video_meta_map['frame_count'],
        session_config=model_map['session_config'],
        session_graph=model_map['session_graph'],
        input_tensor_name=session_name + '/' + tensor_name_map['input_tensor_name'],
        output_tensor_name=session_name + '/' + tensor_name_map['output_tensor_name'],
        image_size_tensor_name=session_name + '/' + tensor_name_map['image_size_tensor_name'],
        batch_size=batch_size,
        num_classes=len(class_names),
        device_type=device_type,
        device_count=device_count)

      end = time.time() - start

      logging.debug('Child process {} released device_id {}'.format(process_id, device_id))
      device_id_queue.put(device_id)

      IO.print_processing_duration(
        end, 'Child process {} processed {} video frames for {} in'.format(
          process_id, len(class_name_probs), video_file_name), loglevel)

      break
    # TODO: permanently update the batch size so as to not waste time on future batches.
    # in the limit, we should only detect OOM once and update a shared batch size variable
    # to benefit all future videos within the current app run.
    except tf.errors.ResourceExhaustedError as ree:
      batch_size = int(batch_size / 2)
      # If an error occurs, retry up to two times
      logging.warning(ree)
      logging.warning('Resources reportedly exhausted. Inference will be reattempted with a new '
                      'batch size of {}'.format(batch_size))
      attempts += 1

      if attempts < 3:
        tf.logging.error('Re-attempting inference for {}'.format(video_file_name))
      else:
        logging.debug('Child process {} released device_id {}'.format(process_id, device_id))
        device_id_queue.put(device_id)

        logging.debug('Child process {} released child_process_semaphore back to parent process {}'
                      .format(process_id, os.getppid()))
        child_process_semaphore.release()

        logging.debug('Child process {} terminated itself'.format(process_id))
        exit()
    except tf.errors.InternalError as ie:
      tf.logging.error('An internal error occured while running inference on {}\n{}'.format(
        video_file_name, ie))

      attempts += 1

      if attempts < 3:
        tf.logging.error('Re-attempting inference for {}'.format(video_file_name))
      else:
        logging.debug('Child process {} released device_id {}'.format(process_id, device_id))
        device_id_queue.put(device_id)

        logging.debug('Child process {} released child_process_semaphore back to parent process {}'
                      .format(process_id, os.getppid()))
        child_process_semaphore.release()

        logging.debug('Child process {} terminated itself'.format(process_id))
        exit()
    except Exception as e:
      tf.logging.error('An unexpected error occured while running inference on {}\n{}'.format(
        video_file_name, e))

      logging.debug('Child process {} released device_id {}'.format(process_id, device_id))
      device_id_queue.put(device_id)

      logging.debug('Child process {} released child_process_semaphore back to parent process {}'
                    .format(process_id, os.getppid()))
      child_process_semaphore.release()

      logging.debug('Child process {} terminated itself'.format(process_id))
      exit()

  try:
    start = time.time()

    timestamp_object = Timestamp(args.timestampheight, args.timestampmaxwidth)
    stimestamp_strings = timestamp_object.stringify_timestamps(timestamps)

    end = time.time() - start

    IO.print_processing_duration(
      end, 'Child process {} converted timestamp images to strings for {} in'.format(
        process_id, video_file_name), loglevel)
  except Exception as e:
    tf.logging.error(
      'An unexpected error occured while converting timestamp images to strings for {}'.
        format(video_file_name))
    tf.logging.error('Throwing exception to main process')
    raise e

  try:
    start = time.time()

    IO.write_report(
      video_file_name, args.reportpath, stimestamp_strings, class_name_probs, class_names,
      args.smoothprobs, args.smoothingfactor, args.binarizeprobs, process_id)

    end = time.time() - start

    IO.print_processing_duration(
      end, 'Child process {} generated report for {} in'.format(
        process_id, video_file_name), loglevel)
  except Exception as e:
    tf.logging.error('An unexpected error occured while generating report on {}'.
                     format(video_file_name))
    tf.logging.error('Throwing exception to main process')
    raise e

  logging.debug('Child process {} released child_process_semaphore back to parent process {}'.
                format(process_id, os.getppid()))
  child_process_semaphore.release()


def main():
  start = time.time()

  # Create a queue to handle log requests from multiple processes
  logqueue = Queue()
  # Configure our log in the main process to write to a file
  if path.exists(args.logpath):
    if path.isfile(args.logpath):
      raise ValueError('The specified logpath {} is expected to be a directory, not a file.'.format(args.logpath))
  else:
    logging.info("Creating log directory {}".format(args.logpath))

    if loglevel == logging.INFO or loglevel == logging.DEBUG:
      print("Creating log directory {}".format(args.logpath))

    os.makedirs(args.logpath)

  logging.basicConfig(filename=path.join(args.logpath, dt.now().strftime('snva_%m_%d_%Y.log')), level=loglevel,
                      format='%(processName)-10s:%(asctime)s:%(levelname)s::%(message)s')
  # Start our listener thread
  lp = Thread(target=logger_thread, args=(logqueue,))
  lp.start()
  logging.info('Entering main process')
  logging.debug('FFMPEG Path: {}'.format(FFMPEG_PATH))
  logging.debug('FFPROBE Path: {}'.format(FFPROBE_PATH))

  if path.isdir(args.videopath):
    video_dir_path = args.videopath
    video_file_names = IO.read_video_file_names(video_dir_path)
  elif path.isfile(args.videopath):
    video_dir_path, video_file_name = path.split(args.videopath)
    video_file_names = [video_file_name]
  else:
    raise ValueError('The video file/folder specified at the path {} could not be found.'.format(
      args.videopath))

  # TODO modelpath should not be required to be passed at the command line
  if not path.isfile(args.modelpath):
    raise ValueError('The model specified at the path {} could not be found.'.format(
      args.modelpath))

  if args.iotensornamespath is None or not path.isfile(args.iotensornamespath):
    model_dir_path, _ = path.split(args.modelpath)
    io_tensor_names_path = path.join(model_dir_path, 'io_tensor_names.txt')
  else:
    io_tensor_names_path = args.iotensornamespath
  logging.debug('io tensors path set to: {}'.format(io_tensor_names_path))

  if args.classnamespath is None or not path.isfile(args.classnamespath):
    model_dir_path, _ = path.split(args.modelpath)
    class_names_path = path.join(model_dir_path, 'class_names.txt')
  else:
    class_names_path = args.classnamespath
  logging.debug('labels path set to: {}'.format(class_names_path))

  if args.cpuonly:
    device_id_list = ['0']
    device_type = 'cpu'
  else:
    device_id_list = IO.get_device_ids()
    device_type = 'gpu'

  device_id_list_len = len(device_id_list)

  logging.info('Found {} available {} device(s).'.format(device_id_list_len, device_type))

  # child processes will dequeue and enqueue device names
  device_id_queue = Queue(device_id_list_len)

  for device_id in device_id_list:
    device_id_queue.put(device_id)

  label_map = IO.read_class_names(class_names_path)
  class_name_list = list(label_map.values())
  tensor_name_map = IO.read_tensor_names(io_tensor_names_path)
  model_map = IO.load_model(args.modelpath, device_type, args.gpumemoryfraction)

  # The chief worker will allow at most device_count + 1 child processes to be created
  # since the greatest number of concurrent operations is the number of compute devices
  # plus one for IO
  child_process_semaphore = BoundedSemaphore(device_id_list_len + 1)
  child_process_list = []
  logging.info('Processing {} videos in directory: {}'.format(len(video_file_names),
                                                              video_dir_path))
  print('\nProcessing {} videos in directory: {}'.format(len(video_file_names), video_dir_path))

  unprocessed_video_file_names = []

  def call_multi_process_video(video_file_name, child_process_semaphore):
    # Before popping the next video off of the list and creating a process to scan it,
    # check to see if fewer than device_id_list_len + 1 processes are active. If not,
    # Wait for a child process to release its semaphore acquisition. If so, acquire the
    # semaphore, pop the next video name, create the next child process, and pass the
    # semaphore to it
    video_file_path = path.join(video_dir_path, video_file_name)

    if device_id_list_len > 1:
      logging.debug('Creating new child process.')

      if loglevel == logging.INFO or loglevel == logging.DEBUG:
        print('Creating new child process.')

      child_process = Process(target=multi_process_video,
                              name='ChildProcess:{}'.format(video_file_name),
                              args=(video_file_path, tensor_name_map, class_name_list,
                                    model_map, device_id_queue, child_process_semaphore,
                                    logqueue, device_type, device_id_list_len))

      logging.debug('Starting starting child process.')

      if loglevel == logging.INFO or loglevel == logging.DEBUG:
        print('Starting starting child process.')

      child_process.start()

      child_process_list.append(child_process)
    else:
      logging.info('Invoking multi_process_video() in main process because device_type == {}'.format(device_type))
      multi_process_video(video_file_path, tensor_name_map, class_name_list, model_map, device_id_queue,
                          child_process_semaphore, logqueue, device_type, device_id_list_len)

  while len(video_file_names) > 0:
    child_process_semaphore.acquire()  # block if three child processes are active
    logging.debug('Main process {} acquired child_process_semaphore'.format(os.getpid()))

    video_file_name = video_file_names.pop()

    try:
      call_multi_process_video(video_file_name, child_process_semaphore)
    except Exception as e:
      tf.logging.error('An unknown error has occured. Appending {} to the end of '
                       'video_file_names to re-attemt processing later'.format(video_file_name))
      tf.logging.error(e)
      unprocessed_video_file_names.append(video_file_name)

      tf.logging.error('Releasing child_process_semaphore for child process with raised exception.')
      child_process_semaphore.release()

  logging.info('Joining remaining active child processes.')

  if loglevel == logging.INFO or loglevel == logging.DEBUG:
    print('Joining remaining active child processes.')

  for child_process in child_process_list:
    if child_process.is_alive():
      logging.debug('Joining child process {}'.format(child_process.pid))

      if loglevel == logging.INFO or loglevel == logging.DEBUG:
        print('Joining child process {}'.format(child_process.pid))

      child_process.join()

  if len(unprocessed_video_file_names) > 0:
    logging.info('Re-attempting to process any video that did not succeed on the first attempt')

    if loglevel == logging.INFO or loglevel == logging.DEBUG:
      print('Re-attempting to process any video that did not succeed on the first attempt')

    child_process_list.clear()

    while len(unprocessed_video_file_names) > 0:
      child_process_semaphore.acquire()  # block if three child processes are active
      logging.debug('Main process {} acquired child_process_semaphore'.format(os.getpid()))

      video_file_name = unprocessed_video_file_names.pop()
      try:
        call_multi_process_video(video_file_name, child_process_semaphore)
      except Exception as e:
        tf.logging.error('An unknown error has occured during the second attempt to process {}. No further attempts '
                         'will be made'.format(video_file_name))
        tf.logging.error(e)
        tf.logging.error('Releasing child_process_semaphore for child process with raised exception.')
        child_process_semaphore.release()

    logging.info('Joining remaining active child processes.')

    if loglevel == logging.INFO or loglevel == logging.DEBUG:
      print('Joining remaining active child processes.')

    for child_process in child_process_list:
      if child_process.is_alive():
        logging.debug('Joining child process {}'.format(child_process.pid))

        if loglevel == logging.INFO or loglevel == logging.DEBUG:
          print('Joining child process {}'.format(child_process.pid))
        child_process.join()

  end = time.time() - start

  IO.print_processing_duration(end, 'Video processing completed with total elapsed time: ', loglevel)
  print('Video processing completed')
  # Signal the logging thread to finish up
  logging.debug('Signaling log queue to end service.')
  logqueue.put(None)
  lp.join()


if __name__ == '__main__':
  main()