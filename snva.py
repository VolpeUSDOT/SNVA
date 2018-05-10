import logging
import logging.handlers
import argparse
from multiprocessing import BoundedSemaphore, Process, Queue
import threading
import numpy as np
import os
import platform
# import psutil
import signal
import subprocess
import sys
import tensorflow as tf
import time
import uuid
from datetime import datetime as dt
from snva.timestamp_utils import numerize_timestamps
from snva.io_utils import IOObject

path = os.path

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

if platform.system() == 'Windows':
  FFMPEG_PATH = 'ffmpeg.exe'
  FFPROBE_PATH = 'ffprobe.exe'
else:
  FFMPEG_PATH = '/usr/local/bin/ffmpeg' if path.exists('/usr/local/bin/ffmpeg') \
    else '/usr/bin/ffmpeg'
  FFPROBE_PATH = '/usr/local/bin/ffprobe' if path.exists('/usr/local/bin/ffprobe') \
    else '/usr/bin/ffprobe'

parser = argparse.ArgumentParser(description='Process some video files using Tensorflow!')

parser.add_argument('--batchsize', '-bs', type=int, default=32,
                    help='')
parser.add_argument('--binarizeprobs', '-b', action='store_true',
                    help='')
parser.add_argument('--crop', '-c', action='store_true',
                    help='Crop video frames to [offsetheight, offsetwidth, targetheight, targetwidth]')
parser.add_argument('--cropheight', '-ch', type=int, default=356,
                    help='y-component of bottom-right corner of crop.')
parser.add_argument('--cropwidth', '-cw', type=int, default=474,
                    help='x-component of bottom-right corner of crop.')
parser.add_argument('--cropx', '-cx', type=int, default=2,
                    help='x-component of top-left corner of crop.')
parser.add_argument('--cropy', '-cy', type=int, default=0,
                    help='y-component of top-left corner of crop.')
parser.add_argument('--gpumemoryfraction', '-gmf', type=float, default=0.9,
                    help='Percentage of GPU memory to permit this process to consume.')
parser.add_argument('--modelinputsize', '-mis', type=int, default=299,
                    help='')
parser.add_argument('--logpath', '-l', default='./logs',
                    help='Path to the directory where log files are stored.')
parser.add_argument('--numchannels', '-nc', type=int, default=3,
                    help='The fourth dimension of image batches.')
parser.add_argument('--modelpath', '-mp', required=True,
                    help='Path to the tensorflow protobuf model file.')
parser.add_argument('--outputheight', '-oh', type=int, default=299,
                    help='Height of the image frame for processing. ')
parser.add_argument('--outputwidth', '-ow', type=int, default=299,
                    help='Width of the image frame for processing. ')
parser.add_argument('--reportpath', '-rp', default='./results',
                    help='Path to the directory where results are stored.')
parser.add_argument('--scale', '-s', action='store_true',
                    help='')
parser.add_argument('--smoothingfactor', '-sf', type=int, default=0,
                    help='Apply a smoothing factor to detection results.')
parser.add_argument('--timestampheight', '-th', type=int, default=16,
                    help='y-component of bottom-right corner of timestamp.')
parser.add_argument('--timestampwidth', '-tw', type=int, default=160,
                    help='x-component of bottom-right corner of timestamp.')
parser.add_argument('--timestampx', '-tx', type=int, default=25,
                    help='x-component of top-left corner of timestamp.')
parser.add_argument('--timestampy', '-ty', type=int, default=340,
                    help='y-component of top-left corner of timestamp.')
parser.add_argument('--videopath', '-v', required=True,
                    help='Path to video file(s).')
parser.add_argument('--verbose', '-vb', action='store_true',
                    help='Print additional information in logs')
parser.add_argument('--debug', '-d', action='store_true',
                    help='Print debug information in logs')

args = parser.parse_args()

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
            logging.debug("Terminating log thread")
            break
        logger = logging.getLogger(record.name)
        logger.handle(record)

def interrupt_handler(signal_number, _):
  tf.logging.info(
    'Received interrupt signal (%d). Unsetting CUDA_VISIBLE_DEVICES environment variable.', signal_number)
  os.unsetenv('CUDA_VISIBLE_DEVICES')
  logging.warning("Interrupt signal recieved: Exiting...")
  sys.exit(0)


def preprocess_for_inception(image):
  logging.debug("Preprocessing image...")
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
    video_file_path, output_width, output_height, num_frames, session_config, session_graph,
    input_tensor_name, output_tensor_name, image_size_tensor_name, batch_size, num_classes):

  ############################
  # construct ffmpeg command #
  ############################
  logging.debug("Constructing ffmpeg command")
  command = [FFMPEG_PATH, '-i', video_file_path]

  filter_args = []

  if args.crop and all([output_width >= args.cropwidth > 0, output_height >= args.cropheight > 0,
                        output_width > args.cropx >= 0, output_height > args.cropy >= 0]):
    filter_args.append('crop=w={}:h={}:x={}:y={}'.format(
      args.cropwidth, args.cropheight, args.cropx, args.cropy))

    output_width = args.cropwidth
    output_height = args.cropheight

  if args.scale and all([args.outputwidth > 0, args.outputheight > 0]):
    filter_args.append('scale=w={}:h={}'.format(args.outputwidth, args.outputheight))

    output_width = args.outputwidth
    output_height = args.outputheight

  filter_args_len = len(filter_args)

  if filter_args_len > 0:
    command.append('-vf')
    filter = ''
    for i in range(filter_args_len - 1):
      filter += filter_args[i] + ','
    filter += filter_args[filter_args_len - 1]
    command.append(filter)

  command.extend(['-vcodec', 'rawvideo', '-pix_fmt', 'rgb24', '-vsync', 'vfr',
                  '-hide_banner', '-loglevel', '0', '-f', 'image2pipe', 'pipe:1'])

  # log the constructed command string if debug
  if (loglevel == logging.DEBUG):
    command_string = command[0]
  
    for elem in command[1:]:
      command_string += ' ' + elem
  
    logging.debug("FFMPEG Command: {}".format(command_string))

  #####################################
  # prepare neural net input pipeline #
  #####################################
  logging.info("Opening image pipe")
  image_pipe = subprocess.Popen(command, stdout=subprocess.PIPE, bufsize=4 * 1024 * 1024)

  image_string_len = output_width * output_height * args.numchannels

  timestamp_array = np.ndarray((args.timestampheight * num_frames, args.timestampwidth, args.numchannels), dtype='uint8')

  tx = args.timestampx - args.cropx
  ty = args.timestampy - args.cropy
  th = args.timestampheight
  tw = args.timestampwidth

  num_channels = args.numchannels
  # feed the tf.data input pipeline one image at a time and, while we're at it,
  # extract timestamp crops and add them to timestamp_array
  def image_array_generator():
    i = 0
    while True:
      image_string = image_pipe.stdout.read(image_string_len)
      if not image_string:
        logging.info("Closing image pipe")
        image_pipe.stdout.close()
        return
      image_pipe.stdout.flush()
      image_array = np.fromstring(image_string, dtype=np.uint8)
      image_array = np.reshape(image_array, (output_height, output_width, num_channels))
      timestamp_array[th * i:th * (i + 1)] = image_array[ty:ty+th, tx:tx+tw]
      i += 1

      yield image_array

  logging.debug("Constructing image dataset")
  image_dataset = tf.data.Dataset.from_generator(
    image_array_generator, tf.uint8, tf.TensorShape([output_height, output_width, args.numchannels]))

  num_physical_cpu_cores = int(len(os.sched_getaffinity(0)) / 2)

  image_dataset = image_dataset.map(preprocess_for_inception,
                                    num_parallel_calls=num_physical_cpu_cores)

  image_dataset = image_dataset.batch(batch_size)

  next_batch = image_dataset.make_one_shot_iterator().get_next()

  # pre-allocate memory for prediction storage
  prob_array = np.ndarray((num_frames, num_classes), dtype=np.float32)

  #################
  # run inference #
  #################
  logging.debug("Starting inference on video {}".format(video_file_name))
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
        logging.debug("Inference complete")
        break

    assert num_processed_frames == num_frames

  return prob_array, timestamp_array


def multi_process_video(video_file_name, tensor_name_map, label_map, model_map,
                        device_num_queue, io_object_queue, child_process_semaphore, logqueue):
  
  # Configure logging for this process
  qh = logging.handlers.QueueHandler(logqueue)
  root = logging.getLogger()
  # Clear any handlers to avoid duplicate entries
  if (root.hasHandlers()):
    root.handlers.clear()
  root.setLevel(loglevel)
  root.addHandler(qh)
  
  process_id = os.getpid()
  logging.info("Start of process {} for video {}".format(process_id, video_file_name))

  # Should this be set to match our command line arg, or should we always output this level of detail?
  tf.logging.set_verbosity(tf.logging.INFO)

  signal.signal(signal.SIGINT, interrupt_handler)

  video_file_path = os.path.join(args.videopath, video_file_name)
  video_file_name, _ = path.splitext(video_file_name)

  io_object = io_object_queue.get()
  logging.debug('io_object acquired by process {}'.format(process_id))
  video_meta_map = io_object.read_video_metadata(video_file_path)

  device_num = device_num_queue.get()
  logging.debug('device_num {} acquired by child process {}'.format(device_num, process_id))

  tf.logging.info('Setting CUDA_VISIBLE_DEVICES environment variable to {} in child process {}.'.
                  format(device_num, process_id))
  os.putenv('CUDA_VISIBLE_DEVICES', device_num)

  logging.debug('io_object released by child process {}'.format(process_id))
  io_object_queue.put(io_object)

  session_name = model_map['session_name']

  batch_size = args.batchsize

  successful = False
  attempts = 0

  while not successful and attempts < 3:
    try:
      start = time.time()

      class_name_probs, timestamps = infer_class_names(
        video_file_path=video_file_path,
        output_width=video_meta_map['width'],
        output_height=video_meta_map['height'],
        num_frames=video_meta_map['frame_count'],
        session_config=model_map['session_config'],
        session_graph=model_map['session_graph'],
        input_tensor_name=session_name + '/' + tensor_name_map['input_tensor_name'],
        output_tensor_name=session_name + '/' + tensor_name_map['output_tensor_name'],
        image_size_tensor_name=session_name + '/' + tensor_name_map['image_size_tensor_name'],
        batch_size=batch_size,
        num_classes=len(label_map))

      end = time.time() - start

      successful = True
    # TODO: permanently update the batch size so as to not waste time on future batches.
    # in the limit, we should only detect OOM once and update a shared batch size variable
    # to benefit all future videos within the current app run.
    except tf.errors.ResourceExhaustedError as ree:
      batch_size = int(batch_size / 2)
      # If an error occurs, retry up to two times
      logging.warning(ree)
      logging.warning('Resources reportedly exhausted, most likely due to beign out of memory. '
            'Inference will be reattempted with a new batch size of {}'.format(batch_size))
      attempts += 1
    except tf.errors.InternalError as ie:
      logging.error(ie)
      logging.error('Internal error reported, most likely due to a failed session creation attempt.')
      attempts += 1
    except Exception as e:
      logging.error("An unexpected error occured")
      logging.error(e)

  logging.debug('device_num {} released by child process {}'.format(device_num, process_id))
  device_num_queue.put(device_num)

  if successful:
    io_object = io_object_queue.get()
    logging.debug('io_object acquired by child process {}'.format(process_id))

    io_object.print_processing_duration(
      end, 'Time elapsed processing {} video frames for {} by child process {}'.format(
        len(class_name_probs), video_file_name, process_id))

    start = time.time()

    numeral_timestamps = numerize_timestamps(timestamps, args.timestampheight)

    end = time.time() - start

    io_object.print_processing_duration(
      end, 'Time elapsed converting timestamp images to timestamp numeral for {} by child process {}'.format(
        video_file_name, process_id))

    start = time.time()

    io_object.write_report(
      video_file_name, args.reportpath, numeral_timestamps, class_name_probs,
      list(label_map.values()), args.smoothingfactor, args.binarizeprobs)

    end = time.time() - start

    io_object.print_processing_duration(
      end, 'Time elapsed generating report for {} by child process {}'.format(video_file_name,
                                                                              process_id))

    logging.debug('child_process_semaphore released by child process {} back to parent process {}'.
          format(process_id, os.getppid()))
    child_process_semaphore.release()

    logging.debug('io_object released by child process {}'.format(process_id))
    io_object_queue.put(io_object)


if __name__ == '__main__':
  
  # Create a queue to handle log requests from multiple processes
  logqueue = Queue()
  # Configure our log in the main process to write to a file
  logging.basicConfig(filename=dt.now().strftime('snva_%m_%d_%Y.log'),level=loglevel,format='%(processName)-10s:%(asctime)s:%(levelname)s::%(message)s')
  # Start our listener thread
  lp = threading.Thread(target=logger_thread, args=(logqueue,))
  lp.start()
  logging.info("Entering main process")

  start = time.time()

  process_id = os.getpid()

  io_object = IOObject()

  video_file_names = io_object.load_video_file_names(args.videopath) \
    if path.isdir(args.videopath) else [args.videopath]

  if args.modelpath.endswith('.pb'):
    tensorpath = args.modelpath[:-3] + '-meta.txt'
    labelpath = args.modelpath[:-3] + '-labels.txt'
  else:
    tensorpath = args.modelpath + '-meta.txt'
    labelpath = args.modelpath + '-labels.txt'

  logging.debug("Tensorpath set: {}".format(tensorpath))
  logging.debug("Labelpath set: {}".format(labelpath))
  label_map = io_object.load_labels(labelpath)
  tensor_name_map = io_object.load_tensor_names(tensorpath)
  model_map = io_object.load_model(args.modelpath, args.gpumemoryfraction)

  device_num_list = io_object.get_gpu_ids()
  device_num_list_len = len(device_num_list)
  logging.info('{} gpu devices available'.format(device_num_list_len))
  # child processes will dequeue and enqueue gpu device names
  device_num_queue = Queue(device_num_list_len)
  for device_name in device_num_list:
    device_num_queue.put(device_name)

  io_object_queue = Queue(1)
  io_object_queue.put(io_object)

  # The chief worker will allow at most device_count + 1 child processes to be created
  # since the greatest number of concurrent operations is the number of compute devices
  # plus one for IO
  child_process_semaphore = BoundedSemaphore(device_num_list_len + 1)

  child_process_list = []
  logging.info("Processing {} videos in specified directory".format(len(video_file_names)))
  while len(video_file_names) > 0:
    # Before popping the next video off of the list and creating a process to scan it,
    # check to see if fewer than device_num_list_len + 1 processes are active. If not,
    # Wait for a child process to release its semaphore acquisition. If so, acquire the
    # semaphore, pop the next video name, create the next child process, and pass the
    # semaphore to it
    child_process_semaphore.acquire()  # block if three child processes are active

    io_object = io_object_queue.get()
    logging.debug('io_object acquired by parent process {}'.format(process_id))
    logging.debug('child_process_semaphore acquired by parent process {}'.format(process_id))
    logging.debug('io_object released by parent process {}'.format(process_id))
    io_object = io_object_queue.put(io_object)

    video_file_name = video_file_names.pop()

    try:
      io_object = io_object_queue.get()
      logging.debug('io_object acquired by parent process {}'.format(process_id))
      logging.debug('creating child process')
      logging.debug('io_object released by parent process {}'.format(process_id))
      io_object = io_object_queue.put(io_object)

      child_process = Process(target=multi_process_video, name='video %s process' % video_file_name,
                              args=(video_file_name, tensor_name_map, label_map, model_map,
                                    device_num_queue, io_object_queue, child_process_semaphore, logqueue))

      io_object = io_object_queue.get()
      logging.debug('io_object acquired by parent process {}'.format(process_id))
      logging.debug('starting child process')
      logging.debug('io_object released by parent process {}'.format(process_id))
      io_object = io_object_queue.put(io_object)

      child_process.start()

      child_process_list.append(child_process)
    except Exception as e:
      logging.exception("Error occured processing video {}".format(video_file_name))
      # If an error occurs, retry (infinitely for now) after processing other videos
      logging.error('An error has occured. Appending {} to the end of video_file_names to re-attemt processing later'.
            format(video_file_name))
      video_file_names.append(video_file_name)

  # io_object = io_object_queue.get()
  # print('io_object acquired by parent process {}'.format(process_id))
  logging.info('joining remaining active children...')
  # print('io_object released by parent process {}'.format(process_id))
  # io_object = io_object_queue.put(io_object)

  for child_process in child_process_list:
    if child_process.is_alive():
      # io_object = io_object_queue.get()
      # print('io_object acquired by parent process {}'.format(process_id))
      logging.debug('joining child process {}'.format(child_process.pid))
      # print('io_object released by parent process {}'.format(process_id))
      # io_object = io_object_queue.put(io_object)
      child_process.join()

  end = time.time() - start

  io_object = io_object_queue.get()
  logging.debug('io_object acquired by parent process {}'.format(process_id))
  io_object.print_processing_duration(end, 'Parent process {} elapsed time'.format(process_id))
  logging.debug('io_object released by parent process {}'.format(process_id))
  io_object = io_object_queue.put(io_object)

  #Signal the logging thread to finish up
  logging.debug('Signaling log queue to stop')
  logqueue.put(None)
  lp.join()
