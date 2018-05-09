import argparse
import csv
import json
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
import logging
import logging.handlers
from datetime import datetime as dt

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

parser.add_argument('--batchsize', '-b', type=int, default=1)
parser.add_argument('--crop', '-c', action='store_true',
                    help='Crop video frames to [offsetheight, offsetwidth, targetheight, targetwidth]')
parser.add_argument('--cropx', '-cx', type=int, help='x-component of top-left corner of crop.')
parser.add_argument('--cropy', '-cy', type=int, help='y-component of top-left corner of crop.')
parser.add_argument('--cropwidth', '-cw', type=int, help='x-component of bottom-right corner of crop.')
parser.add_argument('--cropheight', '-ch', type=int, help='y-component of bottom-right corner of crop.')
parser.add_argument('--gpumemoryfraction', '-gmf', type=float, default=0.9,
                    help='Percentage of GPU memory to permit this process to consume.')
parser.add_argument('--modelinputsize', '-mis', type=int, default=299)
parser.add_argument('--numchannels', '-nc', type=int, default=3,
                    help='The fourth dimension of image batches.')
parser.add_argument('--modelpath', '-mp', required=True,
                    help='Path to the tensorflow protobuf model file.')
parser.add_argument('--outputclips', '-o', dest='outputclips', action='store_true',
                    help='Output results as video clips containing searched for labelname.')
parser.add_argument('--outputheight', '-oh', type=int, default=299,
                    help='Height of the image frame for processing. ')
parser.add_argument('--outputwidth', '-ow', type=int, default=299,
                    help='Width of the image frame for processing. ')
parser.add_argument('--outputpadding', '-op', type=int, default=30,
                    help='Number of seconds added to the start and end of created video clips.')
parser.add_argument('--reportpath', '-rp', default='./results',
                    help='Path to the directory where results are stored.')
parser.add_argument('--scale', '-s', action='store_true', help='')
parser.add_argument('--smoothing', '-sm', type=int, default=0,
                    help='Apply a smoothing factor to detection results.')
parser.add_argument('--videopath', '-v', required=True, help='Path to video file(s).')
parser.add_argument('--verbose', '-v', help='Print additional information in logs', action='store_true')
parser.add_argument('--debug', '-d', help='Print debug information in logs', action='store_true')

args = parser.parse_args()

# Define our log level based on arguments
loglevel = logging.WARNING
if args.verbose:
  loglevel = logging.INFO
if args.debug:
  loglevel = logging.DEBUG

# Configure a process to send its log messages to a queue
def worker_log_config(queue):
  qh = logging.handlers.QueueHandler(queue)
  root = logging.getLogger()
  root.setLevel(loglevel)
  root.addHandler(qh)

# Logger thread: listens for updates to our log queue and writes them as they come in
# Terminates after we add None to the queue
def logger_thread(q):
    while True:
        record = q.get()
        if record is None:
            break
        logger = logging.getLogger(record.name)
        logger.handle(record)

class IOObject:
  @staticmethod
  def get_gpu_nums():
    # TODO: Consider replacing a subprocess invocation with nvml bindings
    command = ['nvidia-smi', '-L']

    pipe = subprocess.run(command, stdout=subprocess.PIPE, encoding='utf-8')

    line_list = pipe.stdout.rstrip().split('\n')
    gpu_nums = [line.split(' ')[1] for line in line_list]
    return [gpu_num.split(':')[0] for gpu_num in gpu_nums]

  @staticmethod
  def load_labels(labels_path):
    meta_map = IOObject.read_meta_file(labels_path)
    return {int(key): value for key, value in meta_map.items()}

  @staticmethod
  def load_model(model_path, gpu_memory_fraction):
    graph_def = tf.GraphDef()
    with tf.gfile.FastGFile(model_path, 'rb') as file:
      graph_def.ParseFromString(file.read())

    session_name = str(uuid.uuid4())

    session_graph = tf.import_graph_def(graph_def, name=session_name)

    gpu_options = tf.GPUOptions(allow_growth=True,
                                per_process_gpu_memory_fraction=gpu_memory_fraction)

    session_config = tf.ConfigProto(allow_soft_placement=True,
                                    # log_device_placement=True,
                                    gpu_options=gpu_options)

    return {'session_name': session_name,
            'session_graph': session_graph,
            'session_config': session_config}

  @staticmethod
  def load_tensor_names(io_tensor_names_path):
    meta_map = IOObject.read_meta_file(io_tensor_names_path)
    return {key: value + ':0' for key, value in meta_map.items()}

  @staticmethod
  def load_video_file_names(video_file_dir_path):
    included_extenstions = ['avi', 'mp4', 'asf', 'mkv']
    return sorted([fn for fn in os.listdir(video_file_dir_path)
                   if any(fn.lower().endswith(ext) for ext in included_extenstions)])

  @staticmethod
  def print_processing_duration(end_time, msg):
    minutes, seconds = divmod(end_time, 60)
    hours, minutes = divmod(minutes, 60)
    print('{:s}: {:02d}:{:02d}:{:02d}\n'.format(
      msg, int(hours), int(minutes), int(seconds)))

  @staticmethod
  def read_meta_file(file_path):
    meta_lines = [line.rstrip().split(':') for line in tf.gfile.GFile(file_path).readlines()]
    return {line[0]: line[1] for line in meta_lines}

  @staticmethod
  def read_video_metadata(video_file_path):
    command = ['ffprobe', '-show_streams', '-print_format',
               'json', '-loglevel', 'quiet', video_file_path]
    pipe = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

    json_string, err = pipe.communicate()

    json_map = json.loads(json_string)

    return {'width': int(json_map['streams'][0]['width']),
            'height': int(json_map['streams'][0]['height']),
            'frame_count': int(json_map['streams'][0]['nb_frames'])}

  @staticmethod
  def smooth_probs(probs, degree=2):
    window = degree * 2 - 1
    weight = np.array([1.0] * window)
    gauss_weight = []
    div_odd = lambda n: (n // 2, n // 2 + 1)

    for i in range(window):
      i = i - degree + 1
      frac = i / float(window)
      gauss = 1 / (np.exp((4 * (frac)) ** 2))
      gauss_weight.append(gauss)

    weight = np.array(gauss_weight) * weight
    
    smoothed_probs = [float("{0:.4f}".format(sum(np.array(probs[i:i + window]) * weight) / sum(weight)))
                      for i in range(len(probs) - window)]

    padfront, padback = div_odd(window)
    for i in range(0, padfront):
      smoothed_probs.insert(0, smoothed_probs[0])
    for i in range(0, padback):
      smoothed_probs.append(smoothed_probs[-1])

    return smoothed_probs

  @staticmethod
  def write_report(video_file_name, report_path, class_probs, class_names, smoothing=0):
    if smoothing > 0:
      class_names = class_names + [class_name + '_smoothed' for class_name in class_names]

      smoothed_probs = [IOObject.smooth_probs(class_probs[:, i], int(smoothing))
                        for i in range(len(class_probs[0]))]
      smoothed_probs = np.array(smoothed_probs)
      smoothed_probs = np.transpose(smoothed_probs)

      class_probs = np.concatenate((class_probs, smoothed_probs), axis=1)

    report_file_path = os.path.join(report_path, video_file_name + '_results.csv')

    with open(report_file_path, 'w', newline='') as logfile:
      csv_writer = csv.writer(logfile)
      csv_writer.writerow(class_names)
      csv_writer.writerows([['{0:.4f}'.format(cls) for cls in class_prob]
                            for class_prob in class_probs])


def interrupt_handler(signal_number, _):
  tf.logging.info(
    'Received interrupt signal (%d). Unsetting CUDA_VISIBLE_DEVICES environment variable.', signal_number)
  os.unsetenv('CUDA_VISIBLE_DEVICES')
  logging.warning("Interrupt signal recieved: Exiting...")
  sys.exit(0)


def preprocess_for_inception(image, height, width):
  if image.dtype != tf.float32:
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
  if image.shape[0] != height or image.shape[1] != width:
    image = tf.expand_dims(image, 0)
    image = tf.image.resize_bilinear(image, [height, width], align_corners=False)
    image = tf.squeeze(image, [0])
  image = tf.subtract(image, 0.5)
  image = tf.multiply(image, 2.0)
  return image


def infer_class_names(
    video_file_path, output_width, output_height, num_frames, session_config, session_graph,
    input_tensor_name, output_tensor_name, image_size_tensor_name, num_classes):
  ############################
  # construct ffmpeg command #
  ############################

  command = [FFMPEG_PATH, '-i', video_file_path]

  filter_args = []

  if args.crop and all([args.cropwidth > 0, args.cropheight > 0, args.cropx >= 0, args.cropy >= 0]):
    filter_args.append('crop={}:{}:{}:{}'.format(
      args.cropwidth, args.cropheight, args.cropx, args.cropy))

    output_width = args.cropwidth
    output_height = args.cropheight

  if args.scale and all([args.outputwidth > 0, args.outputheight > 0]):
    filter_args.append('scale={}:{}'.format(args.outputwidth, args.outputheight))

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
                  '-hide_banner', '-loglevel', '0', '-f', 'image2pipe', '-'])

  # log the constructed command string if debug
  if (loglevel == logging.DEBUG):
    command_string = command[0]
  
    for elem in command[1:]:
      command_string += ' ' + elem
  
    logging.debug("FFMPEG Command: %s" % command_string)

  #####################################
  # prepare neural net input pipeline #
  #####################################

  image_pipe = subprocess.Popen(command, stdout=subprocess.PIPE, bufsize=4 * 1024 * 1024)

  image_string_len = output_width * output_height * args.numchannels

  def image_array_generator():
    while True:
      image_string = image_pipe.stdout.read(image_string_len)
      if not image_string:
        logging.debug("Closing image pipe")
        image_pipe.stdout.close()
        return
      # image_pipe.stdout.flush()  # what does flush even do, really?
      image_array = np.fromstring(image_string, dtype='uint8')
      yield image_array

  image_dataset = tf.data.Dataset.from_generator(
    image_array_generator, tf.uint8, tf.TensorShape([image_string_len]))

  def reshape_and_preprocess_for_inception(image):
    image = tf.reshape(image, (output_height, output_width, args.numchannels))
    return preprocess_for_inception(image, args.modelinputsize, args.modelinputsize)

  num_physical_cpu_cores = int(len(os.sched_getaffinity(0)) / 2)
  image_dataset = image_dataset.map(reshape_and_preprocess_for_inception,
                                    num_parallel_calls=num_physical_cpu_cores)

  image_dataset = image_dataset.batch(args.batchsize)

  # use an iterator that can be reinitialized inside the loop
  batch_iterator = image_dataset.make_initializable_iterator()

  # pre-allocate memory for prediction storage
  prob_array = np.ndarray((num_frames, num_classes), dtype=np.float32)

  #################
  # run inference #
  #################
  logging.debug("Starting inference on video %s" % video_file_name)
  with tf.Session(graph=session_graph, config=session_config) as session:
    session.run(batch_iterator.initializer)

    input_tensor = session.graph.get_tensor_by_name(input_tensor_name)
    output_tensor = session.graph.get_tensor_by_name(output_tensor_name)
    image_size_tensor = session.graph.get_tensor_by_name(image_size_tensor_name)

    num_processed_frames = 0

    while True:
      try:
        image_batch = session.run(batch_iterator.get_next())
        probs = session.run(output_tensor,
                            {input_tensor: image_batch, image_size_tensor: args.modelinputsize})
        num_probs = probs.shape[0]
        prob_array[num_processed_frames:num_processed_frames + num_probs] = probs
        num_processed_frames += num_probs
      except tf.errors.OutOfRangeError:
        break

    assert num_processed_frames == num_frames

  return prob_array


def multi_process_video(video_file_name, tensor_name_map, label_map, model_map,
                        device_num_queue, io_object_queue, child_process_semaphore, logqueue):
  
  # Configure logging for this process
  worker_log_config(logqueue)
  
  process_id = os.getpid()
  logging.debug("Start of process %s for video %s" % process_id, video_file_name)

  tf.logging.set_verbosity(tf.logging.INFO)

  signal.signal(signal.SIGINT, interrupt_handler)

  video_file_path = os.path.join(args.videopath, video_file_name)
  video_file_name, _ = path.splitext(video_file_name)

  io_object = io_object_queue.get()
  logging.info('io_object acquired by process {}'.format(process_id))
  video_meta_map = io_object.read_video_metadata(video_file_path)

  device_num = device_num_queue.get()
  logging.info('device_num {} acquired by child process {}'.format(device_num, process_id))

  tf.logging.info('Setting CUDA_VISIBLE_DEVICES environment variable to {} in child process {}.'.
                  format(device_num, process_id))
  os.putenv('CUDA_VISIBLE_DEVICES', device_num)

  logging.info('io_object released by child process {}'.format(process_id))
  io_object_queue.put(io_object)

  session_name = model_map['session_name']

  start = time.time()

  class_name_probs = infer_class_names(
    video_file_path=video_file_path, 
    output_width=video_meta_map['width'], 
    output_height=video_meta_map['height'], 
    num_frames=video_meta_map['frame_count'], 
    session_config=model_map['session_config'],
    session_graph=model_map['session_graph'], 
    input_tensor_name=session_name + '/' + tensor_name_map['input_tensor_name'], 
    output_tensor_name=session_name + '/' + tensor_name_map['output_tensor_name'], 
    image_size_tensor_name=session_name + '/' + tensor_name_map['image_size_tensor_name'], 
    num_classes=len(label_map))

  end = time.time() - start

  logging.info('device_num {} released by child process {}'.format(device_num, process_id))
  device_num_queue.put(device_num)

  io_object = io_object_queue.get()
  logging.info('io_object acquired by child process {}'.format(process_id))

  io_object.print_processing_duration(
    end, 'Time elapsed processing {} video frames for {} by child process {}'.format(
      len(class_name_probs), video_file_name, process_id))

  start = time.time()

  io_object.write_report(
    video_file_name, args.reportpath, class_name_probs, list(label_map.values()), args.smoothing)

  end = time.time() - start

  io_object.print_processing_duration(
    end, 'Time elapsed generating report for {} by child process {}'.format(video_file_name,
                                                                            process_id))

  logging.info('child_process_semaphore released by child process {} back to parent process {}'.
        format(process_id, os.getppid()))
  child_process_semaphore.release()

  logging.info('io_object released by child process {}'.format(process_id))
  io_object_queue.put(io_object)
  logging.info("End of process %s, video %s complete" % process_id, video_file_name)


if __name__ == '__main__':
  
  start = time.time()

  process_id = os.getpid()

  io_object = IOObject()

  logqueue = Queue()
  logging.basicConfig(filename=dt.now().strftime('snva_%d_%m_%Y.log'),level=loglevel,format='%(processName)-10s:%(asctime)s:%(levelname)s::%(message)s')
  lp = threading.Thread(target=logger_thread, args=(logqueue,))
  lp.start()
  logging.info("Entering main process")

  video_file_names = io_object.load_video_file_names(args.videopath) \
    if path.isdir(args.videopath) else [args.videopath]

  if args.modelpath.endswith('.pb'):
    tensorpath = args.modelpath[:-3] + '-meta.txt'
    labelpath = args.modelpath[:-3] + '-labels.txt'
  else:
    tensorpath = args.modelpath + '-meta.txt'
    labelpath = args.modelpath + '-labels.txt'

  logging.debug("Tensorpath set: %s" % tensorpath)
  logging.debug("Labelpath set: %s" % labelpath)
  label_map = io_object.load_labels(labelpath)
  tensor_name_map = io_object.load_tensor_names(tensorpath)
  model_map = io_object.load_model(args.modelpath, args.gpumemoryfraction)

  device_num_list = io_object.get_gpu_nums()
  device_num_list_len = len(device_num_list)

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

  while len(video_file_names) > 0:
    # Before popping the next video off of the list and creating a process to scan it,
    # check to see if fewer than device_num_list_len + 1 processes are active. If not,
    # Wait for a child process to release its semaphore acquisition. If so, acquire the
    # semaphore, pop the next video name, create the next child process, and pass the
    # semaphore to it
    child_process_semaphore.acquire()  # block if three child processes are active

    io_object = io_object_queue.get()
    logging.info('io_object acquired by parent process {}'.format(process_id))
    logging.info('child_process_semaphore acquired by parent process {}'.format(process_id))
    logging.info('io_object released by parent process {}'.format(process_id))
    io_object = io_object_queue.put(io_object)

    video_file_name = video_file_names.pop()

    try:
      io_object = io_object_queue.get()
      logging.info('io_object acquired by parent process {}'.format(process_id))
      logging.info('creating child process')
      logging.info('io_object released by parent process {}'.format(process_id))
      io_object = io_object_queue.put(io_object)

      child_process = Process(target=multi_process_video, name='video %s process' % video_file_name,
                              args=(video_file_name, tensor_name_map, label_map, model_map,
                                    device_num_queue, io_object_queue, child_process_semaphore, logqueue))

      io_object = io_object_queue.get()
      logging.info('io_object acquired by parent process {}'.format(process_id))
      logging.info('starting child process')
      logging.info('io_object released by parent process {}'.format(process_id))
      io_object = io_object_queue.put(io_object)

      child_process.start()

      child_process_list.append(child_process)
    except Exception as e:
      logging.exception("Error occured processing video %s", video_file_name)
      # If an error occurs, retry (infinitely for now) after processing other videos
      logging.error('An error has occured. Appending {} to the end of video_file_names to re-attemt processing later'.
            format(video_file_name))
      video_file_names.append(video_file_name)

  # io_object = io_object_queue.get()
  # print('io_object acquired by parent process {}'.format(process_id))
  logging.info('joining remaining active children')
  # print('io_object released by parent process {}'.format(process_id))
  # io_object = io_object_queue.put(io_object)

  for child_process in child_process_list:
    if child_process.is_alive():
      # io_object = io_object_queue.get()
      # print('io_object acquired by parent process {}'.format(process_id))
      logging.info('joining child process {}'.format(child_process.pid))
      # print('io_object released by parent process {}'.format(process_id))
      # io_object = io_object_queue.put(io_object)
      child_process.join()

  end = time.time() - start

  io_object = io_object_queue.get()
  logging.info('io_object acquired by parent process {}'.format(process_id))
  io_object.print_processing_duration(end, 'Parent process {} elapsed time'.format(process_id))
  logging.info('io_object released by parent process {}'.format(process_id))
  io_object = io_object_queue.put(io_object)

  #Signal the logging thread to finish up
  logging.info('Signaling log queue to stop')
  logqueue.put(None)
  lp.join()
