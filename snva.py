import argparse
import asyncio
from collections import deque
import json
import logging
from logging.handlers import QueueHandler
from multiprocessing import get_context
import os
import platform
from queue import Empty
import signal
import socket
from threading import Event, Thread
from time import time
from utils.io import IO
from utils.processor import process_video, process_video_signalstate
from utils.logger import configure_logging
from utils.transport import build_control_url, create_client_context
import websockets as ws

path = os.path

logger = logging.getLogger('websockets')
logger.setLevel(logging.INFO)

def main_logger_fn(log_queue):
  while True:
    try:
      message = log_queue.get()
      if message is None:
        break
      logging.getLogger().handle(message)
    except Exception as e:
      logging.error(e)
      break


# Logger thread: listens for updates to log queue and writes them as they arrive
# Terminates after we add None to the queue
def child_logger_fn(main_log_queue, child_log_queue, stop_event=None):
  while stop_event is None or not stop_event.is_set():
    try:
      message = child_log_queue.get(timeout=0.2)
      if message is None:
        break
      if stop_event is not None and stop_event.is_set():
        break
      main_log_queue.put(message)
    except Empty:
      continue
    except Exception as e:
      logging.error(e)
      break


def stringify_command(arg_list):
  command_string = arg_list[0]
  for elem in arg_list[1:]:
    command_string += ' ' + elem
  return 'command string: {}'.format(command_string)


#TODO: accomodate unbounded number of valid process counts
def get_valid_num_processes_per_device(device_type):
  # valid_n_procs = {1, 2}
  # if device_type == 'cpu':
  #   n_cpus = os.cpu_count()
  #   n_procs = 4
  #   while n_procs <= n_cpus:
  #     k = (n_cpus - n_procs) / n_procs
  #     if k == int(k):
  #       valid_n_procs.add(n_procs)
  #     n_procs += 2
  # return valid_n_procs
  return list(range(1, os.cpu_count() + 1))


async def main():
  logging.info('entering snva {} main process'.format(snva_version_string))

  # total_num_video_to_process = None

  def interrupt_handler(signal_number, _):
    logging.warning('Main process received interrupt signal '
                    '{}.'.format(signal_number))
    main_interrupt_queue.put_nowait('_')

    # if total_num_video_to_process is None \
    #     or total_num_video_to_process == len(video_file_paths):

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

  # # TODO validate all video file paths in the provided text file if args.inputpath is a text file
  # if path.isdir(args.inputpath):
  #   video_file_names = set(IO.read_video_file_names(args.inputpath))
  #   video_file_paths = [path.join(args.inputpath, video_file_name)
  #                       for video_file_name in video_file_names]
  # elif path.isfile(args.inputpath):
  #   if args.inputpath[-3:] == 'txt':
  #     if args.inputlistrootdirpath is None:
  #       raise ValueError('--inputlistrootdirpath must be specified when using a'
  #                        ' text file as the input.')
  #     with open(args.inputpath, newline='') as input_file:
  #       video_file_paths = []
  #
  #       for line in input_file.readlines():
  #         line = line.rstrip()
  #         video_file_path = line.lstrip(args.inputlistrootdirpath)
  #         video_file_path = path.join('/media/root', video_file_path)
  #
  #         if path.isfile(video_file_path):
  #           video_file_paths.append(video_file_path)
  #         else:
  #           logging.warning('The video file at host path {} could not be found '
  #                           'at mapped path {} and will not be processed'.
  #             format(line, video_file_path))
  #   else:
  #     video_file_paths = [args.inputpath]
  # else:
  #   raise ValueError('The video file/folder specified at the path {} could '
  #                    'not be found.'.format(args.inputpath))

  models_root_dir_path = path.join(snva_home, args.modelsdirpath)

  models_dir_path = path.join(models_root_dir_path, args.modelname)

  logging.debug('models_dir_path set to {}'.format(models_dir_path))

  # model_file_path = path.join(models_dir_path, args.protobuffilename)
  #
  # if not path.isfile(model_file_path):
  #   raise ValueError('The model specified at the path {} could not be '
  #                    'found.'.format(model_file_path))
  #
  # logging.debug('model_file_path set to {}'.format(model_file_path))

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
  logging.info("Output path set to: {}".format(output_dir_path))
  if not path.isdir(output_dir_path):
    os.makedirs(output_dir_path)

  if args.classnamesfilepath is None \
      or not path.isfile(args.classnamesfilepath):
    class_names_path = path.join(models_root_dir_path, 'class_names.txt')
  else:
    class_names_path = args.classnamesfilepath
  logging.debug('labels path set to: {}'.format(class_names_path))

  num_processes = args.numprocesses

  class_name_map = IO.read_class_names(class_names_path)

  return_code_queue_map = {}
  child_logger_thread_map = {}
  child_process_map = {}
  assignment_map = {}
  completed_result_map = {}
  child_log_queue_map = {}
  child_log_stop_map = {}
  control_messages = deque()
  process_context = get_context('spawn')

  total_num_processed_videos = 0
  total_num_processed_frames = 0
  total_analysis_duration = 0

  def start_video_processor(video_file_path, assignment):
    # Before popping the next video off of the list and creating a process to
    # scan it, check to see if fewer than logical_device_count + 1 processes are
    # active. If not, Wait for a child process to release its semaphore
    # acquisition. If so, acquire the semaphore, pop the next video name,
    # create the next child process, and pass the semaphore to it
    return_code_queue = process_context.Queue()

    assignment_id = assignment['assignmentId']
    return_code_queue_map[assignment_id] = return_code_queue
    assignment_map[assignment_id] = assignment

    logging.debug('creating new child process.')

    child_log_queue = process_context.Queue()
    child_log_queue_map[assignment_id] = child_log_queue

    stop_event = Event()
    child_log_stop_map[assignment_id] = stop_event
    child_logger_thread = Thread(target=child_logger_fn,
                                 args=(log_queue, child_log_queue, stop_event),
                                 daemon=True)

    child_logger_thread.start()
    child_logger_thread_map[assignment_id] = child_logger_thread

    if 'signalstate' == args.processormode:
      child_process = process_context.Process(
        target=process_video_signalstate,
        name=path.splitext(path.split(video_file_path)[1])[0],
        args=(video_file_path, output_dir_path, class_name_map, args.modelname, args.modelsignaturename, args.modelserverhost,model_input_size,
              return_code_queue, child_log_queue, log_level,
              ffmpeg_path, ffprobe_path, args.crop, args.cropwidth, args.cropheight,
              args.cropx, args.cropy, args.extracttimestamps,
              args.timestampmaxwidth, args.timestampheight, args.timestampx,
              args.timestampy, args.deinterlace, args.numchannels, args.batchsize,
              args.smoothprobs, args.smoothingfactor, args.binarizeprobs,
              args.writebbox, args.writeeventreports, args.maxanalyzerthreads, args.processormode,
              args.tls_ca, args.tls_cert, args.tls_key))
    else:
      child_process = process_context.Process(
      target=process_video,
      name=path.splitext(path.split(video_file_path)[1])[0],
      args=(video_file_path, output_dir_path, class_name_map, args.modelname, args.modelsignaturename, args.modelserverhost,model_input_size,
            return_code_queue, child_log_queue, log_level,
            ffmpeg_path, ffprobe_path, args.crop, args.cropwidth, args.cropheight,
            args.cropx, args.cropy, args.extracttimestamps,
            args.timestampmaxwidth, args.timestampheight, args.timestampx,
            args.timestampy, args.deinterlace, args.numchannels, args.batchsize,
            args.smoothprobs, args.smoothingfactor, args.binarizeprobs,
            args.writeinferencereports, args.writeeventreports, args.maxanalyzerthreads, args.processormode,
            args.tls_ca, args.tls_cert, args.tls_key))
    logging.debug('starting child process.')

    child_process_map[assignment_id] = child_process
    child_process.start()

  def close_worker(assignment_id, abort=False):
    child = child_process_map.get(assignment_id)
    forced = False
    if child is not None and child.pid is not None:
      if abort and child.is_alive():
        forced = True
        child.terminate()
      child.join(timeout=15)
      if child.is_alive():
        forced = True
        child.terminate()
        child.join(timeout=15)
      if child.is_alive():
        os.kill(child.pid, signal.SIGKILL)
        child.join()
      forced = forced or child.exitcode not in (None, 0)
    child_queue = child_log_queue_map.get(assignment_id)
    forwarder = child_logger_thread_map.get(assignment_id)
    if child_queue is not None:
      # The producer is stopped before the sentinel; forward all final records.
      child_queue.put(None)
    if forwarder is not None:
      forwarder.join(timeout=1 if forced else None)
      if forced:
        # A killed Queue producer can abandon a pipe/write lock mid-record.
        child_log_stop_map[assignment_id].set()
        if forwarder.is_alive():
          logging.warning('abandoning corrupted worker log queue after forced termination')
    for queue in (child_queue, return_code_queue_map.get(assignment_id)):
      if queue is not None:
        if forced:
          queue.cancel_join_thread()
        queue.close()
        if not forced:
          queue.join_thread()
    for mapping in (return_code_queue_map, child_logger_thread_map,
                    child_process_map, child_log_queue_map, assignment_map,
                    completed_result_map, child_log_stop_map):
      mapping.pop(assignment_id, None)

  async def receive_message(conn):
    if control_messages:
      return control_messages.popleft()
    return json.loads(await conn.recv())

  async def close_completed_video_processors(websocket_conn):
    nonlocal total_num_processed_videos, total_num_processed_frames
    nonlocal total_analysis_duration
    for assignment_id in list(return_code_queue_map.keys()):
      return_code_queue = return_code_queue_map[assignment_id]
      assignment = assignment_map[assignment_id]

      try:
        if assignment_id not in completed_result_map:
          completed_result_map[assignment_id] = return_code_queue.get_nowait()
        return_code_map = completed_result_map[assignment_id]

        return_code = return_code_map['return_code']
        return_value = return_code_map['return_value']

        child_process = child_process_map[assignment_id]

        logging.debug(
          'child process {} returned with exit code {} and exit value '
          '{}'.format(child_process.pid, return_code, return_value))

        if return_code == 'success':
          logging.info('notifying control node of completion')

          complete_request = json.dumps({
            'action': 'COMPLETE',
            'video': assignment['path'],
            'assignmentId': assignment_id,
            'output': return_code_map['output_locations']})
          await websocket_conn.send(complete_request)
          while True:
            receipt = json.loads(await asyncio.wait_for(websocket_conn.recv(), 30))
            if receipt.get('action') == 'COMPLETE_ACCEPTED':
              if (receipt.get('video') == assignment['path']
                  and receipt.get('assignmentId') == assignment_id):
                break
              continue
            control_messages.append(receipt)
          total_num_processed_videos += 1
          total_num_processed_frames += return_value
          total_analysis_duration += return_code_map['analysis_duration']

        close_worker(assignment_id)
      except Empty:
        pass

  start = time()

  sleep_duration = 1
  breakLoop = False
  shutdown_requested = False
  connectionId = None
  reconnect_token = None
  control_tls = create_client_context(args.tls_ca, args.tls_cert, args.tls_key)
  isIdle = False
  try:
    while not breakLoop:
      try:
        wsUrl = build_control_url(args.controlnodehost, connectionId, reconnect_token)
        logging.debug("Connecting to secure control node %s", args.controlnodehost)
        async with ws.connect(wsUrl, ssl=control_tls) as conn:
          response = json.loads(await conn.recv())
          if response['action'] != 'CONNECTION_SUCCESS':
            raise ConnectionError('control node registration failed')
          if connectionId is None:
            connectionId = response['id']
          reconnect_token = response['reconnectToken']
          logging.debug("Assigned id {}".format(connectionId))
          # Replay locally sent but unconfirmed results before requesting work.
          if completed_result_map:
            await close_completed_video_processors(conn)
          while not shutdown_requested:
            while len(return_code_queue_map) >= num_processes:
              await close_completed_video_processors(conn)
              await asyncio.sleep(sleep_duration)
            try:
              main_interrupt_queue.get_nowait()
              shutdown_requested = True
              break
            except Empty:
              pass
            if control_messages:
              response = await receive_message(conn)
            elif not isIdle:
              logging.info('requesting video')
              await conn.send(json.dumps({'action': 'REQUEST_VIDEO'}))
              response = await receive_message(conn)
            else:
              response = None
              while return_code_queue_map:
                try:
                  response = await asyncio.wait_for(receive_message(conn), 1)
                  break
                except asyncio.TimeoutError:
                  await close_completed_video_processors(conn)
                  if control_messages:
                    response = await receive_message(conn)
                    break
                  if return_code_queue_map:
                    await asyncio.sleep(sleep_duration)
              if response is None:
                response = await receive_message(conn)
            if response['action'] == 'STATUS_REQUEST':
              logging.info('control node requested status request')
            elif response['action'] == 'CEASE_REQUESTS':
              isIdle = True
            elif response['action'] == 'RESUME_REQUESTS':
              isIdle = False
            elif response['action'] == 'SHUTDOWN':
              logging.info('control node requested shutdown')
              shutdown_requested = True
            elif response['action'] == 'COMPLETE_ACCEPTED':
              # A duplicate receipt cannot release or count another assignment.
              continue
            elif response['action'] == 'PROCESS':
              if (not isinstance(response.get('path'), str) or not response['path']
                  or not isinstance(response.get('assignmentId'), str)
                  or not response['assignmentId']
                  or response['assignmentId'] in assignment_map):
                raise ValueError('invalid or duplicate video assignment')
              video_file_path = os.path.join(args.inputpath, response['path'])
              await conn.send(json.dumps({
                'action': 'REQUEST_RECEIVED', 'video': response['path'],
                'assignmentId': response['assignmentId']}))
              start_video_processor(video_file_path, response)
            else:
              raise ConnectionError('unexpected control node action')
          while return_code_queue_map:
            await close_completed_video_processors(conn)
            if return_code_queue_map:
              await asyncio.sleep(sleep_duration)
          logging.info(IO.get_processing_duration(
            time() - start, 'snva {} processed a total of {} videos and {} frames in:'.format(
              snva_version_string, total_num_processed_videos, total_num_processed_frames)))
          logging.info('Video analysis alone spanned a cumulative {:.02f} '
                       'seconds'.format(total_analysis_duration))
          breakLoop = True
      except socket.gaierror:
        logging.info('control node name resolution failed')
        await asyncio.sleep(sleep_duration)
      except ConnectionRefusedError:
        logging.info('connection refused')
        break
      except (ws.exceptions.ConnectionClosed, asyncio.TimeoutError):
        logging.info('Connection or completion receipt lost. Attempting reconnect...')
  finally:
    # Child log producers and forwarders stop before the outer writer sentinel.
    for assignment_id in list(assignment_map):
      close_worker(assignment_id, abort=True)

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
  parser.add_argument('--controlnodehost', '-cnh', default='localhost:8081',
                       help='control node host:port or wss://host:port; TLS is required')
  parser.add_argument('--tls-ca', required=True,
                      help='PEM CA bundle trusted for control and inference servers')
  parser.add_argument('--tls-cert', required=True,
                      help='PEM client certificate chain for mutual TLS')
  parser.add_argument('--tls-key', required=True,
                      help='PEM private key for the client certificate')
  parser.add_argument('--numprocesses', '-np', type=int, default=3, 
                      help='Number of videos to process at one time')
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
  parser.add_argument('--writebbox', '-bb', action='store_true',
                      help='Create JSON files with bounding box data for signal state')
  # parser.add_argument('--excludepreviouslyprocessed', '-epp',
  #                     action='store_true',
  #                     help='Skip processing of videos for which reports '
  #                          'already exist in outputpath.')
  parser.add_argument('--extracttimestamps', '-et', action='store_true',
                      help='Crop timestamps out of video frames and map them to'
                           ' strings for inclusion in the output CSV.')
  parser.add_argument('--gpumemoryfraction', '-gmf', type=float, default=0.9,
                      help='% of GPU memory available to this process.')
  parser.add_argument('--inputpath', '-ip', required=True,
                      help='Path to a single video file, a folder containing '
                           'video files, or a text file that lists absolute '
                           'video file paths.')
  parser.add_argument('--loglevel', '-ll', default='info',
                      help='Defaults to \'info\'. Pass \'debug\' or \'error\' '
                           'for verbose or minimal logging, respectively.')
  parser.add_argument('--logmode', '-lm', default='verbose',
                      help='If verbose, log to file and console. If silent, '
                           'log to file only.')
  parser.add_argument('--logpath', '-l', default='logs',
                      help='Path to the directory where log files are stored.')
  parser.add_argument('--logmaxbytes', '-lmb', type=int, default=2**23,
                      help='File size in bytes at which the log rolls over.')
  parser.add_argument('--maxanalyzerthreads', '-mat', type=int,
                      default=4,
                      help='Maximum number of threads to assign to each video '
                           'processor')
  parser.add_argument('--modelsdirpath', '-mdp',
                      default='models/work_zone_scene_detection',
                      help='Path to the parent directory of model directories.')
  parser.add_argument('--modelname', '-mn', default='mobilenet_v2',
                      help='The name of the model directory under modelsdirpath to use.')
  parser.add_argument('--modelsignaturename', '-msn', default='serving_default',
                      help='Name of the signature that specifies what model is '
                           'being served, and that model\'s input and output '
                           'tensors')
  parser.add_argument('--modelserverhost', '-msh', default='localhost:8500',
                      help='tensorflow serving colon-separated host name or IP '
                           'and port')
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
                      help='Apply class-wise smoothing across video frame class'
                           ' probability distributions.')
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
  parser.add_argument('--writeeventreports', '-wer', type=bool, default=True,
                      help='Output a CVS file for each video containing one or '
                           'more feature events')
  parser.add_argument('--writeinferencereports', '-wir', type=bool,
                      default=False,
                      help='For every video, output a CSV file containing a '
                           'probability distribution over class labels, a '
                           'timestamp, and a frame number for each frame')
  parser.add_argument('--clocktype', '-ct', default='wall',
                      help='Specify whether profiling should use "gpu" or "wall" clock type')
  parser.add_argument('--profformat', '-pfmt', default='pstat',
                      help='Specify whether profiling should save output in "pstat" or "callgrind" formats')
  parser.add_argument('--processormode', '-pm', default='workzone',
                      help='Specify wheter processor should use "workzone", "weather", or "signalstate" pipelines')


  args = parser.parse_args()
  # Validate transport credentials before starting logging or media workers.
  create_client_context(args.tls_ca, args.tls_cert, args.tls_key)
  build_control_url(args.controlnodehost)

  try:
    snva_home = os.environ['SNVA_HOME']
  except KeyError:
    snva_home = '.'

  snva_version_string = 'v0.1.2'

  os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

  

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

  log_format = '%(asctime)s:%(processName)s:%(process)d:%(levelname)s:' \
               '%(module)s:%(lineno)d:%(funcName)s:%(message)s'

  configure_logging(log_file_path, log_format, log_level,
                    args.logmode, args.logmaxbytes)

  log_queue = get_context('spawn').Queue()

  logger_thread = Thread(target=main_logger_fn, args=(log_queue,))

  logger_thread.start()

  logging.debug('SNVA_HOME set to {}'.format(snva_home))

  main_interrupt_queue = get_context('spawn').Queue()

  try:
    asyncio.get_event_loop().run_until_complete(main())
  except Exception as e:
    logging.error(e)
    raise
  finally:
    logging.debug('signaling logger thread to end service.')
    log_queue.put(None)
    logger_thread.join()
    log_queue.close()
    log_queue.join_thread()
    logging.shutdown()
