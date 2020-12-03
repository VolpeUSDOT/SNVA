import argparse
import asyncio
import json
import logging
from logging.handlers import QueueHandler, SocketHandler
from multiprocessing import Process, Queue
import os
import platform
from queue import Empty
import signal
import socket
from subprocess import PIPE, Popen
from threading import Thread
from time import sleep, time
from utils.io import IO
from utils.processor import process_video, process_video_signalstate
import websockets as ws

path = os.path

logger = logging.getLogger('websockets')
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

def main_logger_fn(log_queue):
  while True:
    try:
      message = log_queue.get()
      if message is None:
        break
      logger = logging.getLogger(__name__)
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

  total_num_processed_videos = 0
  total_num_processed_frames = 0
  total_analysis_duration = 0

  def start_video_processor(video_file_path):
    # Before popping the next video off of the list and creating a process to
    # scan it, check to see if fewer than logical_device_count + 1 processes are
    # active. If not, Wait for a child process to release its semaphore
    # acquisition. If so, acquire the semaphore, pop the next video name,
    # create the next child process, and pass the semaphore to it
    return_code_queue = Queue()

    return_code_queue_map[video_file_path] = return_code_queue

    logging.debug('creating new child process.')

    child_log_queue = Queue()

    child_logger_thread = Thread(target=child_logger_fn,
                                 args=(log_queue, child_log_queue))

    child_logger_thread.start()

    child_logger_thread_map[video_file_path] = child_logger_thread

    if 'signalstate' == args.processormode:
      child_process = Process(
        target=process_video_signalstate,
        name=path.splitext(path.split(video_file_path)[1])[0],
        args=(video_file_path, output_dir_path, class_name_map, args.modelname, args.modelsignaturename, args.modelserverhost,model_input_size,
              return_code_queue, child_log_queue, log_level,
              ffmpeg_path, ffprobe_path, args.crop, args.cropwidth, args.cropheight,
              args.cropx, args.cropy, args.extracttimestamps,
              args.timestampmaxwidth, args.timestampheight, args.timestampx,
              args.timestampy, args.deinterlace, args.numchannels, args.batchsize,
              args.smoothprobs, args.smoothingfactor, args.binarizeprobs,
              args.writebbox, args.writeeventreports, args.maxanalyzerthreads, args.processormode))
    else:
      child_process = Process(
      target=process_video,
      name=path.splitext(path.split(video_file_path)[1])[0],
      args=(video_file_path, output_dir_path, class_name_map, args.modelname, args.modelsignaturename, args.modelserverhost,model_input_size,
            return_code_queue, child_log_queue, log_level,
            ffmpeg_path, ffprobe_path, args.crop, args.cropwidth, args.cropheight,
            args.cropx, args.cropy, args.extracttimestamps,
            args.timestampmaxwidth, args.timestampheight, args.timestampx,
            args.timestampy, args.deinterlace, args.numchannels, args.batchsize,
            args.smoothprobs, args.smoothingfactor, args.binarizeprobs,
            args.writeinferencereports, args.writeeventreports, args.maxanalyzerthreads, args.processormode))
    logging.debug('starting child process.')

    child_process.start()

    child_process_map[video_file_path] = child_process

  async def close_completed_video_processors(
      total_num_processed_videos, total_num_processed_frames,
      total_analysis_duration, websocket_conn):
    for video_file_path in list(return_code_queue_map.keys()):
      return_code_queue = return_code_queue_map[video_file_path]

      try:
        return_code_map = return_code_queue.get_nowait()

        return_code = return_code_map['return_code']
        return_value = return_code_map['return_value']

        child_process = child_process_map[video_file_path]

        logging.debug(
          'child process {} returned with exit code {} and exit value '
          '{}'.format(child_process.pid, return_code, return_value))

        if return_code == 'success':
          total_num_processed_videos += 1
          total_num_processed_frames += return_value
          total_analysis_duration += return_code_map['analysis_duration']

          logging.info('notifying control node of completion')

          complete_request = json.dumps({
            'action': 'COMPLETE',
            'video': os.path.basename(video_file_path),
            'output': return_code_map['output_locations']})
          await websocket_conn.send(complete_request)

        child_logger_thread = child_logger_thread_map[video_file_path]
        
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
          os.kill(child_process.pid, signal.SIGKILL)
          logging.warning(
            'child process {} remained alive following join timeout and had to '
            'be killed'.format(child_process.pid))
        except:
          pass
        
        return_code_queue.close()
        
        return_code_queue_map.pop(video_file_path)
        child_logger_thread_map.pop(video_file_path)
        child_process_map.pop(video_file_path)
      except Empty:
        pass

    return total_num_processed_videos, total_num_processed_frames, \
           total_analysis_duration

  start = time()

  sleep_duration = 1
  breakLoop = False
  connectionId = None
  isIdle = False
  while True:
    try:
      if breakLoop:
        break
      wsUrl = 'ws://' + args.controlnodehost + '/registerProcess'
      if connectionId is not None:
        wsUrl = wsUrl + '?id=' + connectionId
      logging.debug("Connecting with URL {}".format(wsUrl))
      async with ws.connect(wsUrl) as conn:
        response = await conn.recv()
        response = json.loads(response)
        logging.info(response)

        if response['action'] != 'CONNECTION_SUCCESS':
          raise ConnectionError(
            'control node connection failed with response: {}'.format(response))
        if connectionId is None:
          connectionId = response['id']
        logging.debug("Assigned id {}".format(connectionId))
        while True:
          # block if num_processes child processes are active
          while len(return_code_queue_map) >= num_processes:
            total_num_processed_videos, total_num_processed_frames, \
            total_analysis_duration = await close_completed_video_processors(
              total_num_processed_videos, total_num_processed_frames,
              total_analysis_duration, conn)
            sleep(sleep_duration)

          try:  # todo poll for termination signal from control node
            _ = main_interrupt_queue.get_nowait()
            logging.debug(
              'breaking out of child process generation following interrupt signal')
            break
          except:
            pass
          
          if not isIdle:
            logging.info('requesting video')
            request = json.dumps({'action': 'REQUEST_VIDEO'})
            await conn.send(request)
            logging.info('reading response')
            response = await conn.recv()
          else:
            # If idle, we will try to close completed processors until all are done
            while len(return_code_queue_map) > 0:
              # Before checking for completed processes, check for a new message
              logging.info('Checking for new message')
              try:
                # If we get a response quickly, break our waiting loop and process the command
                response = await asyncio.wait_for(conn.recv(), 1)
                break
              except asyncio.TimeoutError:
                # Otherwise, go back to finishing our current tasks
                logging.debug('No new message from control node, continuing...')
                pass
              total_num_processed_videos, total_num_processed_frames, \
              total_analysis_duration = await close_completed_video_processors(
                total_num_processed_videos, total_num_processed_frames,
                total_analysis_duration, conn)
              # by now, the last device_id_queue_len videos are being processed,
              # so we can afford to poll for their completion infrequently
              if len(return_code_queue_map) > 0:
                sleep(sleep_duration)
            # Once all are complete, if still idle we have no work left to do - we just wait for a new message
            response = await conn.recv() 
          
          response = json.loads(response)

          if response['action'] == 'STATUS_REQUEST':
            logging.info('control node requested status request')
            pass
          elif response['action'] == 'CEASE_REQUESTS':
            logging.info('control node has no more videos to process')
            isIdle = True
            pass
          elif response['action'] == 'RESUME_REQUESTS':
            logging.info('control node has instructed to resume requests')
            isIdle = False
            pass
          elif response['action'] == 'SHUTDOWN':
            logging.info('control node requested shutdown')
            breakLoop = True
            break
          elif response['action'] == 'PROCESS':
            # TODO Prepend input path
            video_file_path = os.path.join(args.inputpath, response['path'])
            request_received = json.dumps({'action': 'REQUEST_RECEIVED', 'video': response['path']})
            await conn.send(request_received)
            try:
              start_video_processor(video_file_path)
            except Exception as e:
              logging.error('an unknown error has occured while processing {}'.format(video_file_path))
              logging.error(e)
          else:
            raise ConnectionError(
              'control node replied with unexpected response: {}'.format(response))
        logging.debug('{} child processes remain enqueued'.format(len(return_code_queue_map)))
        while len(return_code_queue_map) > 0:
          #logging.debug('waiting for the final {} child processes to '
          #              'terminate'.format(len(return_code_queue_map)))

          total_num_processed_videos, total_num_processed_frames, \
          total_analysis_duration = await close_completed_video_processors(
            total_num_processed_videos, total_num_processed_frames,
            total_analysis_duration, conn)

          # by now, the last device_id_queue_len videos are being processed,
          # so we can afford to poll for their completion infrequently
          if len(return_code_queue_map) > 0:
            #logging.debug('sleeping for {} seconds'.format(sleep_duration))
            sleep(sleep_duration)

        end = time() - start

        processing_duration = IO.get_processing_duration(
          end, 'snva {} processed a total of {} videos and {} frames in:'.format(
            snva_version_string, total_num_processed_videos,
            total_num_processed_frames))
        logging.info(processing_duration)

        logging.info('Video analysis alone spanned a cumulative {:.02f} '
                    'seconds'.format(total_analysis_duration))

        logging.info('exiting snva {} main process'.format(snva_version_string))
        breakLoop = True
    except socket.gaierror:
      # log something
      logging.info('gaierror')
      continue
    except ConnectionRefusedError:
      # log something else
      logging.info('connection refused')
      break
    except ws.exceptions.ConnectionClosed:
      logging.info('Connection lost.  Attempting reconnect...')
      continue
    except Exception as e:
      logging.error("Unknown Exception")
      logging.error(e)
      raise e
    if breakLoop:
      break

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
  parser.add_argument('--controlnodehost', '-cnh', default='localhost:8080',
                      help='control node colon-separated host name or IP and '
                           'port')
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
  parser.add_argument('--modelserverhost', '-msh', default='0.0.0.0:8500',
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

  logger_script_path = path.join(snva_home, 'utils/logger.py')

  log_file_max_bytes = '{}'.format(args.logmaxbytes)

  stdin = os.dup(0)

  logger_subprocess = Popen(
    ['python', logger_script_path, log_file_path, log_format, args.loglevel,
     args.logmode, log_file_max_bytes, '{}'.format(stdin)], stdout=PIPE)

  # wait for logger.py to indicate readiness
  _ = logger_subprocess.stdout.readline()

  log_handlers = [SocketHandler(
    host='localhost', port=logging.handlers.DEFAULT_TCP_LOGGING_PORT)]

  valid_log_modes = ['verbose', 'silent']

  if args.logmode == 'verbose':
    log_handlers.append(logging.StreamHandler())
  elif not args.logmode == 'silent':
    raise ValueError(
      'The specified logmode is not in the set {}.'.format(valid_log_modes))

  logging.basicConfig(level=log_level, format=log_format, handlers=log_handlers)

  log_queue = Queue()

  logger_thread = Thread(target=main_logger_fn, args=(log_queue,))

  logger_thread.start()

  logging.debug('SNVA_HOME set to {}'.format(snva_home))

  main_interrupt_queue = Queue()

  try:
    asyncio.get_event_loop().run_until_complete(main())
  except Exception as e:
    logging.error(e)

  logging.debug('signaling logger thread to end service.')
  log_queue.put(None)

  logger_thread.join()

  logging.shutdown()

  logger_subprocess.terminate()