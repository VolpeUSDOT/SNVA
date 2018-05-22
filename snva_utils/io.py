import csv
import json
import logging
import numpy as np
import os
import subprocess
import tensorflow as tf
import uuid

path = os.path
PIPE = subprocess.PIPE


class IO:
  @staticmethod
  def get_device_ids():
    # TODO: Consider replacing a subprocess invocation with nvml bindings
    command = ['nvidia-smi', '-L']
    pipe = subprocess.run(command, stdout=subprocess.PIPE, encoding='utf-8')
    line_list = pipe.stdout.rstrip().split('\n')
    device_labels = [line.split(':')[0] for line in line_list]
    return [device_label.split(' ')[1] for device_label in device_labels]

  @staticmethod
  def read_class_names(class_names_path):
    meta_map = IO._read_meta_file(class_names_path)
    return {int(key): value for key, value in meta_map.items()}

  @staticmethod
  def load_model(model_path, device_type, gpu_memory_fraction):
    logging.debug('Process {} is loading model at path: {}'.format(
      os.getpid(), model_path))

    graph_def = tf.GraphDef()

    with tf.gfile.FastGFile(model_path, 'rb') as file:
      graph_def.ParseFromString(file.read())

    session_name = str(uuid.uuid4())
    session_graph = tf.import_graph_def(graph_def, name=session_name)

    if device_type == 'gpu':
      gpu_options = tf.GPUOptions(allow_growth=True,
                                  per_process_gpu_memory_fraction=gpu_memory_fraction)
      session_config = tf.ConfigProto(allow_soft_placement=True,
                                      # log_device_placement=True,
                                      gpu_options=gpu_options)
    else:
      session_config = None
    return {'session_name': session_name,
            'session_graph': session_graph,
            'session_config': session_config}

  @staticmethod
  def read_tensor_names(io_tensor_names_path):
    meta_map = IO._read_meta_file(io_tensor_names_path)
    return {key: value + ':0' for key, value in meta_map.items()}

  @staticmethod
  def read_video_file_names(video_file_dir_path):
    included_extenstions = ['avi', 'mp4', 'asf', 'mkv', 'm4v', 'mpeg', 'mov']
    return sorted([fn for fn in os.listdir(video_file_dir_path)
                   if any(fn.lower().endswith(ext) for ext in included_extenstions)])

  @staticmethod
  def print_processing_duration(end_time, msg, loglevel):
    minutes, seconds = divmod(end_time, 60)
    hours, minutes = divmod(minutes, 60)
    hours = int(round(hours))
    minutes = int(round(minutes))
    milliseconds = int(round(end_time * 1000))
    logging.info('{} {:02d}:{:02d}:{:05.2f} ({:d} ms)\n'.format(
      msg, hours, minutes, seconds, milliseconds))
    if loglevel == logging.INFO or loglevel == logging.DEBUG:
      print('{} {:02d}:{:02d}:{:05.2f} ({:d} ms)\n'.format(
        msg, hours, minutes, seconds, milliseconds))

  @staticmethod
  def print_subprocess_command(arg_list, process_id=os.getpid()):
    command_string = arg_list[0]

    for elem in arg_list[1:]:
      command_string += ' ' + elem

    logging.debug('Process {} invoked command: {}'.format(process_id, command_string))

  @staticmethod
  def _read_meta_file(file_path):
    meta_lines = [line.rstrip().split(':') for line in tf.gfile.GFile(file_path).readlines()]
    return {line[0]: line[1] for line in meta_lines}

  @staticmethod
  def read_video_metadata(video_file_path, ffprobe_path):
    command = [ffprobe_path, '-show_streams', '-print_format',
               'json', '-loglevel', 'warning', video_file_path]

    process_id = os.getpid()

    IO.print_subprocess_command(command, process_id)

    completed_subprocess = subprocess.run(command, stdout=PIPE, stderr=PIPE, timeout=600)

    if len(completed_subprocess.stderr) > 0:
      raise Exception(str(completed_subprocess.stderr))

    logging.info('Process {} received raw processed response: {}'.format(
      process_id, completed_subprocess.stdout))

    json_map = json.loads(completed_subprocess.stdout)

    logging.info('Process {} received ffprobe processed response: {}'.format(
      process_id, json.dumps(json_map)))

    return {'width': int(json_map['streams'][0]['width']),
            'height': int(json_map['streams'][0]['height']),
            'frame_count': int(json_map['streams'][0]['nb_frames'])}

  @staticmethod
  def _div_odd(n):
    return n // 2, n // 2 + 1

  @staticmethod
  def _get_gauss_weight_and_window(smoothing_factor):
    window = smoothing_factor * 2 - 1
    weight = np.ndarray((window,))

    for i in range(window):
      frac = (i - smoothing_factor + 1) / window
      weight[i] = 1 / np.exp((4 * frac) ** 2)

    return weight, window

  @staticmethod
  def _smooth_class_prob_sequence(probs, weight, weight_sum, indices, head_padding_len, tail_padding_len):
    smoothed_probs = weight * probs[indices]
    smoothed_probs = np.sum(smoothed_probs, axis=1)
    smoothed_probs = smoothed_probs / weight_sum

    head_padding = np.ones((head_padding_len, )) * smoothed_probs[0]
    tail_padding = np.ones((tail_padding_len, )) * smoothed_probs[-1]

    smoothed_probs = np.concatenate((head_padding, smoothed_probs, tail_padding))

    return smoothed_probs

  @staticmethod
  def _smooth_probs(class_probs, smoothing_factor):
    weight, window = IO._get_gauss_weight_and_window(smoothing_factor)
    weight_sum = np.sum(weight)

    indices = np.arange(class_probs.shape[0] - window)
    indices = np.expand_dims(indices, axis=1) + np.arange(weight.shape[0])

    head_padding_len, tail_padding_len = IO._div_odd(window)

    smoothed_probs = np.ndarray(class_probs.shape)

    for i in range(class_probs.shape[1]):
      smoothed_probs[:, i] = IO._smooth_class_prob_sequence(
        class_probs[:, i], weight, weight_sum, indices, head_padding_len, tail_padding_len)

    return smoothed_probs

  @staticmethod
  def _expand_class_names(class_names, appendage):
    return class_names + [class_name + appendage for class_name in class_names]


  @staticmethod
  def _binarize_probs(class_probs):
    # because numpy will round 0.5 down to 0.0, we need to identify occurrences of 0.5
    # and replace them with 1.0. If a prob has two 0.5s, replace them both with 1.0
    binarized_probs = class_probs.copy()
    uncertain_probs = binarized_probs == 0.5
    binarized_probs[uncertain_probs] = 1.0
    binarized_probs = np.round(binarized_probs)
    return binarized_probs


# TODO: confirm that the csv can be opened after writing
  @staticmethod
  def write_report(video_file_name, report_path, stimestamps, class_probs, class_names,
                   smooth_probs, smoothing_factor, binarize_probs, process_id):
    class_names = ['{}_probability'.format(class_name) for class_name in class_names]

    if smooth_probs and smoothing_factor > 1:
      class_names = IO._expand_class_names(class_names, '_smoothed')
      smoothed_probs = IO._smooth_probs(class_probs, smoothing_factor)
      class_probs = np.concatenate((class_probs, smoothed_probs), axis=1)

    if binarize_probs:
      class_names = IO._expand_class_names(class_names, '_binarized')
      binarized_probs = IO._binarize_probs(class_probs)
      class_probs = np.concatenate((class_probs, binarized_probs), axis=1)

    header = ['file_name', 'frame_number', 'frame_timestamp'] + class_names

    rows = [[video_file_name, '{:07d}'.format(i+1), stimestamps[i]]
            + ['{0:.4f}'.format(cls) for cls in class_probs[i]]
            for i in range(len(class_probs))]

    if not path.exists(report_path):
      os.makedirs(report_path)

    report_file_path = path.join(report_path, video_file_name + '_results.csv')

    with open(report_file_path, 'w', newline='') as report_file:
      csv_writer = csv.writer(report_file)
      csv_writer.writerow(header)
      csv_writer.writerows(rows)



