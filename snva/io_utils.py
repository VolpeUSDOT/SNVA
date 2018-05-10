import csv
import json
import numpy as np
import os
import subprocess
import tensorflow as tf
import uuid
import logging

path = os.path


class IOObject:
  @staticmethod
  def get_gpu_ids():
    # TODO: Consider replacing a subprocess invocation with nvml bindings
    command = ['nvidia-smi', '-L']

    pipe = subprocess.run(command, stdout=subprocess.PIPE, encoding='utf-8')

    line_list = pipe.stdout.rstrip().split('\n')
    gpu_labels = [line.split(':')[0] for line in line_list]
    return [gpu_label.split(' ')[1] for gpu_label in gpu_labels]

  @staticmethod
  def load_labels(labels_path):
    meta_map = IOObject._read_meta_file(labels_path)
    return {int(key): value for key, value in meta_map.items()}

  @staticmethod
  def load_model(model_path, gpu_memory_fraction):
    logging.debug("Loading model")
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
    meta_map = IOObject._read_meta_file(io_tensor_names_path)
    return {key: value + ':0' for key, value in meta_map.items()}

  @staticmethod
  def load_video_file_names(video_file_dir_path):
    included_extenstions = ['avi', 'mp4', 'asf', 'mkv', 'm4v', 'mpeg', 'mov']
    return sorted([fn for fn in os.listdir(video_file_dir_path)
                   if any(fn.lower().endswith(ext) for ext in included_extenstions)])

  @staticmethod
  def print_processing_duration(end_time, msg):
    minutes, seconds = divmod(end_time, 60)
    hours, minutes = divmod(minutes, 60)
    print('{:s}: {:02d}:{:02d}:{:02d}\n'.format(
      msg, int(hours), int(minutes), int(seconds)))

  @staticmethod
  def _read_meta_file(file_path):
    meta_lines = [line.rstrip().split(':') for line in tf.gfile.GFile(file_path).readlines()]
    return {line[0]: line[1] for line in meta_lines}

  @staticmethod
  def read_video_metadata(video_file_path):
    command = ['ffprobe', '-show_streams', '-print_format',
               'json', '-loglevel', 'quiet', video_file_path]

    logging.debug("ffprobe command: {}".format(command))
    pipe = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    json_string, err = pipe.communicate()

    json_map = json.loads(json_string)
    logging.debug("ffprobe response {}".format(json.dumps(json_map)))
    if not json_map:
      logging.error("ffprobe returned no response. This is possibly a permissions issue")

    return {'width': int(json_map['streams'][0]['width']),
            'height': int(json_map['streams'][0]['height']),
            'frame_count': int(json_map['streams'][0]['nb_frames'])}

  @staticmethod
  def _div_odd(n):
    return n // 2, n // 2 + 1

  @staticmethod
  def _smooth_class_prob_sequence(probs, smoothing_factor):
    window = smoothing_factor * 2 - 1
    weight = np.array([1.0] * window)
    gauss_weight = []

    for i in range(window):
      i = i - smoothing_factor + 1
      frac = i / float(window)
      gauss = 1 / (np.exp((4 * (frac)) ** 2))
      gauss_weight.append(gauss)

    weight = np.array(gauss_weight) * weight

    smoothed_probs = [sum(np.array(probs[i:i + window]) * weight) / sum(weight)
                      for i in range(len(probs) - window)]

    padfront, padback = IOObject._div_odd(window)
    for i in range(0, padfront):
      smoothed_probs.insert(0, smoothed_probs[0])
    for i in range(0, padback):
      smoothed_probs.append(smoothed_probs[-1])

    return smoothed_probs

  @staticmethod
  def _smooth_probs(class_probs, smoothing_factor):
    smoothed_probs = [IOObject._smooth_class_prob_sequence(class_probs[:, i], int(smoothing_factor))
                      for i in range(len(class_probs[0]))]
    smoothed_probs = np.array(smoothed_probs)
    smoothed_probs = np.transpose(smoothed_probs)

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


  @staticmethod
  def write_report(video_file_name, report_path, time_stamps, class_probs,
                   class_names, smoothing_factor=0, binarize=False):
    num_frames = len(class_probs)

    if smoothing_factor > 1:
      class_names = IOObject._expand_class_names(class_names, '_smoothed')
      smoothed_probs = IOObject._smooth_probs(class_probs, smoothing_factor)
      class_probs = np.concatenate((class_probs, smoothed_probs), axis=1)

    if binarize:
      class_names = IOObject._expand_class_names(class_names, '_binarized')
      binarized_probs = IOObject._binarize_probs(class_probs)
      class_probs = np.concatenate((class_probs, binarized_probs), axis=1)

    class_names = ['{}_probability'.format(class_name) for class_name in class_names]

    header = ['video_file_name', 'video_frame_number', 'time_stamp'] + class_names

    rows = [['{:s}'.format(video_file_name), '{:07d}'.format(i+1), time_stamps[i]]
            + ['{0:.4f}'.format(cls) for cls in class_probs[i]]
            for i in range(num_frames)]

    if not path.exists(report_path):
      os.makedirs(report_path)

    report_file_path = path.join(report_path, video_file_name + '_results.csv')

    with open(report_file_path, 'w', newline='') as report_file:
      csv_writer = csv.writer(report_file)
      csv_writer.writerow(header)
      csv_writer.writerows(rows)


