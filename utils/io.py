import csv
import json
import logging
import numpy as np
import os
import math
import subprocess as sp

path = os.path


class IO:
  @staticmethod
  def _invoke_subprocess(command):
    completed_subprocess = sp.run(
      command, stdout=sp.PIPE, stderr=sp.PIPE, timeout=60)

    if len(completed_subprocess.stderr) > 0:
      std_err = str(completed_subprocess.stderr, encoding='utf-8')

      raise Exception(std_err)

    return str(completed_subprocess.stdout, encoding='utf-8')

  @staticmethod
  def get_device_ids():
    command = ['nvidia-smi', '--query-gpu=index', '--format=csv']
    output = IO._invoke_subprocess(command)
    line_list = output.rstrip().split('\n')
    return line_list[1:]

  @staticmethod
  def read_class_names(class_names_path):
    meta_map = IO._read_meta_file(class_names_path)
    return {int(key): value for key, value in meta_map.items()}

  @staticmethod
  def read_node_names(io_node_names_path):
    meta_map = IO._read_meta_file(io_node_names_path)
    return {key: value + ':0' for key, value in meta_map.items()}

  @staticmethod
  def read_video_file_names(video_file_dir_path):
    included_extenstions = ['avi', 'mp4', 'asf', 'mkv', 'm4v', 'mpeg', 'mov']
    return sorted([fn for fn in os.listdir(video_file_dir_path) if any(
      fn.lower().endswith(ext) for ext in included_extenstions)])

  @staticmethod
  def _div_odd(n):
    return n // 2, n // 2 + 1

  @staticmethod
  def get_processing_duration(end_time, msg):
    minutes, seconds = divmod(end_time, 60)
    hours, minutes = divmod(minutes, 60)
    hours = int(round(hours))
    minutes = int(round(minutes))
    milliseconds = int(round(end_time * 1000))
    return '{} {:02d}:{:02d}:{:05.2f} ({:d} ms)'.format(
      msg, hours, minutes, seconds, milliseconds)

  @staticmethod
  def _read_meta_file(file_path):
    meta_lines = [line.rstrip().split(':')
                  for line in open(file_path).readlines()]
    return {line[0]: line[1] for line in meta_lines}

  @staticmethod
  def get_video_dimensions(video_file_path, ffprobe_path):
    command = [ffprobe_path, '-show_streams', '-print_format',
               'json', '-loglevel', 'warning', video_file_path]
    output = IO._invoke_subprocess(command)
    try:
      json_map = json.loads(output)
    except Exception as e:
      logging.error('encountered an exception while parsing ffprobe JSON file.')
      logging.debug('received raw ffprobe response: {}'.format(output))
      logging.debug('will raise exception to caller.')
      raise e
    return int(json_map['streams'][0]['width']),\
           int(json_map['streams'][0]['height']),\
           int(json_map['streams'][0]['nb_frames']),\
           int(math.ceil(float(json_map['streams'][0]['duration']))) + 1

  @staticmethod
  def _get_gauss_weight_and_window(smoothing_factor):
    window = smoothing_factor * 2 - 1
    weight = np.ndarray((window,))
    for i in range(window):
      frac = (i - smoothing_factor + 1) / window
      weight[i] = 1 / np.exp((4 * frac) ** 2)
    return weight, window

  @staticmethod
  def _smooth_class_prob_sequence(
      probs, weight, weight_sum, indices, head_padding_len, tail_padding_len):
    smoothed_probs = weight * probs[indices]
    smoothed_probs = np.sum(smoothed_probs, axis=1)
    smoothed_probs = smoothed_probs / weight_sum
    head_padding = np.ones((head_padding_len, )) * smoothed_probs[0]
    tail_padding = np.ones((tail_padding_len, )) * smoothed_probs[-1]
    smoothed_probs = np.concatenate(
      (head_padding, smoothed_probs, tail_padding))
    return smoothed_probs

  @staticmethod
  def smooth_probs(class_probs, smoothing_factor):
    weight, window = IO._get_gauss_weight_and_window(smoothing_factor)
    weight_sum = np.sum(weight)
    indices = np.arange(class_probs.shape[0] - window)
    indices = np.expand_dims(indices, axis=1) + np.arange(weight.shape[0])
    head_padding_len, tail_padding_len = IO._div_odd(window)
    smoothed_probs = np.ndarray(class_probs.shape)
    for i in range(class_probs.shape[1]):
      smoothed_probs[:, i] = IO._smooth_class_prob_sequence(
        class_probs[:, i], weight, weight_sum, indices,
        head_padding_len, tail_padding_len)
    return smoothed_probs

  @staticmethod
  def _expand_class_names(class_names, appendage):
    return class_names + [class_name + appendage for class_name in class_names]

  @staticmethod
  def _binarize_probs(class_probs):
    # since numpy rounds 0.5 to 0.0, identify occurrences of 0.5 and replace
    # them with 1.0. If a prob has two 0.5s, replace them both with 1.0
    binarized_probs = class_probs.copy()
    uncertain_prob_indices = binarized_probs == 0.5
    binarized_probs[uncertain_prob_indices] = 1.0
    binarized_probs = np.round(binarized_probs)

    return binarized_probs

  @staticmethod
  def open_report(report_file_path):
    report_file = open(report_file_path, newline='')

    return csv.reader(report_file)

  @staticmethod
  def read_report_header(
      report_reader, frame_col_num=None, timestamp_col_num=None, qa_flag_col_num=None,
      data_col_range=None, header_mask=None, return_data_col_range=False):
    if data_col_range is None and header_mask is None:
      raise ValueError('data_col_range and header_mask cannot both be None.')

    csv_header = next(report_reader)

    report_header = []
    
    if frame_col_num:
      report_header.append(csv_header[frame_col_num])
    
    if timestamp_col_num:
      report_header.append(csv_header[timestamp_col_num])

    if qa_flag_col_num:
      report_header.append(csv_header[qa_flag_col_num])
        
    if len(report_header) == 0:
      raise ValueError(
        'frame_col_num and timestamp_col_num cannot both be None.')

    if data_col_range is None:
      data_col_indices = [csv_header.index(data_col_name)
                          for data_col_name in header_mask[len(report_header):]]
      data_col_range = (data_col_indices[0], data_col_indices[-1] + 1)

    report_header.extend(csv_header[data_col_range[0]: data_col_range[1]])

    if header_mask and report_header != header_mask:
      raise ValueError(
        'report header: {} was expected to match header mask: {}\ngiven '
        'frame_col_num: {}, timestamp_col_num: {} and data_col_range: '
        '{}'.format(report_header, header_mask, frame_col_num, 
                    timestamp_col_num, data_col_range))
    
    if return_data_col_range:
      return report_header, data_col_range
    else:
      return report_header

  @staticmethod
  def read_report_data(
      report_reader, frame_col_num=None, timestamp_col_num=None,
      qa_flag_col_num=None, data_col_range=None):
    if frame_col_num and timestamp_col_num and data_col_range:
      frame_numbers = []
      timestamps = []
      probabilities = []

      for row in report_reader:
        frame_numbers.append(row[frame_col_num])
        timestamps.append(row[timestamp_col_num])
        probabilities.append(row[data_col_range[0]:data_col_range[1]])

      report_data = {'frame_numbers': np.array(frame_numbers),
                     'frame_timestamps': np.array(timestamps),
                     'probabilities': np.array(probabilities)}
    elif frame_col_num and data_col_range:
      frame_numbers = []
      probabilities = []

      for row in report_reader:
        frame_numbers.append(row[frame_col_num])
        probabilities.append(row[data_col_range[0]:data_col_range[1]])

      report_data = {'frame_numbers': np.array(frame_numbers),
                     'probabilities': np.array(probabilities)}
    elif timestamp_col_num and data_col_range:
      timestamps = []
      probabilities = []

      for row in report_reader:
        timestamps.append(row[timestamp_col_num])
        probabilities.append(row[data_col_range[0]:data_col_range[1]])

      report_data = {'frame_timestamps': np.array(timestamps),
                     'probabilities': np.array(probabilities)}
    elif data_col_range:
      probabilities = []

      for row in report_reader:
        probabilities.append(row[data_col_range[0]:data_col_range[1]])

      report_data = {'probabilities': np.array(probabilities)}
    else:
      report_data = np.array([row for row in report_reader])

    return report_data

  @staticmethod
  def read_report(report_file_path, frame_col_num=None, timestamp_col_num=None,
                  qa_flag_col_num=None, data_col_range=None, header_mask=None,
                  return_data_col_range=False):
    report_reader = IO.open_report(report_file_path)

    if return_data_col_range:
      report_header, data_col_range = IO.read_report_header(
        report_reader, frame_col_num=frame_col_num,
        timestamp_col_num=timestamp_col_num, qa_flag_col_num=qa_flag_col_num,
        data_col_range=data_col_range, header_mask=header_mask,
        return_data_col_range=True)
    else:
      report_header = IO.read_report_header(
        report_reader, frame_col_num=frame_col_num,
        timestamp_col_num=timestamp_col_num, qa_flag_col_num=qa_flag_col_num,
        data_col_range=data_col_range, header_mask=header_mask,
        return_data_col_range=False)

    report_data = IO.read_report_data(
      report_reader, frame_col_num=frame_col_num,
      timestamp_col_num=timestamp_col_num, qa_flag_col_num=qa_flag_col_num,
      data_col_range=data_col_range)

    if return_data_col_range:
      return report_header, report_data, data_col_range
    else:
      return report_header, report_data

  @staticmethod
  def write_csv(file_path, header, rows):
    with open(file_path, 'w', newline='') as file:
      csv_writer = csv.writer(file)
      csv_writer.writerow(header)
      csv_writer.writerows(rows)

  # TODO: confirm that the csv can be opened after writing
  @staticmethod
  def write_inference_report(
      report_file_name, report_dir_path, class_probs, class_name_map,
      timestamp_strings=None, qa_flags=None, smooth_probs=False,
      smoothing_factor=0, binarize_probs=False):
    class_names = ['{}_probability'.format(class_name)
                   for class_name in class_name_map.values()]

    if smooth_probs and smoothing_factor > 1:
      class_names = IO._expand_class_names(class_names, '_smoothed')
      smoothed_probs = IO.smooth_probs(class_probs, smoothing_factor)
      class_probs = np.concatenate((class_probs, smoothed_probs), axis=1)

    if binarize_probs:
      class_names = IO._expand_class_names(class_names, '_binarized')
      binarized_probs = IO._binarize_probs(class_probs)
      class_probs = np.concatenate((class_probs, binarized_probs), axis=1)

    if timestamp_strings is not None:
      header = ['file_name', 'frame_number', 'frame_timestamp', 'qa_flag'] + \
               class_names
      rows = [[report_file_name, '{:d}'.format(i + 1), timestamp_strings[i],
               qa_flags[i]] + ['{0:.4f}'.format(cls) for cls in class_probs[i]]
              for i in range(len(class_probs))]
    else:
      header = ['file_name', 'frame_number'] + class_names
      rows = [[report_file_name, '{:d}'.format(i + 1)]
              + ['{0:.4f}'.format(cls) for cls in class_probs[i]]
              for i in range(len(class_probs))]

    report_dir_path = path.join(report_dir_path, 'inference_reports')

    if not path.exists(report_dir_path):
      os.makedirs(report_dir_path)

    report_file_path = path.join(
      report_dir_path, report_file_name + '.csv')

    IO.write_csv(report_file_path, header, rows)
    return report_file_path

  # TODO: confirm that the csv can be opened after writing
  @staticmethod
  def write_event_report(report_file_name, report_dir_path, events):
    report_dir_path = path.join(report_dir_path, 'event_reports')

    if not path.exists(report_dir_path):
      os.makedirs(report_dir_path)

    report_file_path = path.join(
      report_dir_path, report_file_name + '.csv')

    header = ['file_name', 'sequence_number', 'start_frame_number',
              'end_frame_number', 'start_timestamp', 'end_timestamp']

    rows = [[report_file_name, event.event_id + 1, event.start_frame_number,
             event.end_frame_number, event.start_timestamp, event.end_timestamp]
            for event in events]

    IO.write_csv(report_file_path, header, rows)
    return report_dir_path
  
  @staticmethod
  def write_json(file_name, dir_path, json_data):
    file_path = path.join(dir_path, 'bbox_reports')
    if not path.exists(file_path):
      os.makedirs(file_path)
    file_path = path.join(file_path, file_name + '.json')
    with open(file_path, mode='w', newline='') as output_file:
        json.dump(json_data, output_file)
    return file_path

  @staticmethod
  def write_weather_report(report_file_name, report_dir_path, weather_features):
    report_dir_path = path.join(report_dir_path, 'event_reports')

    if not path.exists(report_dir_path):
      os.makedirs(report_dir_path)

    report_file_path = path.join(
      report_dir_path, report_file_name + '.csv')

    header = ['file_name', 'sequence_number', 'classification', 'start_frame_number',
              'end_frame_number', 'start_timestamp', 'end_timestamp']
    rows = [[report_file_name, feat.feature_id, feat.class_name, feat.start_frame_number,
             feat.end_frame_number, feat.start_timestamp, feat.end_timestamp]
            for feat in weather_features]
    IO.write_csv(report_file_path, header, rows)
    return report_file_path

  @staticmethod
  def write_signalstate_report(report_file_name, report_dir_path, detections):
    report_dir_path = path.join(report_dir_path, 'event_reports')

    if not path.exists(report_dir_path):
      os.makedirs(report_dir_path)

    report_file_path = path.join(
      report_dir_path, report_file_name + '.csv')

    header = ['file_name', 'frame_number', 'timestamp',
              'classification']

    rows = [[report_file_name, det['frame_num'], det['timestamp'],
             det['classification']]
            for det in detections]

    IO.write_csv(report_file_path, header, rows)
    return report_file_path
