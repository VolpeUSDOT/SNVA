import csv
import numpy as np
import os
from utils.io import IO

path = os.path


class Feature:
  def __init__(self, feature_id, class_id, class_name, start_timestamp,
               end_timestamp, start_frame_number, end_frame_number):
    # the position of the feature in the source video relative to other features
    self.feature_id = feature_id
    # id of the class predicted by the video analyzer
    self.class_id = class_id
    # name of the class predicted by the video analyzer
    self.class_name = class_name
    # the timestamp extracted from the first frame in which the feature occurred
    self.start_timestamp = start_timestamp
    # the timestamp extracted from the last frame in which the feature occurred
    self.end_timestamp = end_timestamp
    # the number of milliseconds over which the feature occurs
    self.duration = self.end_timestamp - self.start_timestamp
    # the number of the first frame in which the feature occurred
    self.start_frame_number = start_frame_number
    # the number of the last frame in which the feature occurred
    self.end_frame_number = end_frame_number
    # the number of consecutive frames over which the feature occurs
    self.length = self.end_frame_number - self.start_frame_number

  def __str__(self):
    print_string = '\tfeature_id: ' + str(self.feature_id) + '\n'
    print_string += '\tclass_id: ' + str(self.class_id) + '\n'
    print_string += '\tclass_name: ' + str(self.class_name) + '\n'
    print_string += '\tstart_timestamp: ' + str(self.start_timestamp) + '\n'
    print_string += '\tend_timestamp: ' + str(self.end_timestamp) + '\n'
    print_string += '\tduration: ' + str(self.duration) + '\n'
    print_string += '\tstart_frame_number: ' + str(self.start_frame_number) + \
                    '\n'
    print_string += '\tend_frame_number: ' + str(self.end_frame_number) + '\n'
    print_string += '\tlength: ' + str(self.length)

    return print_string


class Event:
  def __init__(self, event_id, trip_id, target_feature,
               preceding_feature=None, following_feature=None):
    self.event_id = event_id

    self.trip_id = trip_id

    self.target_feature = target_feature

    self.preceding_feature = preceding_feature

    self.following_feature = following_feature

    if self.preceding_feature:
      self.start_timestamp = self.preceding_feature.start_timestamp
    else:
      self.start_timestamp = self.target_feature.start_timestamp

    if self.following_feature:
      self.end_timestamp = self.following_feature.end_timestamp
    else:
      self.end_timestamp = self.target_feature.end_timestamp

  def __str__(self):
    print_string = 'SHRP2 NDS Video Event\n\n'

    if self.preceding_feature:
      print_string += 'Preceding Feature:\n{}\n\n'.format(
        self.preceding_feature)

    print_string += 'Target Feature:\n{}\n\n'.format(self.target_feature)

    if self.following_feature:
      print_string += 'Following Feature:\n{}\n\n'.format(
        self.following_feature)

    return print_string


class Trip:
  def __init__(self, trip_id, report_file_path,
               class_names_file_path, smooth_probs=False):
    self.trip_id = trip_id

    self.class_names = IO.read_class_names(class_names_file_path)
    self.class_ids = {value: key for key, value in self.class_names.items()}

    with open(report_file_path, newline='') as report_file:
      report_reader = csv.reader(report_file)

      report_header = next(report_reader)
      report_rows = np.array(list(report_reader))

      class_header_names = [class_name + '_probability'
                            for class_name in self.class_names.values()]

      class_prob_indices = [report_header.index(class_header_name)
                                   for class_header_name in class_header_names]
      class_prob_indices.sort()  # in case no order can safely be assumed

      report_probs = report_rows[:,
                     class_prob_indices[0]:class_prob_indices[-1] + 1]
      report_probs = report_probs.astype(np.float32)
      if smooth_probs:
        report_probs = IO._smooth_probs(report_probs, 16)
      report_class_ids = np.argmax(report_probs, axis=1)

      frame_number_index = report_header.index('frame_number')
      report_frame_numbers = report_rows[:, frame_number_index]
      report_frame_numbers = report_frame_numbers.astype(np.int32)

      frame_timestamp_index = report_header.index('frame_timestamp')
      report_timestamps = report_rows[:, frame_timestamp_index]
      report_timestamps = report_timestamps.astype(np.int32)

      self.feature_sequence = []

      feature_id = 0
      class_id = report_class_ids[0]
      start_timestamp = report_timestamps[0]
      start_frame_number = report_frame_numbers[0]

      for i in range(len(report_class_ids)):
        if report_class_ids[i] != class_id:
          end_timestamp = report_timestamps[i - 1]
          end_frame_number = report_frame_numbers[i - 1]

          # the beginning of the next feature has been reached.
          # create an object for the preceding feature.
          self.feature_sequence.append(Feature(
            feature_id=feature_id, class_id=class_id,
            class_name=self.class_names[class_id], start_timestamp=start_timestamp,
            end_timestamp=end_timestamp, start_frame_number=start_frame_number,
            end_frame_number=end_frame_number))

          feature_id += 1
          class_id = report_class_ids[i]
          start_timestamp = report_timestamps[i]
          start_frame_number = report_frame_numbers[i]

  def events(
      self, target_feature_class_id, preceding_feature_class_id,
      following_feature_class_id, target_feature_class_name=None,
      preceding_feature_class_name=None, following_feature_class_name=None):
    if target_feature_class_id is None:
      if target_feature_class_name is None:
        raise ValueError('target_feature_class_id and target_feature_class_name'
                         ' cannot both be None')
      else:
        target_feature_class_id = self.class_ids[target_feature_class_name]

    if preceding_feature_class_id is None \
        and preceding_feature_class_name is not None:
        preceding_feature_class_id = self.class_ids[
          preceding_feature_class_name]

    if target_feature_class_id == preceding_feature_class_id:
      raise ValueError('target_feature_class_id and preceding_feature_class_id'
                       ' cannot be equal')

    if following_feature_class_id is None \
        and following_feature_class_name is not None:
        following_feature_class_id = self.class_ids[
          following_feature_class_name]

    if target_feature_class_id == following_feature_class_id:
      raise ValueError('target_feature_class_id and following_feature_class_id'
                       ' cannot be equal')

    events = []

    previous_event = None

    event_id = 0

    if preceding_feature_class_id and following_feature_class_id:
      previous_preceding_feature = None

      for current_feature in self.feature_sequence:
        if current_feature.class_id == preceding_feature_class_id:
          previous_preceding_feature = current_feature
  
        if current_feature.class_id == following_feature_class_id:
          previous_following_feature = current_feature

          if previous_event \
              and previous_event.following_feature is None:
            previous_event.following_feature = previous_following_feature
  
        if current_feature.class_id == target_feature_class_id:
          current_event = Event(event_id=event_id, trip_id=self.trip_id,
                                target_feature=current_feature)

          # if two consecutive events share a common following/preceding
          # feature, and that feature is closer to the current event than the
          # previous event, reassign it to the current event.
          if previous_event and previous_preceding_feature \
              and previous_event.following_feature \
                  == previous_preceding_feature:
            previous_target_feature = previous_event.target_feature

            previous_target_feature_distance = abs(
              previous_target_feature.end_timestamp
              - previous_preceding_feature.start_timestamp)

            current_feature_distance = abs(
              current_feature.start_timestamp -
              previous_preceding_feature.end_timestamp)

            if current_feature_distance < previous_target_feature_distance:
              previous_event.following_feature = None
              current_event.preceding_feature = previous_preceding_feature

          events.append(current_event)

          event_id += 1
  
          previous_event = current_event

    return events

  def work_zone_events(self):
    return self.events(
      target_feature_class_id=self.class_ids['work_zone'],
      target_feature_class_name='work_zone',
      preceding_feature_class_id=self.class_ids['warning_sign'],
      preceding_feature_class_name='warning_sign',
      following_feature_class_id=self.class_ids['warning_sign'],
      following_feature_class_name='warning_sign')
