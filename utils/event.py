import logging
import numpy as np
import os
from utils.io import IO

path = os.path


class Feature:
  def __init__(self, feature_id, class_id, class_name, start_timestamp,
               end_timestamp, start_timestamp_qa_flag, end_timestamp_qa_flag,
               start_frame_number, end_frame_number, event_id=None):
    """Create a new 'Feature' object.
    
    Args:
      feature_id: The position of the feature in the source video relative to
        other features
      event_id: The id of the event to which this feature was assigned
      class_id: The id of the class predicted by the video analyzer
      class_name: The name of the class predicted by the video analyzer
      start_timestamp: The timestamp extracted from the first frame in which the
        feature occurred
      end_timestamp: The timestamp extracted from the last frame in which the
        feature occurred
      start_frame_number: The number of the first frame in which the feature
        occurred
      end_frame_number: The number of the last frame in which the feature
        occurred
    """
    self.feature_id = feature_id
    self.class_id = class_id
    self.class_name = class_name
    self.start_timestamp = start_timestamp
    self.end_timestamp = end_timestamp
    self.start_timestamp_qa_flag = start_timestamp_qa_flag
    self.end_timestamp_qa_flag = end_timestamp_qa_flag
    self.start_frame_number = start_frame_number
    self.end_frame_number = end_frame_number
    self.event_id = event_id

    # the number of consecutive frames over which the feature occurs
    # if self.end_frame_number and self.start_frame_number:
    self.length = self.end_frame_number - self.start_frame_number
    # else:
    #   self.length = None

  def __str__(self):
    print_string = '\tfeature_id: ' + str(self.feature_id) + '\n'
    print_string += '\tevent_id: ' + str(self.event_id) + '\n'
    print_string += '\tclass_id: ' + str(self.class_id) + '\n'
    print_string += '\tclass_name: ' + str(self.class_name) + '\n'
    print_string += '\tstart_timestamp: ' + str(self.start_timestamp) + '\n'
    print_string += '\tend_timestamp: ' + str(self.end_timestamp) + '\n'
    print_string += '\tstart_frame_number: ' + str(self.start_frame_number) + \
                    '\n'
    print_string += '\tend_frame_number: ' + str(self.end_frame_number) + '\n'
    print_string += '\tlength: ' + str(self.length)

    return print_string


class Event:
  def __init__(self, event_id, target_feature_list, preceding_feature=None,
               following_feature=None):
    """Create a new 'Event' object.
    
    Args:
      event_id: int. The position of the event in the source video relative to
        other events.
      target_feature: Feature. The feature constituting the event of interest.
      preceding_feature: Feature. An auxiliary feature strictly different in
        type from the target feature that should be included in the event if it
        occurs just before the target feature in the source video.
      following_feature: Feature. An auxiliary feature strictly different in
        type from the target feature that should be included in the event if it
        occurs just after the target feature in the source video.
    """
    self.event_id = event_id

    self.target_feature_list = target_feature_list

    for target_feature in self.target_feature_list:
      target_feature.event_id = self.event_id

    self.start_timestamp = self.target_feature_list[0].start_timestamp
    self.end_timestamp = self.target_feature_list[-1].end_timestamp

    self.start_frame_number = self.target_feature_list[0].start_frame_number
    self.end_frame_number = self.target_feature_list[-1].end_frame_number

    self.length = self.end_frame_number - self.start_frame_number

    self._preceding_feature = preceding_feature
    self._following_feature = following_feature

  @property
  def preceding_feature(self):
    return self._preceding_feature
  
  @preceding_feature.setter
  def preceding_feature(self, preceding_feature):
    self._preceding_feature = preceding_feature
    self.start_timestamp = self.preceding_feature.start_timestamp
    self.start_frame_number = self.preceding_feature.start_frame_number
  
  @property  
  def following_feature(self):
    return self._following_feature
  
  @following_feature.setter
  def following_feature(self, following_feature):
    self._following_feature = following_feature
    # if this event's following feature is being reassigned to a later event,
    # the 'following_feature' argument will be None
    if self.following_feature:
      self.end_timestamp = self.following_feature.end_timestamp
      self.end_frame_number = self.following_feature.end_frame_number
    else:
      self.end_timestamp = self.target_feature_list[-1].end_timestamp
      self.end_frame_number = self.target_feature_list[-1].end_frame_number

  def __str__(self):
    print_string = 'SHRP2 NDS Video Event\n\n'

    if self.preceding_feature:
      print_string += 'Preceding Feature:\n{}\n\n'.format(
        self.preceding_feature)

    print_string += 'Target Features:\n{}\n\n'.format(self.target_feature_list)

    if self.following_feature:
      print_string += 'Following Feature:\n{}\n\n'.format(
        self.following_feature)

    return print_string


class Trip:
  def __init__(self, report_frame_numbers, report_timestamps, qa_flags,
               report_probs, class_name_map, non_event_weight_scale=0.05):
    self.class_names = class_name_map
    self.class_ids = {value: key for key, value in self.class_names.items()}

    report_class_ids = np.argmax(report_probs, axis=1)

    self.feature_sequence = []

    feature_id = 0
    class_id = report_class_ids[0]

    if report_timestamps is not None:
      start_timestamp = report_timestamps[0]
      start_timestamp_qa_flag = qa_flags[0]
    else:
      start_timestamp = None
      start_timestamp_qa_flag = None

    start_frame_number = report_frame_numbers[0]

    for i in range(1, len(report_class_ids)):
      if report_class_ids[i] != class_id:
        if report_timestamps is not None:
          end_timestamp = report_timestamps[i - 1]
          end_timestamp_qa_flag = qa_flags[i - 1]
        else:
          end_timestamp = None
          end_timestamp_qa_flag = None

        end_frame_number = report_frame_numbers[i - 1]

        # the beginning of the next feature has been reached.
        # create an object for the preceding feature.
        self.feature_sequence.append(Feature(
          feature_id, class_id, self.class_names[class_id], start_timestamp,
          end_timestamp, start_timestamp_qa_flag, end_timestamp_qa_flag,
          start_frame_number, end_frame_number))

        # logging.debug('added feature:\n{}'.format(self.feature_sequence[-1]))

        feature_id += 1
        class_id = report_class_ids[i]

        if report_timestamps is not None:
          start_timestamp = report_timestamps[i]
          start_timestamp_qa_flag = qa_flags[i]
        else:
          start_timestamp = None
          start_timestamp_qa_flag = None

        start_frame_number = report_frame_numbers[i]

        if i == len(report_class_ids) - 1:
          self.feature_sequence.append(Feature(
            feature_id, class_id, self.class_names[class_id], start_timestamp,
            start_timestamp, start_timestamp_qa_flag, start_timestamp_qa_flag,
            start_frame_number, start_frame_number))
      elif i == len(report_class_ids) - 1:
        if report_timestamps is not None:
          end_timestamp = report_timestamps[i]
          end_timestamp_qa_flag = qa_flags[i]
        else:
          end_timestamp = None
          end_timestamp_qa_flag = None

        end_frame_number = report_frame_numbers[i]

        self.feature_sequence.append(Feature(
          feature_id, class_id, self.class_names[class_id], start_timestamp,
          end_timestamp, start_timestamp_qa_flag, end_timestamp_qa_flag,
          start_frame_number, end_frame_number))

    self.weight_scale = non_event_weight_scale

  def find_events(
      self, target_feature_class_ids, target_feature_class_names=None,
      preceding_feature_class_id=None, preceding_feature_class_name=None,
      following_feature_class_id=None, following_feature_class_name=None,
      minimum_event_length=100):
    if target_feature_class_ids is None:
      if target_feature_class_names is None:
        raise ValueError('target_feature_class_ids and target_'
                         'feature_class_names cannot both be None')
      else:
        target_feature_class_ids = [self.class_ids[name]
                                    for name in target_feature_class_names]

    if preceding_feature_class_id is None and \
            preceding_feature_class_name is not None:
      preceding_feature_class_id = self.class_ids[preceding_feature_class_name]

    if preceding_feature_class_id in target_feature_class_ids:
      raise ValueError('preceding_feature_class_id cannot be equal to any '
                       'target_feature_class_id')

    if following_feature_class_id is None and \
            following_feature_class_name is not None:
      following_feature_class_id = self.class_ids[following_feature_class_name]

    if following_feature_class_id in target_feature_class_ids:
      raise ValueError('following_feature_class_id cannot be equal to any '
                       'target_feature_class_id')

    events = []

    previous_event = None

    event_id = 0

    i = 0

    weight = 0.0

    if preceding_feature_class_id and following_feature_class_id:
      previous_preceding_feature = None
      previous_following_feature = None

      while i < len(self.feature_sequence):
        current_feature = self.feature_sequence[i]
        i += 1
        if current_feature.class_id in target_feature_class_ids:
          target_feature_list = [current_feature]
          longest_target_feature_gap = 0
          # logging.debug('increasing weight from {} to {}'.format(
          #   weight, weight + current_feature.length))
          weight += current_feature.length

          while i < len(self.feature_sequence) and current_feature.class_id \
              not in [preceding_feature_class_id, following_feature_class_id]:
            current_feature = self.feature_sequence[i]
            i += 1
            if current_feature.class_id in target_feature_class_ids:
              current_feature_gap = current_feature.start_frame_number - \
                      target_feature_list[-1].end_frame_number
              if longest_target_feature_gap < current_feature_gap:
                longest_target_feature_gap = current_feature_gap

              target_feature_list.append(current_feature)
              # logging.debug('increasing weight from {} to {}'.format(
              #   weight, weight + current_feature.length))
              weight += current_feature.length
            else:
              # logging.debug('decreasing weight from {} to {}'.format(
              #   weight, weight - self.weight_scale * current_feature.length))
              weight -= self.weight_scale * current_feature.length

            if weight <= 0:
              logging.debug('event detection complete')
              break

          # if a detected event begins or ends in a frame from which the timestamp
          # could not be read or syntehsized, just ignore the event.
          # if not (target_feature_list[0].start_timestamp == -1
          #         or target_feature_list[-1].end_timestamp == -1):
          current_event = Event(event_id=event_id,
                                target_feature_list=target_feature_list)

          weight = 0

          # if two consecutive events share a common following/preceding
          # feature, and that feature is closer to the current event than the
          # previous event, reassign it to the current event.
          if previous_preceding_feature:
            if current_event.start_frame_number - \
                previous_preceding_feature.end_frame_number < \
                longest_target_feature_gap * 10:
              if previous_preceding_feature.event_id:
                previous_target_feature = events[
                  previous_preceding_feature.event_id].target_feature_list[-1]

                previous_target_feature_distance = \
                  previous_preceding_feature.start_frame_number - \
                  previous_target_feature.end_frame_number

                assert previous_target_feature_distance >= 0

                current_feature_distance = \
                  current_event.target_feature_list[0].start_frame_number - \
                  previous_preceding_feature.end_frame_number

                assert current_feature_distance >= 0

                if current_feature_distance < previous_target_feature_distance:
                  previous_event.following_feature = None
                  current_event.preceding_feature = previous_preceding_feature
                  previous_preceding_feature.event_id = event_id
              else:
                current_event.preceding_feature = previous_preceding_feature
                previous_preceding_feature.event_id = event_id

              if previous_preceding_feature == previous_following_feature:
                previous_following_feature = None

              previous_preceding_feature = None

          if current_event.length >= minimum_event_length:
            events.append(current_event)
            event_id += 1
            previous_event = current_event

        if current_feature.class_id == preceding_feature_class_id:
          previous_preceding_feature = current_feature

        if current_feature.class_id == following_feature_class_id:
          previous_following_feature = current_feature

          if previous_event and \
                  previous_event.following_feature is None:
            previous_event.following_feature = previous_following_feature
            previous_following_feature.event_id = previous_event.event_id
            previous_following_feature = None
    elif not preceding_feature_class_id and following_feature_class_id:
      while i < len(self.feature_sequence):
        current_feature = self.feature_sequence[i]
        i += 1
        if current_feature.class_id in target_feature_class_ids:
          target_feature_list = [current_feature]

          while i < len(self.feature_sequence) \
              and current_feature.class_id != following_feature_class_id:
            current_feature = self.feature_sequence[i]
            i += 1
            if current_feature.class_id in target_feature_class_ids:
              target_feature_list.append(current_feature)

          current_event = Event(event_id=event_id,
                                target_feature_list=target_feature_list)
          logging.debug('created event with target_feature_list = {}'.format(
            target_feature_list))

          if current_event.length >= minimum_event_length:
            events.append(current_event)
            event_id += 1
            previous_event = current_event

        if current_feature.class_id == following_feature_class_id:
          previous_following_feature = current_feature

          if previous_event and \
                  previous_event.following_feature is None:
            previous_event.following_feature = previous_following_feature
            previous_following_feature.event_id = previous_event.event_id
    elif preceding_feature_class_id and not following_feature_class_id:
      previous_preceding_feature = None

      while i < len(self.feature_sequence):
        current_feature = self.feature_sequence[i]
        i += 1

        if current_feature.class_id in target_feature_class_ids:
          target_feature_list = [current_feature]

          while i < len(self.feature_sequence) \
              and current_feature.class_id != preceding_feature_class_id:
            current_feature = self.feature_sequence[i]
            i += 1

            if current_feature.class_id in target_feature_class_ids:
              target_feature_list.append(current_feature)

          current_event = Event(event_id=event_id,
                                target_feature_list=target_feature_list)
          logging.debug('created event with target_feature_list = {}'.format(
            target_feature_list))

          # if two consecutive events share a common following/preceding
          # feature, and that feature is closer to the current event than the
          # previous event, reassign it to the current event.
          if previous_preceding_feature:
            current_event.preceding_feature = previous_preceding_feature
            previous_preceding_feature.event_id = event_id
            previous_preceding_feature = None

          if current_event.length >= minimum_event_length:
            events.append(current_event)
            event_id += 1

        if current_feature.class_id == preceding_feature_class_id:
          previous_preceding_feature = current_feature
    else:
      while i < len(self.feature_sequence):
        current_feature = self.feature_sequence[i]
        i += 1
        if current_feature.class_id in target_feature_class_ids:
          target_feature_list = [current_feature]
          longest_target_feature_gap = 0
          # logging.debug('increasing weight from {} to {}'.format(
          #   weight, weight + current_feature.length))
          weight += current_feature.length

          while i < len(self.feature_sequence):
            current_feature = self.feature_sequence[i]
            i += 1
            if current_feature.class_id in target_feature_class_ids:
              current_feature_gap = current_feature.start_frame_number - \
                      target_feature_list[-1].end_frame_number
              if longest_target_feature_gap < current_feature_gap:
                longest_target_feature_gap = current_feature_gap

              target_feature_list.append(current_feature)
              # logging.debug('increasing weight from {} to {}'.format(
              #   weight, weight + current_feature.length))
              weight += current_feature.length
            else:
              # logging.debug('decreasing weight from {} to {}'.format(
              #   weight, weight - self.weight_scale * current_feature.length))
              weight -= self.weight_scale * current_feature.length

            if weight <= 0:
              logging.debug('event detection complete')
              break

          current_event = Event(event_id=event_id,
                                target_feature_list=target_feature_list)

          weight = 0
          # logging.debug('created event with target_feature_list = {}'.format(
          #   target_feature_list))
          if current_event.length >= minimum_event_length:
            events.append(current_event)
            event_id += 1

    return events

  def find_work_zone_events(self):
    return self.find_events(
      target_feature_class_ids=[self.class_ids['regulatory_sign'],
                                self.class_ids['warning_sign'],
                                self.class_ids['work_zone']],
      target_feature_class_names=[
        'regulatory_sign', 'warning_sign', 'work_zone'],
      preceding_feature_class_id=None,
      preceding_feature_class_name=None,
      following_feature_class_id=None,
      following_feature_class_name=None
    )


class TripFromReportFile(Trip):
  def __init__(self, report_file_path, class_names_file_path,
               smooth_probs=False, smoothing_factor=16):
    class_name_map = IO.read_class_names(class_names_file_path)

    class_header_names = [class_name + '_probability'
                          for class_name in class_name_map.values()]

    header_mask = ['frame_number', 'frame_timestamp', 'qa_flag']
    header_mask.extend(class_header_names)

    report_header, report_data, data_col_range = IO.read_report(
      report_file_path, frame_col_num=1, timestamp_col_num=2, qa_flag_col_num=3,
      header_mask=header_mask, return_data_col_range=True)

    report_frame_numbers = report_data['frame_numbers']
    report_frame_numbers = report_frame_numbers.astype(np.int32)

    try:
      report_timestamps = report_data['frame_timestamps']
      report_timestamps = report_timestamps.astype(np.int32)
      qa_flags = report_data['qa_flag']
    except:
      report_timestamps = None
      qa_flags = None

    report_probs = report_data['probabilities']
    report_probs = report_probs.astype(np.float32)

    if smooth_probs:
      report_probs = IO.smooth_probs(report_probs, smoothing_factor)

    Trip.__init__(self, report_frame_numbers, report_timestamps, qa_flags,
                  report_probs, class_name_map)
