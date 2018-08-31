import logging
import numpy as np


class Timestamp:
  digit_mask_array = np.array(
    [[[0, 0, 0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0, 0, 0],
      [0, 0, 0,   0,   0, 255, 255, 255, 255, 255, 255,   0,   0,   0, 0, 0],
      [0, 0, 0,   0,   0, 255, 255, 255, 255, 255, 255,   0,   0,   0, 0, 0],
      [0, 0, 0, 255, 255,   0,   0,   0,   0,   0,   0, 255, 255,   0, 0, 0],
      [0, 0, 0, 255, 255,   0,   0,   0,   0,   0,   0, 255, 255,   0, 0, 0],
      [0, 0, 0, 255, 255,   0,   0,   0,   0, 255, 255, 255, 255,   0, 0, 0],
      [0, 0, 0, 255, 255,   0,   0,   0,   0, 255, 255, 255, 255,   0, 0, 0],
      [0, 0, 0, 255, 255,   0,   0, 255, 255,   0,   0, 255, 255,   0, 0, 0],
      [0, 0, 0, 255, 255,   0,   0, 255, 255,   0,   0, 255, 255,   0, 0, 0],
      [0, 0, 0, 255, 255, 255, 255,   0,   0,   0,   0, 255, 255,   0, 0, 0],
      [0, 0, 0, 255, 255, 255, 255,   0,   0,   0,   0, 255, 255,   0, 0, 0],
      [0, 0, 0, 255, 255,   0,   0,   0,   0,   0,   0, 255, 255,   0, 0, 0],
      [0, 0, 0, 255, 255,   0,   0,   0,   0,   0,   0, 255, 255,   0, 0, 0],
      [0, 0, 0,   0,   0, 255, 255, 255, 255, 255, 255,   0,   0,   0, 0, 0],
      [0, 0, 0,   0,   0, 255, 255, 255, 255, 255, 255,   0,   0,   0, 0, 0],
      [0, 0, 0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0, 0, 0]],
     [[0, 0, 0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0, 0, 0],
      [0, 0, 0,   0,   0,   0,   0,   0, 255, 255,   0,   0,   0,   0, 0, 0],
      [0, 0, 0,   0,   0,   0,   0, 255, 255, 255,   0,   0,   0,   0, 0, 0],
      [0, 0, 0,   0,   0,   0, 255, 255, 255, 255,   0,   0,   0,   0, 0, 0],
      [0, 0, 0,   0,   0,   0,   0,   0, 255, 255,   0,   0,   0,   0, 0, 0],
      [0, 0, 0,   0,   0,   0,   0,   0, 255, 255,   0,   0,   0,   0, 0, 0],
      [0, 0, 0,   0,   0,   0,   0,   0, 255, 255,   0,   0,   0,   0, 0, 0],
      [0, 0, 0,   0,   0,   0,   0,   0, 255, 255,   0,   0,   0,   0, 0, 0],
      [0, 0, 0,   0,   0,   0,   0,   0, 255, 255,   0,   0,   0,   0, 0, 0],
      [0, 0, 0,   0,   0,   0,   0,   0, 255, 255,   0,   0,   0,   0, 0, 0],
      [0, 0, 0,   0,   0,   0,   0,   0, 255, 255,   0,   0,   0,   0, 0, 0],
      [0, 0, 0,   0,   0,   0,   0,   0, 255, 255,   0,   0,   0,   0, 0, 0],
      [0, 0, 0,   0,   0,   0,   0,   0, 255, 255,   0,   0,   0,   0, 0, 0],
      [0, 0, 0,   0,   0,   0,   0, 255, 255, 255, 255,   0,   0,   0, 0, 0],
      [0, 0, 0,   0,   0,   0, 255, 255, 255, 255, 255, 255,   0,   0, 0, 0],
      [0, 0, 0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0, 0, 0]],
     [[0, 0, 0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0, 0, 0],
      [0, 0, 0,   0,   0, 255, 255, 255, 255, 255, 255, 255,   0,   0, 0, 0],
      [0, 0, 0,   0,   0, 255, 255, 255, 255, 255, 255, 255,   0,   0, 0, 0],
      [0, 0, 0, 255, 255,   0,   0,   0,   0,   0, 255, 255,   0,   0, 0, 0],
      [0, 0, 0, 255, 255,   0,   0,   0,   0,   0, 255, 255,   0,   0, 0, 0],
      [0, 0, 0,   0,   0,   0,   0,   0,   0,   0, 255, 255,   0,   0, 0, 0],
      [0, 0, 0,   0,   0,   0,   0,   0,   0,   0, 255, 255,   0,   0, 0, 0],
      [0, 0, 0,   0,   0,   0,   0,   0, 255, 255,   0,   0,   0,   0, 0, 0],
      [0, 0, 0,   0,   0,   0,   0,   0, 255, 255,   0,   0,   0,   0, 0, 0],
      [0, 0, 0,   0,   0,   0, 255, 255,   0,   0,   0,   0,   0,   0, 0, 0],
      [0, 0, 0,   0,   0,   0, 255, 255,   0,   0,   0,   0,   0,   0, 0, 0],
      [0, 0, 0,   0, 255, 255,   0,   0,   0,   0,   0,   0,   0,   0, 0, 0],
      [0, 0, 0,   0, 255, 255,   0,   0,   0,   0,   0,   0,   0,   0, 0, 0],
      [0, 0, 0, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,   0, 0, 0],
      [0, 0, 0, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,   0, 0, 0],
      [0, 0, 0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0, 0, 0]],
     [[0, 0, 0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0, 0, 0],
      [0, 0, 0, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,   0, 0, 0],
      [0, 0, 0, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,   0, 0, 0],
      [0, 0, 0,   0,   0,   0,   0,   0,   0, 255, 255,   0,   0,   0, 0, 0],
      [0, 0, 0,   0,   0,   0,   0,   0,   0, 255, 255,   0,   0,   0, 0, 0],
      [0, 0, 0,   0,   0,   0,   0, 255, 255,   0,   0,   0,   0,   0, 0, 0],
      [0, 0, 0,   0,   0,   0,   0, 255, 255,   0,   0,   0,   0,   0, 0, 0],
      [0, 0, 0,   0,   0,   0,   0,   0,   0, 255, 255,   0,   0,   0, 0, 0],
      [0, 0, 0,   0,   0,   0,   0,   0,   0, 255, 255,   0,   0,   0, 0, 0],
      [0, 0, 0,   0,   0,   0,   0,   0,   0,   0,   0, 255, 255,   0, 0, 0],
      [0, 0, 0,   0,   0,   0,   0,   0,   0,   0,   0, 255, 255,   0, 0, 0],
      [0, 0, 0, 255, 255,   0,   0,   0,   0,   0,   0, 255, 255,   0, 0, 0],
      [0, 0, 0, 255, 255,   0,   0,   0,   0,   0,   0, 255, 255,   0, 0, 0],
      [0, 0, 0,   0,   0, 255, 255, 255, 255, 255, 255,   0,   0,   0, 0, 0],
      [0, 0, 0,   0,   0, 255, 255, 255, 255, 255, 255,   0,   0,   0, 0, 0],
      [0, 0, 0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0, 0, 0]],
     [[0, 0, 0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0, 0, 0],
      [0, 0, 0,   0,   0,   0,   0,   0,   0, 255, 255,   0,   0,   0, 0, 0],
      [0, 0, 0,   0,   0,   0,   0,   0,   0, 255, 255,   0,   0,   0, 0, 0],
      [0, 0, 0,   0,   0,   0,   0, 255, 255, 255, 255,   0,   0,   0, 0, 0],
      [0, 0, 0,   0,   0,   0,   0, 255, 255, 255, 255,   0,   0,   0, 0, 0],
      [0, 0, 0,   0,   0, 255, 255,   0,   0, 255, 255,   0,   0,   0, 0, 0],
      [0, 0, 0,   0,   0, 255, 255,   0,   0, 255, 255,   0,   0,   0, 0, 0],
      [0, 0, 0, 255, 255,   0,   0,   0,   0, 255, 255,   0,   0,   0, 0, 0],
      [0, 0, 0, 255, 255,   0,   0,   0,   0, 255, 255,   0,   0,   0, 0, 0],
      [0, 0, 0, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,   0, 0, 0],
      [0, 0, 0, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,   0, 0, 0],
      [0, 0, 0,   0,   0,   0,   0,   0,   0, 255, 255,   0,   0,   0, 0, 0],
      [0, 0, 0,   0,   0,   0,   0,   0,   0, 255, 255,   0,   0,   0, 0, 0],
      [0, 0, 0,   0,   0,   0,   0,   0,   0, 255, 255,   0,   0,   0, 0, 0],
      [0, 0, 0,   0,   0,   0,   0,   0,   0, 255, 255,   0,   0,   0, 0, 0],
      [0, 0, 0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0, 0, 0]],
     [[0, 0, 0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0, 0, 0],
      [0, 0, 0, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,   0, 0, 0],
      [0, 0, 0, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,   0, 0, 0],
      [0, 0, 0, 255, 255,   0,   0,   0,   0,   0,   0,   0,   0,   0, 0, 0],
      [0, 0, 0, 255, 255,   0,   0,   0,   0,   0,   0,   0,   0,   0, 0, 0],
      [0, 0, 0, 255, 255, 255, 255, 255, 255, 255, 255,   0,   0,   0, 0, 0],
      [0, 0, 0, 255, 255, 255, 255, 255, 255, 255, 255,   0,   0,   0, 0, 0],
      [0, 0, 0,   0,   0,   0,   0,   0,   0,   0,   0, 255, 255,   0, 0, 0],
      [0, 0, 0,   0,   0,   0,   0,   0,   0,   0,   0, 255, 255,   0, 0, 0],
      [0, 0, 0,   0,   0,   0,   0,   0,   0,   0,   0, 255, 255,   0, 0, 0],
      [0, 0, 0,   0,   0,   0,   0,   0,   0,   0,   0, 255, 255,   0, 0, 0],
      [0, 0, 0,   0,   0,   0,   0,   0,   0,   0,   0, 255, 255,   0, 0, 0],
      [0, 0, 0,   0,   0,   0,   0,   0,   0,   0,   0, 255, 255,   0, 0, 0],
      [0, 0, 0, 255, 255, 255, 255, 255, 255, 255, 255,   0,   0,   0, 0, 0],
      [0, 0, 0, 255, 255, 255, 255, 255, 255, 255, 255,   0,   0,   0, 0, 0],
      [0, 0, 0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0, 0, 0]],
     [[0, 0, 0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0, 0, 0],
      [0, 0, 0,   0,   0,   0,   0, 255, 255, 255, 255,   0,   0,   0, 0, 0],
      [0, 0, 0,   0,   0,   0,   0, 255, 255, 255, 255,   0,   0,   0, 0, 0],
      [0, 0, 0,   0,   0, 255, 255,   0,   0,   0,   0,   0,   0,   0, 0, 0],
      [0, 0, 0,   0,   0, 255, 255,   0,   0,   0,   0,   0,   0,   0, 0, 0],
      [0, 0, 0, 255, 255,   0,   0,   0,   0,   0,   0,   0,   0,   0, 0, 0],
      [0, 0, 0, 255, 255,   0,   0,   0,   0,   0,   0,   0,   0,   0, 0, 0],
      [0, 0, 0, 255, 255, 255, 255, 255, 255, 255, 255,   0,   0,   0, 0, 0],
      [0, 0, 0, 255, 255, 255, 255, 255, 255, 255, 255,   0,   0,   0, 0, 0],
      [0, 0, 0, 255, 255,   0,   0,   0,   0,   0,   0, 255, 255,   0, 0, 0],
      [0, 0, 0, 255, 255,   0,   0,   0,   0,   0,   0, 255, 255,   0, 0, 0],
      [0, 0, 0, 255, 255,   0,   0,   0,   0,   0,   0, 255, 255,   0, 0, 0],
      [0, 0, 0, 255, 255,   0,   0,   0,   0,   0,   0, 255, 255,   0, 0, 0],
      [0, 0, 0,   0,   0, 255, 255, 255, 255, 255, 255,   0,   0,   0, 0, 0],
      [0, 0, 0,   0,   0, 255, 255, 255, 255, 255, 255,   0,   0,   0, 0, 0],
      [0, 0, 0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0, 0, 0]],
     [[0, 0, 0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0, 0, 0],
      [0, 0, 0, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,   0, 0, 0],
      [0, 0, 0, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,   0, 0, 0],
      [0, 0, 0,   0,   0,   0,   0,   0,   0,   0,   0, 255, 255,   0, 0, 0],
      [0, 0, 0,   0,   0,   0,   0,   0,   0,   0,   0, 255, 255,   0, 0, 0],
      [0, 0, 0,   0,   0,   0,   0,   0,   0, 255, 255,   0,   0,   0, 0, 0],
      [0, 0, 0,   0,   0,   0,   0,   0,   0, 255, 255,   0,   0,   0, 0, 0],
      [0, 0, 0,   0,   0,   0,   0, 255, 255,   0,   0,   0,   0,   0, 0, 0],
      [0, 0, 0,   0,   0,   0,   0, 255, 255,   0,   0,   0,   0,   0, 0, 0],
      [0, 0, 0,   0,   0, 255, 255,   0,   0,   0,   0,   0,   0,   0, 0, 0],
      [0, 0, 0,   0,   0, 255, 255,   0,   0,   0,   0,   0,   0,   0, 0, 0],
      [0, 0, 0,   0,   0, 255, 255,   0,   0,   0,   0,   0,   0,   0, 0, 0],
      [0, 0, 0,   0,   0, 255, 255,   0,   0,   0,   0,   0,   0,   0, 0, 0],
      [0, 0, 0,   0,   0, 255, 255,   0,   0,   0,   0,   0,   0,   0, 0, 0],
      [0, 0, 0,   0,   0, 255, 255,   0,   0,   0,   0,   0,   0,   0, 0, 0],
      [0, 0, 0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0, 0, 0]],
     [[0, 0, 0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0, 0, 0],
      [0, 0, 0,   0,   0, 255, 255, 255, 255, 255, 255,   0,   0,   0, 0, 0],
      [0, 0, 0,   0,   0, 255, 255, 255, 255, 255, 255,   0,   0,   0, 0, 0],
      [0, 0, 0, 255, 255,   0,   0,   0,   0,   0,   0, 255, 255,   0, 0, 0],
      [0, 0, 0, 255, 255,   0,   0,   0,   0,   0,   0, 255, 255,   0, 0, 0],
      [0, 0, 0, 255, 255,   0,   0,   0,   0,   0,   0, 255, 255,   0, 0, 0],
      [0, 0, 0, 255, 255,   0,   0,   0,   0,   0,   0, 255, 255,   0, 0, 0],
      [0, 0, 0,   0,   0, 255, 255, 255, 255, 255, 255,   0,   0,   0, 0, 0],
      [0, 0, 0,   0,   0, 255, 255, 255, 255, 255, 255,   0,   0,   0, 0, 0],
      [0, 0, 0, 255, 255,   0,   0,   0,   0,   0,   0, 255, 255,   0, 0, 0],
      [0, 0, 0, 255, 255,   0,   0,   0,   0,   0,   0, 255, 255,   0, 0, 0],
      [0, 0, 0, 255, 255,   0,   0,   0,   0,   0,   0, 255, 255,   0, 0, 0],
      [0, 0, 0, 255, 255,   0,   0,   0,   0,   0,   0, 255, 255,   0, 0, 0],
      [0, 0, 0,   0,   0, 255, 255, 255, 255, 255, 255,   0,   0,   0, 0, 0],
      [0, 0, 0,   0,   0, 255, 255, 255, 255, 255, 255,   0,   0,   0, 0, 0],
      [0, 0, 0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0, 0, 0]],
     [[0, 0, 0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0, 0, 0],
      [0, 0, 0,   0,   0, 255, 255, 255, 255, 255, 255,   0,   0,   0, 0, 0],
      [0, 0, 0,   0,   0, 255, 255, 255, 255, 255, 255,   0,   0,   0, 0, 0],
      [0, 0, 0, 255, 255,   0,   0,   0,   0,   0,   0, 255, 255,   0, 0, 0],
      [0, 0, 0, 255, 255,   0,   0,   0,   0,   0,   0, 255, 255,   0, 0, 0],
      [0, 0, 0, 255, 255,   0,   0,   0,   0,   0,   0, 255, 255,   0, 0, 0],
      [0, 0, 0, 255, 255,   0,   0,   0,   0,   0,   0, 255, 255,   0, 0, 0],
      [0, 0, 0,   0,   0, 255, 255, 255, 255, 255, 255, 255, 255,   0, 0, 0],
      [0, 0, 0,   0,   0, 255, 255, 255, 255, 255, 255, 255, 255,   0, 0, 0],
      [0, 0, 0,   0,   0,   0,   0,   0,   0,   0,   0, 255, 255,   0, 0, 0],
      [0, 0, 0,   0,   0,   0,   0,   0,   0,   0,   0, 255, 255,   0, 0, 0],
      [0, 0, 0,   0,   0,   0,   0,   0,   0, 255, 255,   0,   0,   0, 0, 0],
      [0, 0, 0,   0,   0,   0,   0,   0,   0, 255, 255,   0,   0,   0, 0, 0],
      [0, 0, 0,   0,   0, 255, 255, 255, 255,   0,   0,   0,   0,   0, 0, 0],
      [0, 0, 0,   0,   0, 255, 255, 255, 255,   0,   0,   0,   0,   0, 0, 0],
      [0, 0, 0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0, 0, 0]]])

  def __init__(self, timestamp_height, timestamp_maxwidth):
    self.height = timestamp_height
    self.maxwidth = timestamp_maxwidth
    self.num_digits = int(self.maxwidth / self.height)
    self.digit_mask_array = np.tile(
      Timestamp.digit_mask_array, [self.num_digits])  # (10, 16, 16 * nd)
    self.digit_mask_array = np.transpose(
      self.digit_mask_array, (0, 2, 1))  # (10, 16 * nd, 16)
    self.digit_mask_array = np.reshape(
      self.digit_mask_array,
      (-1, self.num_digits, self.height, self.height))  # (10, nd, 16, 16)

  def _binarize_timestamps(self, timestamp_array):
    timestamp_array = np.average(timestamp_array, axis=2)
    timestamp_array = np.where(timestamp_array >= 128, [255], [0])

    return timestamp_array

  # alternative per-frame implementation in case the per-video method fails,
  # e.g. due to unreadable digit in between readable digits
  def _stringify_timestamps_per_frame(
      self, timestamp_image_array, num_timestamps):
    # (16 * nt, 16 * nd)
    timestamp_image_array = self._binarize_timestamps(timestamp_image_array)
    # (16 * nd, 16 * nt)
    timestamp_image_array = np.transpose(timestamp_image_array)

    # 32-bit ints/uints should be fine given no trip exceeds 24 days in length
    timestamp_string_array = np.ndarray((num_timestamps,), dtype=np.uint32)
    quality_assurance_array = np.zeros((num_timestamps,), dtype=np.uint8)

    current_timestamp_length = 0

    current_range_left_index = None
    previous_timestamp_was_missing = False
    total_true_num_unreadable_timestamps = 0
    total_observed_num_unreadable_timestamps = 0
    total_num_unreadable_sequences = 0

    for i in range(num_timestamps):
      try:
        timestamp_image = timestamp_image_array[
                          :, self.height * i:self.height * (i + 1)]
        # (nd, 16, 16)
        timestamp_image = np.reshape(
          timestamp_image, (self.num_digits, self.height, self.height))
        # (10, nd, 16, 16)
        _equal = np.equal(timestamp_image, self.digit_mask_array)
        # (10, nd)
        _all = np.all(_equal, axis=(2, 3))
        # ((10,), (nd,))
        digits, positions = np.nonzero(_all)

        digits_len = len(digits)

        if digits_len > 0:
          if digits_len >= current_timestamp_length:
            current_timestamp_length = digits_len

            sorted_positions = np.argsort(positions)
            digits = digits[sorted_positions]
            digits = digits.astype(np.unicode_)

            timestamp_string_array[i] = ''.join(digits)

            if previous_timestamp_was_missing:
              previous_timestamp_was_missing = False

              # we don't subtract 1 from current_range_left_index because frame
              # numbers are indexed starting at 1, not 0
              earlier_readable_timestamp = timestamp_string_array[
                current_range_left_index]

              later_readable_timestamp = timestamp_string_array[i]

              observed_num_unreadable_timestamps = i - current_range_left_index

              milliseconds_between_readable_timestamps = \
                later_readable_timestamp - earlier_readable_timestamp

              mod_67_remainder = milliseconds_between_readable_timestamps % 67

              div_67_whole = int(
                milliseconds_between_readable_timestamps / 67)

              if mod_67_remainder == 0:
                num_66_occurrences = 0
                num_67_occurrences = div_67_whole
              else:
                num_66_occurrences = 66 - mod_67_remainder

                num_67_occurrences = div_67_whole - num_66_occurrences

                num_66_occurrences += 1

              true_num_unreadable_timestamps = \
                num_66_occurrences + num_67_occurrences

              total_true_num_unreadable_timestamps += true_num_unreadable_timestamps
              total_observed_num_unreadable_timestamps += observed_num_unreadable_timestamps

              # if no frames are inferred to be missing
              if observed_num_unreadable_timestamps == true_num_unreadable_timestamps:
                timesteps = [66 for _ in range(num_66_occurrences)]
                timesteps.extend([67 for _ in range(num_67_occurrences)])

                timesteps = np.array(timesteps)

                np.random.shuffle(timesteps)

                cumulative_timesteps = 0

                for j in range(observed_num_unreadable_timestamps - 1):
                  cumulative_timesteps += timesteps[j]
                  timestamp_string_array[current_range_left_index + 1 + j] = \
                    earlier_readable_timestamp + cumulative_timesteps
                  quality_assurance_array[current_range_left_index + 1 + j] = 1
              else:  # if at least one frame is inferred to be missing
                for j in range(observed_num_unreadable_timestamps - 1):
                  timestamp_string_array[current_range_left_index + 1 + j] = 0
                total_num_unreadable_sequences += 1
          else:
            timestamp_string_array[i] = 0

            if not previous_timestamp_was_missing:
              previous_timestamp_was_missing = True

              if i > 0:
                current_range_left_index = i - 1
              else:
                logging.error('Unable to synthesize replacements for sequence '
                              'of unreadable timestamps starting with frame 0')
        else:
          timestamp_string_array[i] = 0

          if not previous_timestamp_was_missing:
            previous_timestamp_was_missing = True

            if i > 0:
              current_range_left_index = i - 1
              logging.error('setting current_range_left_index to {}'.format(i))
            else:
              logging.error('Unable to synthesize replacements for sequence of '
                            'unreadable timestamps starting with frame 0')
      except Exception as e:
        logging.debug('the {}th timestamp could not be interpreted or '
                      'synthesized'.format(i))
        logging.error(e)

    logging.debug(
      '{} frames predicted to be missing across {} instances of observed signal'
      ' loss'.format(total_true_num_unreadable_timestamps -
                     total_observed_num_unreadable_timestamps,
                     total_num_unreadable_sequences))

    # handle case where last timestamp is missing
    timestamp_errors = timestamp_string_array == 0

    if len(timestamp_errors) > 0:
      logging.warning(
        '{} timestamps could not be read nor synthesized and will'
        ' be placeheld using the QA value -1.'.format(len(timestamp_errors)))

    timestamp_string_array = timestamp_string_array.astype(np.unicode_)
    timestamp_string_array[timestamp_errors] = '-1'  # for quality control
    quality_assurance_array[timestamp_errors] = 2

    quality_assurance_array = quality_assurance_array.astype(np.unicode_)

    return timestamp_string_array, quality_assurance_array

  # (16 * nt, 16 * nd, nc)
  def _stringify_timestamps(self, timestamp_image_array, num_timestamps):
    # (16 * nt, 16 * nd)
    timestamp_image_array = self._binarize_timestamps(timestamp_image_array)
    # (nt, 16, 16 * nd)
    timestamp_image_array = np.reshape(
      timestamp_image_array, (num_timestamps, self.height, self.maxwidth))
    # (nt, 16 * nd, 16)
    timestamp_image_array = np.transpose(
      timestamp_image_array, (0, 2, 1))
    # (nt, nd, 16, 16)
    timestamp_image_array = np.reshape(
      timestamp_image_array,
      (num_timestamps, self.num_digits, self.height, self.height))
    # (nt, 1, nd, 16, 16)
    timestamp_image_array = np.expand_dims(timestamp_image_array, 1)
    # (nt, 10, nd, 16, 16)
    _equal = np.equal(timestamp_image_array, self.digit_mask_array)
    # (nt, 10, nd)
    _all = np.all(_equal, axis=(3, 4))
    # ((nt,), (nd,), (nd,))
    frame_numbers, digits, positions = np.nonzero(_all)

    unique_frame_numbers, counts = np.unique(frame_numbers, return_counts=True)

    unique_counts, unique_count_indices, unique_count_counts = np.unique(
      counts, return_index=True, return_counts=True)

    if np.sum(unique_count_counts) != num_timestamps:
      raise ValueError(
        'The cumulative sum of frames having a common number of digits should '
        'be equal to the total number of frames regardless of digit length')

    # the unique count indices should be monotonically increasing
    for i in range(1, len(unique_count_indices)):
      if unique_count_indices[i] < unique_count_indices[i - 1]:
        raise ValueError(
          'Timestamp string lengths should be monotonically non-decreasing, but'
          ' a timestamp of length {} at array index {} follows a timestamp of '
          'length {} at array index {}'.format(
            unique_counts[i - 1], unique_count_indices[i - 1],
            unique_counts[i], unique_count_indices[i]))

    digits = digits.astype(np.unicode_)

    timestamp_string_array = np.ndarray((num_timestamps,), dtype=np.uint32)

    r_idx = 0

    for i in range(unique_counts.shape[0]):
      unique_count = unique_counts[i]

      unique_count_count = unique_count_counts[i]

      num_count_len_digits = unique_count * unique_count_count

      l_idx = r_idx
      r_idx = l_idx + num_count_len_digits

      count_len_positions = positions[l_idx:r_idx]
      count_len_positions = np.reshape(
        count_len_positions, (unique_count_count, unique_count))
      count_len_positions = np.argsort(count_len_positions)

      offsets = np.arange(0, num_count_len_digits, unique_count)
      offsets = np.expand_dims(offsets, axis=1)

      count_len_positions = np.add(offsets, count_len_positions)

      count_len_digits = digits[l_idx:r_idx][count_len_positions]

      unique_count_idx = unique_count_indices[i]

      for j in range(unique_count_idx, unique_count_idx + unique_count_count):
        timestamp_string_array[j] = ''.join(
          count_len_digits[j - unique_count_idx])

    return timestamp_string_array, np.zeros((num_timestamps,), dtype=np.uint8)

  def stringify_timestamps(self, timestamp_image_array):
    num_timestamps = int(timestamp_image_array.shape[0] / self.height)

    try:
      timestamp_string_array, quality_assurance_array = \
        self._stringify_timestamps(
          timestamp_image_array, num_timestamps)
      return timestamp_string_array, quality_assurance_array
    except Exception as e:
      logging.warning('encountered an exception while converting timestamp '
                      'images to strings en masse')
      logging.warning(e)
      logging.warning('will re-attempt conversions one-at-a-time')

    try:
      # slower, but will isolate and gracefully handle individual failures
      timestamp_string_array, quality_assurance_array = \
        self._stringify_timestamps_per_frame(
          timestamp_image_array, num_timestamps)

      return timestamp_string_array, quality_assurance_array
    except Exception as e:
      logging.debug('encountered an exception while converting '
                    'timestamp images to strings one-at-a-time.')
      logging.debug('will raise exception to caller')

      raise e



