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
    binary_timestamp_array = np.average(timestamp_array, axis=2)
    above_threshold_indices = binary_timestamp_array >= 128
    binary_timestamp_array[:] = 0
    binary_timestamp_array[above_threshold_indices] = 255

    return binary_timestamp_array

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

    current_timestamp_length = 0

    # TODO confirm that the lengths match and that no gaps exist between digits
    # TODO: quality controll all unreadable timestamps, not just blank ones
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
          else:
            # TODO synthetically generate timestamp and set quality control bit
            timestamp_string_array[i] = 0
        else:
          timestamp_string_array[i] = 0
      except Exception as e:
        # TODO log error message when a timestamp image cannot be interpreted
        logging.debug(
          'the {}th timestamp could not be interpreted or synthesized'.format(i))
        logging.error(e)

    timestamp_errors = timestamp_string_array == 0

    if len(timestamp_errors) > 0:
      logging.warning(
        'Some timestamp strings were observed to have lengths shorter than '
      #   'timestamps preceding them and had to be symthetically regenerated.')
        'timestamps preceding them and had to be replaced with QA value -1.')

    timestamp_string_array = timestamp_string_array.astype(np.unicode_)
    timestamp_string_array[timestamp_errors] = '-1'  # for quality control

    return timestamp_string_array

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
    # (nt, 10, 10)
    _all = np.all(_equal, axis=(3, 4))
    # ((nt,), (nd,), (nd,))
    frame_numbers, digits, positions = np.nonzero(_all)

    digits = digits.astype(np.unicode_)

    timestamp_string_array = np.ndarray((num_timestamps,), dtype=np.uint32)

    # TODO: confirm that counts are monotonically non-decreasing,
    # else re-attempt with frame-level function
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

    r_idx = 0

    for i in range(unique_counts.shape[0]):
      unique_count = unique_counts[i]

      unique_count_count = unique_count_counts[i]

      l_idx = r_idx
      r_idx = l_idx + unique_count * unique_count_count

      count_len_timestamp_digits = digits[l_idx:r_idx]

      count_len_timestamp_positions = positions[l_idx:r_idx]
      count_len_timestamp_positions = np.reshape(
        count_len_timestamp_positions, (unique_count_count, unique_count))
      count_len_timestamp_positions = np.argsort(count_len_timestamp_positions)

      offsets = np.arange(0, unique_count * unique_count_count, unique_count)
      offsets = np.expand_dims(offsets, axis=1)

      count_len_timestamp_positions = offsets + count_len_timestamp_positions

      count_len_timestamp_digits = count_len_timestamp_digits[
        count_len_timestamp_positions]

      unique_count_idx = unique_count_indices[i]

      for j in range(unique_count_idx, unique_count_idx + unique_count_count):
        timestamp_string_array[j] = ''.join(
          count_len_timestamp_digits[j - unique_count_idx])

    return timestamp_string_array

  def stringify_timestamps(self, timestamp_image_array):
    num_timestamps = int(timestamp_image_array.shape[0] / self.height)

    try:
      timestamp_string_array = self._stringify_timestamps(
        timestamp_image_array, num_timestamps)
      return timestamp_string_array
    except Exception as e:
      logging.warning('encountered an exception while converting timestamp '
                      'images to strings en masse')
      logging.warning(e)
      logging.warning('will re-attempt conversions one-at-a-time')

    try:
      # slower, but will isolate and gracefully handle individual failures
      timestamp_string_array = self._stringify_timestamps_per_frame(
        timestamp_image_array, num_timestamps)

      return timestamp_string_array
    except Exception as e:
      logging.debug('encountered an exception while converting '
                    'timestamp images to strings one-at-a-time.')
      logging.debug('will raise exception to caller')

      raise e



