import logging
import numpy as np
import os
# from PIL import Image

path = os.path


class Timestamp:
  digit_mask_array = np.array([[[0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
                                [0,   0,   0,   0,   0, 255, 255, 255, 255, 255, 255,   0,   0,   0,   0,   0],
                                [0,   0,   0,   0,   0, 255, 255, 255, 255, 255, 255,   0,   0,   0,   0,   0],
                                [0,   0,   0, 255, 255,   0,   0,   0,   0,   0,   0, 255, 255,   0,   0,   0],
                                [0,   0,   0, 255, 255,   0,   0,   0,   0,   0,   0, 255, 255,   0,   0,   0],
                                [0,   0,   0, 255, 255,   0,   0,   0,   0, 255, 255, 255, 255,   0,   0,   0],
                                [0,   0,   0, 255, 255,   0,   0,   0,   0, 255, 255, 255, 255,   0,   0,   0],
                                [0,   0,   0, 255, 255,   0,   0, 255, 255,   0,   0, 255, 255,   0,   0,   0],
                                [0,   0,   0, 255, 255,   0,   0, 255, 255,   0,   0, 255, 255,   0,   0,   0],
                                [0,   0,   0, 255, 255, 255, 255,   0,   0,   0,   0, 255, 255,   0,   0,   0],
                                [0,   0,   0, 255, 255, 255, 255,   0,   0,   0,   0, 255, 255,   0,   0,   0],
                                [0,   0,   0, 255, 255,   0,   0,   0,   0,   0,   0, 255, 255,   0,   0,   0],
                                [0,   0,   0, 255, 255,   0,   0,   0,   0,   0,   0, 255, 255,   0,   0,   0],
                                [0,   0,   0,   0,   0, 255, 255, 255, 255, 255, 255,   0,   0,   0,   0,   0],
                                [0,   0,   0,   0,   0, 255, 255, 255, 255, 255, 255,   0,   0,   0,   0,   0],
                                [0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0]],
                               [[0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
                                [0,   0,   0,   0,   0,   0,   0,   0, 255, 255,   0,   0,   0,   0,   0,   0],
                                [0,   0,   0,   0,   0,   0,   0, 255, 255, 255,   0,   0,   0,   0,   0,   0],
                                [0,   0,   0,   0,   0,   0, 255, 255, 255, 255,   0,   0,   0,   0,   0,   0],
                                [0,   0,   0,   0,   0,   0,   0,   0, 255, 255,   0,   0,   0,   0,   0,   0],
                                [0,   0,   0,   0,   0,   0,   0,   0, 255, 255,   0,   0,   0,   0,   0,   0],
                                [0,   0,   0,   0,   0,   0,   0,   0, 255, 255,   0,   0,   0,   0,   0,   0],
                                [0,   0,   0,   0,   0,   0,   0,   0, 255, 255,   0,   0,   0,   0,   0,   0],
                                [0,   0,   0,   0,   0,   0,   0,   0, 255, 255,   0,   0,   0,   0,   0,   0],
                                [0,   0,   0,   0,   0,   0,   0,   0, 255, 255,   0,   0,   0,   0,   0,   0],
                                [0,   0,   0,   0,   0,   0,   0,   0, 255, 255,   0,   0,   0,   0,   0,   0],
                                [0,   0,   0,   0,   0,   0,   0,   0, 255, 255,   0,   0,   0,   0,   0,   0],
                                [0,   0,   0,   0,   0,   0,   0,   0, 255, 255,   0,   0,   0,   0,   0,   0],
                                [0,   0,   0,   0,   0,   0,   0, 255, 255, 255, 255,   0,   0,   0,   0,   0],
                                [0,   0,   0,   0,   0,   0, 255, 255, 255, 255, 255, 255,   0,   0,   0,   0],
                                [0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0]],
                               [[0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
                                [0,   0,   0,   0,   0, 255, 255, 255, 255, 255, 255, 255,   0,   0,   0,   0],
                                [0,   0,   0,   0,   0, 255, 255, 255, 255, 255, 255, 255,   0,   0,   0,   0],
                                [0,   0,   0, 255, 255,   0,   0,   0,   0,   0, 255, 255,   0,   0,   0,   0],
                                [0,   0,   0, 255, 255,   0,   0,   0,   0,   0, 255, 255,   0,   0,   0,   0],
                                [0,   0,   0,   0,   0,   0,   0,   0,   0,   0, 255, 255,   0,   0,   0,   0],
                                [0,   0,   0,   0,   0,   0,   0,   0,   0,   0, 255, 255,   0,   0,   0,   0],
                                [0,   0,   0,   0,   0,   0,   0,   0, 255, 255,   0,   0,   0,   0,   0,   0],
                                [0,   0,   0,   0,   0,   0,   0,   0, 255, 255,   0,   0,   0,   0,   0,   0],
                                [0,   0,   0,   0,   0,   0, 255, 255,   0,   0,   0,   0,   0,   0,   0,   0],
                                [0,   0,   0,   0,   0,   0, 255, 255,   0,   0,   0,   0,   0,   0,   0,   0],
                                [0,   0,   0,   0, 255, 255,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
                                [0,   0,   0,   0, 255, 255,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
                                [0,   0,   0, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,   0,   0,   0],
                                [0,   0,   0, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,   0,   0,   0],
                                [0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0]],
                               [[0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
                                [0,   0,   0, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,   0,   0,   0],
                                [0,   0,   0, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,   0,   0,   0],
                                [0,   0,   0,   0,   0,   0,   0,   0,   0, 255, 255,   0,   0,   0,   0,   0],
                                [0,   0,   0,   0,   0,   0,   0,   0,   0, 255, 255,   0,   0,   0,   0,   0],
                                [0,   0,   0,   0,   0,   0,   0, 255, 255,   0,   0,   0,   0,   0,   0,   0],
                                [0,   0,   0,   0,   0,   0,   0, 255, 255,   0,   0,   0,   0,   0,   0,   0],
                                [0,   0,   0,   0,   0,   0,   0,   0,   0, 255, 255,   0,   0,   0,   0,   0],
                                [0,   0,   0,   0,   0,   0,   0,   0,   0, 255, 255,   0,   0,   0,   0,   0],
                                [0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0, 255, 255,   0,   0,   0],
                                [0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0, 255, 255,   0,   0,   0],
                                [0,   0,   0, 255, 255,   0,   0,   0,   0,   0,   0, 255, 255,   0,   0,   0],
                                [0,   0,   0, 255, 255,   0,   0,   0,   0,   0,   0, 255, 255,   0,   0,   0],
                                [0,   0,   0,   0,   0, 255, 255, 255, 255, 255, 255,   0,   0,   0,   0,   0],
                                [0,   0,   0,   0,   0, 255, 255, 255, 255, 255, 255,   0,   0,   0,   0,   0],
                                [0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0]],
                               [[0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
                                [0,   0,   0,   0,   0,   0,   0,   0,   0, 255, 255,   0,   0,   0,   0,   0],
                                [0,   0,   0,   0,   0,   0,   0,   0,   0, 255, 255,   0,   0,   0,   0,   0],
                                [0,   0,   0,   0,   0,   0,   0, 255, 255, 255, 255,   0,   0,   0,   0,   0],
                                [0,   0,   0,   0,   0,   0,   0, 255, 255, 255, 255,   0,   0,   0,   0,   0],
                                [0,   0,   0,   0,   0, 255, 255,   0,   0, 255, 255,   0,   0,   0,   0,   0],
                                [0,   0,   0,   0,   0, 255, 255,   0,   0, 255, 255,   0,   0,   0,   0,   0],
                                [0,   0,   0, 255, 255,   0,   0,   0,   0, 255, 255,   0,   0,   0,   0,   0],
                                [0,   0,   0, 255, 255,   0,   0,   0,   0, 255, 255,   0,   0,   0,   0,   0],
                                [0,   0,   0, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,   0,   0,   0],
                                [0,   0,   0, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,   0,   0,   0],
                                [0,   0,   0,   0,   0,   0,   0,   0,   0, 255, 255,   0,   0,   0,   0,   0],
                                [0,   0,   0,   0,   0,   0,   0,   0,   0, 255, 255,   0,   0,   0,   0,   0],
                                [0,   0,   0,   0,   0,   0,   0,   0,   0, 255, 255,   0,   0,   0,   0,   0],
                                [0,   0,   0,   0,   0,   0,   0,   0,   0, 255, 255,   0,   0,   0,   0,   0],
                                [0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0]],
                               [[0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
                                [0,   0,   0, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,   0,   0,   0],
                                [0,   0,   0, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,   0,   0,   0],
                                [0,   0,   0, 255, 255,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
                                [0,   0,   0, 255, 255,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
                                [0,   0,   0, 255, 255, 255, 255, 255, 255, 255, 255,   0,   0,   0,   0,   0],
                                [0,   0,   0, 255, 255, 255, 255, 255, 255, 255, 255,   0,   0,   0,   0,   0],
                                [0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0, 255, 255,   0,   0,   0],
                                [0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0, 255, 255,   0,   0,   0],
                                [0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0, 255, 255,   0,   0,   0],
                                [0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0, 255, 255,   0,   0,   0],
                                [0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0, 255, 255,   0,   0,   0],
                                [0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0, 255, 255,   0,   0,   0],
                                [0,   0,   0, 255, 255, 255, 255, 255, 255, 255, 255,   0,   0,   0,   0,   0],
                                [0,   0,   0, 255, 255, 255, 255, 255, 255, 255, 255,   0,   0,   0,   0,   0],
                                [0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0]],
                               [[0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
                                [0,   0,   0,   0,   0,   0,   0, 255, 255, 255, 255,   0,   0,   0,   0,   0],
                                [0,   0,   0,   0,   0,   0,   0, 255, 255, 255, 255,   0,   0,   0,   0,   0],
                                [0,   0,   0,   0,   0, 255, 255,   0,   0,   0,   0,   0,   0,   0,   0,   0],
                                [0,   0,   0,   0,   0, 255, 255,   0,   0,   0,   0,   0,   0,   0,   0,   0],
                                [0,   0,   0, 255, 255,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
                                [0,   0,   0, 255, 255,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
                                [0,   0,   0, 255, 255, 255, 255, 255, 255, 255, 255,   0,   0,   0,   0,   0],
                                [0,   0,   0, 255, 255, 255, 255, 255, 255, 255, 255,   0,   0,   0,   0,   0],
                                [0,   0,   0, 255, 255,   0,   0,   0,   0,   0,   0, 255, 255,   0,   0,   0],
                                [0,   0,   0, 255, 255,   0,   0,   0,   0,   0,   0, 255, 255,   0,   0,   0],
                                [0,   0,   0, 255, 255,   0,   0,   0,   0,   0,   0, 255, 255,   0,   0,   0],
                                [0,   0,   0, 255, 255,   0,   0,   0,   0,   0,   0, 255, 255,   0,   0,   0],
                                [0,   0,   0,   0,   0, 255, 255, 255, 255, 255, 255,   0,   0,   0,   0,   0],
                                [0,   0,   0,   0,   0, 255, 255, 255, 255, 255, 255,   0,   0,   0,   0,   0],
                                [0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0]],
                               [[0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
                                [0,   0,   0, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,   0,   0,   0],
                                [0,   0,   0, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,   0,   0,   0],
                                [0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0, 255, 255,   0,   0,   0],
                                [0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0, 255, 255,   0,   0,   0],
                                [0,   0,   0,   0,   0,   0,   0,   0,   0, 255, 255,   0,   0,   0,   0,   0],
                                [0,   0,   0,   0,   0,   0,   0,   0,   0, 255, 255,   0,   0,   0,   0,   0],
                                [0,   0,   0,   0,   0,   0,   0, 255, 255,   0,   0,   0,   0,   0,   0,   0],
                                [0,   0,   0,   0,   0,   0,   0, 255, 255,   0,   0,   0,   0,   0,   0,   0],
                                [0,   0,   0,   0,   0, 255, 255,   0,   0,   0,   0,   0,   0,   0,   0,   0],
                                [0,   0,   0,   0,   0, 255, 255,   0,   0,   0,   0,   0,   0,   0,   0,   0],
                                [0,   0,   0,   0,   0, 255, 255,   0,   0,   0,   0,   0,   0,   0,   0,   0],
                                [0,   0,   0,   0,   0, 255, 255,   0,   0,   0,   0,   0,   0,   0,   0,   0],
                                [0,   0,   0,   0,   0, 255, 255,   0,   0,   0,   0,   0,   0,   0,   0,   0],
                                [0,   0,   0,   0,   0, 255, 255,   0,   0,   0,   0,   0,   0,   0,   0,   0],
                                [0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0]],
                               [[0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
                                [0,   0,   0,   0,   0, 255, 255, 255, 255, 255, 255,   0,   0,   0,   0,   0],
                                [0,   0,   0,   0,   0, 255, 255, 255, 255, 255, 255,   0,   0,   0,   0,   0],
                                [0,   0,   0, 255, 255,   0,   0,   0,   0,   0,   0, 255, 255,   0,   0,   0],
                                [0,   0,   0, 255, 255,   0,   0,   0,   0,   0,   0, 255, 255,   0,   0,   0],
                                [0,   0,   0, 255, 255,   0,   0,   0,   0,   0,   0, 255, 255,   0,   0,   0],
                                [0,   0,   0, 255, 255,   0,   0,   0,   0,   0,   0, 255, 255,   0,   0,   0],
                                [0,   0,   0,   0,   0, 255, 255, 255, 255, 255, 255,   0,   0,   0,   0,   0],
                                [0,   0,   0,   0,   0, 255, 255, 255, 255, 255, 255,   0,   0,   0,   0,   0],
                                [0,   0,   0, 255, 255,   0,   0,   0,   0,   0,   0, 255, 255,   0,   0,   0],
                                [0,   0,   0, 255, 255,   0,   0,   0,   0,   0,   0, 255, 255,   0,   0,   0],
                                [0,   0,   0, 255, 255,   0,   0,   0,   0,   0,   0, 255, 255,   0,   0,   0],
                                [0,   0,   0, 255, 255,   0,   0,   0,   0,   0,   0, 255, 255,   0,   0,   0],
                                [0,   0,   0,   0,   0, 255, 255, 255, 255, 255, 255,   0,   0,   0,   0,   0],
                                [0,   0,   0,   0,   0, 255, 255, 255, 255, 255, 255,   0,   0,   0,   0,   0],
                                [0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0]],
                               [[0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
                                [0,   0,   0,   0,   0, 255, 255, 255, 255, 255, 255,   0,   0,   0,   0,   0],
                                [0,   0,   0,   0,   0, 255, 255, 255, 255, 255, 255,   0,   0,   0,   0,   0],
                                [0,   0,   0, 255, 255,   0,   0,   0,   0,   0,   0, 255, 255,   0,   0,   0],
                                [0,   0,   0, 255, 255,   0,   0,   0,   0,   0,   0, 255, 255,   0,   0,   0],
                                [0,   0,   0, 255, 255,   0,   0,   0,   0,   0,   0, 255, 255,   0,   0,   0],
                                [0,   0,   0, 255, 255,   0,   0,   0,   0,   0,   0, 255, 255,   0,   0,   0],
                                [0,   0,   0,   0,   0, 255, 255, 255, 255, 255, 255, 255, 255,   0,   0,   0],
                                [0,   0,   0,   0,   0, 255, 255, 255, 255, 255, 255, 255, 255,   0,   0,   0],
                                [0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0, 255, 255,   0,   0,   0],
                                [0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0, 255, 255,   0,   0,   0],
                                [0,   0,   0,   0,   0,   0,   0,   0,   0, 255, 255,   0,   0,   0,   0,   0],
                                [0,   0,   0,   0,   0,   0,   0,   0,   0, 255, 255,   0,   0,   0,   0,   0],
                                [0,   0,   0,   0,   0, 255, 255, 255, 255,   0,   0,   0,   0,   0,   0,   0],
                                [0,   0,   0,   0,   0, 255, 255, 255, 255,   0,   0,   0,   0,   0,   0,   0],
                                [0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0]]])

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

  # alternative per-frame implementation in case the per-video implementation
  # fails, e.g. due to unreadable digit in between readable digits
  def _stringify_timestamps_per_frame(self, timestamp_image_array, num_timestamps):
    logging.debug('Entering timestamp._stringify_timestamps_per_frame')
    timestamp_image_array = self._binarize_timestamps(timestamp_image_array)  # (16 * nt, 16 * nd)
    timestamp_image_array = np.transpose(timestamp_image_array)  # (16 * nd, 16 * nt)
    
    timestamp_string_array = np.ndarray((num_timestamps,), dtype=np.uint32)

    for i in range(num_timestamps):
      # TODO confirm that the lengths match and that no gaps exist between digits
      try:
        timestamp_image = timestamp_image_array[:, self.height*i:self.height*(i+1)]
        timestamp_image = np.reshape(
          timestamp_image, (self.num_digits, self.height, self.height))  # (nd, 16, 16)
        _equal = np.equal(timestamp_image, self.digit_mask_array)  # (10, nd, 16, 16)
        _all = np.all(_equal, axis=(2, 3))  # (10, nd)
        digits, positions = np.nonzero(_all)  # ((10,), (nd,))
        digits = digits.astype(np.unicode_)
        sorted_positions = np.argsort(positions)
        sorted_digits = digits[sorted_positions]
        timestamp_string_array[i] = ''.join(sorted_digits)
      except Exception as e:
        # TODO log error message when a timestamp image cannot be interpreted
        logging.debug(e)
        timestamp_string_array[i] = 0

    timestamp_errors = timestamp_string_array == 0
    timestamp_string_array = timestamp_string_array.astype(np.unicode_)
    timestamp_string_array[timestamp_errors] = ''
    logging.debug('Exiting timestamp._stringify_timestamps_per_frame')

    return timestamp_string_array

  def _stringify_timestamps(self, timestamp_image_array, num_timestamps):  # (16 * nt, 16 * nd, nc)
    logging.debug('Entering timestamp._stringify_timestamps')
    timestamp_image_array = self._binarize_timestamps(
      timestamp_image_array)  # (16 * nt, 16 * nd)
    timestamp_image_array = np.reshape(
      timestamp_image_array, (num_timestamps, self.height, self.maxwidth))  # (nt, 16, 16 * nd)
    timestamp_image_array = np.transpose(
      timestamp_image_array, (0, 2, 1))  # (nt, 16 * nd, 16)
    timestamp_image_array = np.reshape(
      timestamp_image_array,
      (num_timestamps, self.num_digits, self.height, self.height))  # (nt, nd, 16, 16)
    timestamp_image_array = np.expand_dims(timestamp_image_array, 1)  # (nt, 1, nd, 16, 16)

    _equal = np.equal(timestamp_image_array, self.digit_mask_array)  # (nt, 10, nd, 16, 16)
    _all = np.all(_equal, axis=(3, 4))  # (nt, 10, 10)
    frame_numbers, digits, positions = np.nonzero(_all)  # ((nt,), (nd,), (nd,))

    digits = digits.astype(np.unicode_)
    timestamp_string_array = np.ndarray((num_timestamps,), dtype=np.uint32)

    # TODO: confirm that counts are monotonically non-decreasing,
    # else re-attempt with frame-level function
    unique_frame_numbers, counts = np.unique(
      frame_numbers, return_counts=True)

    unique_counts, unique_count_indices, unique_count_counts = np.unique(
      counts, return_index=True, return_counts=True)

    rindex = 0

    for i in range(unique_counts.shape[0]):
      unique_count = unique_counts[i]
      unique_count_count = unique_count_counts[i]

      lindex = rindex
      rindex = lindex + unique_count * unique_count_count

      count_length_timestamp_digits = digits[lindex:rindex]

      count_length_timestamp_positions = positions[lindex:rindex]
      count_length_timestamp_positions = np.reshape(count_length_timestamp_positions,
                                                    (unique_count_count, unique_count))
      count_length_timestamp_positions = np.argsort(count_length_timestamp_positions)

      offsets = np.arange(0, unique_count * unique_count_count, unique_count)
      offsets = np.expand_dims(offsets, axis=1)

      count_length_timestamp_positions = offsets + count_length_timestamp_positions

      count_length_timestamp_digits = count_length_timestamp_digits[
        count_length_timestamp_positions]

      unique_count_index = unique_count_indices[i]

      for j in range(unique_count_index, unique_count_index + unique_count_count):
        timestamp_string_array[j] = ''.join(
          count_length_timestamp_digits[j - unique_count_index])

    logging.debug('Exiting timestamp._stringify_timestamps')
    return timestamp_string_array

  def stringify_timestamps(self, timestamp_image_array):
    logging.debug('Entering timestamp.stringify_timestamps')
    num_timestamps = int(timestamp_image_array.shape[0] / self.height)
    try:
      timestamp_string_array = self._stringify_timestamps(timestamp_image_array,
                                                          num_timestamps)
      logging.debug('Exiting timestamp.stringify_timestamps')
      return timestamp_string_array
    # slower, but will isolate and gracefully handle failures to read timestamp digits
    except Exception as e:
      logging.debug(e)
      timestamp_string_array = self._stringify_timestamps_per_frame(timestamp_image_array,
                                                                    num_timestamps)
      logging.debug('Exiting timestamp.stringify_timestamps')
      return timestamp_string_array

# for debugging
# def save_timestamp_images(timestamp_array, num_timestamps, timestamp_image_height, video_file_path):
#   video_file_name, _ = path.splitext(path.basename(video_file_path))
#   image_dir_path = './timestamp_images/' + video_file_name
#
#   if not path.exists(image_dir_path):
#     os.makedirs(image_dir_path)
#
#   for i in range(num_timestamps):
#     image_file_name = '{}_Timestamp_{:07d}.jpg'.format(video_file_name, i)
#     image_file_path = path.join(image_dir_path, image_file_name)
#     image = Image.fromarray(timestamp_array[timestamp_image_height*i:timestamp_image_height*(i+1)])
#     image.save(image_file_path, 'JPEG')

# def save_numeral_mask(
#     digit_image, video_file_path, video_frame_number, digit_position, digit_value):
#   video_file_name, _ = path.splitext(path.basename(video_file_path))
#   image_dir_path = './numeral_masks/' + video_file_name
#
  # if not path.exists(image_dir_path):
  #   os.makedirs(image_dir_path)
#
#   image_file_name = '{}_Timestamp_{:07d}_Digit_{}_Equals_{}.jpg'.format(
#     video_file_name, video_frame_number, digit_position, digit_value)
#   image_file_path = path.join(image_dir_path, image_file_name)
#   image = Image.fromarray(digit_image)
#   image.save(image_file_path, 'JPEG')

# def save_timestamp_digit(
#     digit_image, video_file_path, video_frame_number, digit_position, digit_value):
#   video_file_name, _ = path.splitext(path.basename(video_file_path))
#   image_dir_path = './timestamp_images/' + video_file_name
#
#   if not path.exists(image_dir_path):
#     os.makedirs(image_dir_path)
#
#   image_file_name = '{}_Timestamp_{:07d}_Digit_{}_Equals_{}.jpg'.format(
#     video_file_name, video_frame_number, digit_position, digit_value)
#   image_file_path = path.join(image_dir_path, image_file_name)
#   image = Image.fromarray(digit_image)
#   image.save(image_file_path, 'JPEG')


