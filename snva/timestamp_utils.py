import numpy as np
import os
# from PIL import Image

path = os.path

numeral_mask_map = {'0': np.array([[0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
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
                                  dtype=np.uint8),
                    '1': np.array([[0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
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
                                  dtype=np.uint8),
                    '2': np.array([[0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
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
                                  dtype=np.uint8),
                    '3': np.array([[0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
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
                                  dtype=np.uint8),
                    '4': np.array([[0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
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
                                  dtype=np.uint8),
                    '5': np.array([[0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
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
                                  dtype=np.uint8),
                    '6': np.array([[0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
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
                                  dtype=np.uint8),
                    '7': np.array([[0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
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
                                  dtype=np.uint8),
                    '8': np.array([[0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
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
                                  dtype=np.uint8),
                    '9': np.array([[0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
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
                                   [0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0]],
                                  dtype=np.uint8)}

# for debugging
# def save_timestamp_images(timestamp_array, num_timestamps, timestamp_size, video_file_path):
#   video_file_name, _ = path.splitext(path.basename(video_file_path))
#   image_dir_path = './timestamp_images/' + video_file_name
#
#   if not path.exists(image_dir_path):
#     os.makedirs(image_dir_path)
#
#   for i in range(num_timestamps):
#     image_file_name = '{}_Timestamp_{:07d}.jpg'.format(video_file_name, i)
#     image_file_path = path.join(image_dir_path, image_file_name)
#     image = Image.fromarray(timestamp_array[timestamp_size*i:timestamp_size*(i+1)])
#     image.save(image_file_path, 'JPEG')

# for debugging
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

# for debugging
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


def _binarize_timestamps(timestamp_array):
  grayscale_timestamp_array = np.average(timestamp_array, axis=2)
  above_threshold_indices = grayscale_timestamp_array >= 128
  binary_timestamp_array = np.zeros(grayscale_timestamp_array.shape, dtype=np.uint8)
  binary_timestamp_array[above_threshold_indices] = 255

  return binary_timestamp_array


# initial naive implementation
def numerize_timestamps(timestamp_array, timestamp_size):
  binary_timestamp_array = _binarize_timestamps(timestamp_array)
  num_timestamps = int(timestamp_array.shape[0] / timestamp_size)
  numeral_timestamps = np.ndarray((num_timestamps,), dtype=np.uint32)

  # TODO: Replace triple for loop with numpy broadcasts
  for i in range(num_timestamps):
    timestamp_image = binary_timestamp_array[timestamp_size*i:timestamp_size*(i+1)]
    # TODO: replace string concatenation method with integer arithmetic method
    timestamp_string = ''
    for j in range(int(timestamp_image.shape[1] / timestamp_size)):
      digit_image = timestamp_image[:, timestamp_size*j:timestamp_size*(j+1)]

      digit_string = None

      for numeral_string, numeral_mask in numeral_mask_map.items():
        if np.all(np.equal(digit_image, numeral_mask)):
          digit_string = numeral_string
          break

      if digit_string is None:
        digit_string = '_'

      timestamp_string += digit_string

    # remove trailing blanks. they are expected given the variable widths of timestamp overlays
    timestamp_string = timestamp_string.rstrip('_')

    # if a frame's entire timestamp could not be read,
    # let downstream algorithms know with an empty string
    if timestamp_string.count('_') == 0:
      numeral_timestamps[i] = timestamp_string
    else:
      numeral_timestamps[i] = ''

  return numeral_timestamps