import cv2
import numpy as np
import math

from matplotlib.lines import Line2D

from utils.Rectangles import rect_intersection_percent
from utils.TemplateMatching import find_letter_position

colors = [
    [230, 25, 75],    [60, 180, 75],    [255, 225, 25], [0, 130, 200],
    [245, 130, 48],   [70, 240, 240],   [240, 50, 230], [250, 190, 212],
    [0, 128, 128],    [220, 190, 255],  [170, 110, 40], [255, 250, 200],
    [128, 0, 0],      [170, 255, 195],  [0, 0, 128],    [128, 128, 128],
    [50, 50, 50],     [100, 100, 100],  [30, 30, 30],   [10, 10, 10]
  ]


def rgb_sim(color1, color2):
  r1, g1, b1 = color1
  r2, g2, b2 = color2
  n = math.sqrt((r2 - r1) ** 2 + (g2 - g1) ** 2 + (b2 - b1) ** 2)
  d = math.sqrt(255 ** 2 + 255 ** 2 + 255 ** 2)
  return 1 - n / d


class ColorMap:
  """
    Utility class to plot ImageLetter in a 2D space
  """
  def __init__(self, letters_image, background='_'):
    self.letters_image = letters_image
    self.color_dict = {}
    distinct_letters = np.unique(letters_image)
    for idx, letter in enumerate(distinct_letters):
      self.color_dict[letter] = colors[idx]
    self.color_dict[background] = [255, 255, 255]

  def convert_letter(self, letter):
    return self.color_dict[letter]

  def get_rgb_image(self):
    h, w = self.letters_image.shape[:2]
    result = np.full((h, w, 3), 0)
    for j, row in enumerate(self.letters_image):
      for i, cell in enumerate(row):
        result[j, i] = self.convert_letter(cell)
    return result

  def get_legend_elements(self, background='_'):
    keys = list(self.color_dict.keys() - background)
    values = [self.color_dict[k] for k in keys]
    return list(map(
      lambda d: Line2D([0], [0],
                       color=tuple(np.array(d[1]) / 255),
                       linewidth=5,
                       label=d[0], ),
      zip(keys, values)))


def label_to_binary(label, background='_'):
  return 0 if label == background else 255
