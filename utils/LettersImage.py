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
  ]


def rgb_sim(color1, color2):
  r1, g1, b1 = color1
  r2, g2, b2 = color2
  n = math.sqrt((r2 - r1) ** 2 + (g2 - g1) ** 2 + (b2 - b1) ** 2)
  d = math.sqrt(255 ** 2 + 255 ** 2 + 255 ** 2)
  return 1 - n / d


class ColorMap:
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


class LetterImage:
  def __init__(self, image, letter_image):
    assert np.shape(image) == np.shape(letter_image)
    self.letter_image = letter_image
    self.image = image

  def get_title(self):
    positions = []
    for letter in np.unique(self.letter_image):
      if letter != '_':
        im = self.image.copy()
        im[self.letter_image != letter] = 255
        last_valid = True
        im_copy = im.copy()
        while last_valid:
          rect, score = find_letter_position(im_copy, letter)
          x, y, w, h = rect
          candidates = list(filter(lambda t: rect_intersection_percent(rect, t[0]) > 0.6, positions))
          if len(candidates) == 1 and score > 2:
            candidates = sorted(candidates, key=lambda t: t[2])
            if candidates[0][2] > score:
              last_valid = False
            else:
              positions.remove(candidates[0])
              cv2.rectangle(im, (x, y), (x + w, y + h), 125)
              im_copy[y:y + h, x:x + h] = 255
              positions.append((rect, letter, score))
          elif len(candidates) > 1:
            print("more than one candidate!")
          elif score > 2:
            cv2.rectangle(im, (x, y), (x + w, y + h), 125)
            im_copy[y:y + h, x:x + h] = 255
            positions.append((rect, letter, score))
          else:
            last_valid = False
    return ''.join(list(map(lambda x: x[1], sorted(positions, key=lambda x: x[0][0]))))

