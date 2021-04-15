import cv2
import numpy as np
from math import ceil

from .Rectangles import get_rect
from .Rectangles import check_rect


class LabelBinarizer:

  def __init__(self, labels):
    self.labels = list(sorted(labels))
    self._size = len(labels)

  def convert_binary(self, label):
    def _binary(label_):
      val = np.zeros((self._size,))
      val[self.index(label_)] = 1
      return val

    if len(np.shape(label)) > 0:
      return [self.convert_binary(item) for item in label]
    return _binary(label)

  def index(self, label):
    if len(np.shape(label)) > 0:
      return [self.index(item) for item in label]
    return self.labels.index(label)

  def get_label(self, index):
    if len(np.shape(index)) > 0:
      return [self.get_label(item) for item in index]
    return self.labels[index]

  def get_label_from_binary(self, binary):
    if np.shape(binary) == (self._size,):
      return self.get_label(np.argmax(binary))
    return [self.get_label_from_binary(item) for item in binary]

  def get_size(self):
    return self._size


def image_split(image, add_extra_end_crop=True, **kwargs):
  """
    return an array of tuple crop_images, rects
    where rect contain the rectangle information relative to the cropped image

    :shape: tuple w, h of split shape
    :steps: tuple w, h of step between every split

    if add_extra_end_crop is True create also the last rect matching end of rows and/or columns

    example: in a image(10, 9) with shape(7,7) and steps(2, 2) creates:
      (0,0,7,7)
      (2,0,7,7)
      (0,2,7,7)
      (2,2,7,7)
      and (3,2,7,7) that not respect the steps(2, 2) but includes all the points in image
  """
  options = {
    "shape"  : (36, 36),
    "stride" : (1, 1),
    "padding": (0, 0),
    **kwargs
  }

  image_h, image_w = image.shape[:2]
  shape_w, shape_h = options['shape']
  step_x, step_y = options['stride']
  padding_x, padding_y = options['padding']
  # print(options)
  extra_h = []
  extra_w = []
  if add_extra_end_crop:
    extra_h = [(image_h - padding_y) - shape_h, ]
    extra_w = [(image_w - padding_x) - shape_w, ]

  output_rect = []
  for y in list(range(padding_y, (image_h - padding_y) - shape_h + 1, step_y)) + extra_h:
    for x in list(range(padding_x, (image_w - padding_x) - shape_w + 1, step_x)) + extra_w:
      output_rect.append((x, y))

  tuples = np.unique(output_rect, axis=0)

  rects, res_images = zip(*[((x, y, shape_w, shape_h), get_rect(image, (x, y, shape_w, shape_h))) for x, y in tuples])
  # res_images = []
  # rects = []
  # for x, y in tuples:
  #   rect = (x, y, shape_w, shape_h)
  #   res_image = get_rect(image, rect)
  #   res_images.append(res_image.copy())
  #   rects.append(rect)

  return np.array(res_images), np.array(rects)


def image_preparation(image_x, image_letters, classify_fun, **kwargs):
  """
    Layout:
      Train image shape : 36x36 pixel
      Train output : 3 x 3 pixel, corresponding to the 30 x 30 centered pixel

    Every output cell corresponds to an area of 15x15 that's overfit the 12x12 area to improve
    the ability of understand the behavior.
  """

  options_dict = {
    'shape'            : (36, 36),
    'stride'           : (5, 5),
    'padding'          : (8, 8),
    'borders'          : (8, 8, 8, 8),
    'background_value' : 255,
    'background_letter': '_',
    'fill_image'       : True,
    **kwargs
  }
  output_size = (np.array(options_dict['shape']) - np.array(options_dict['padding']) * 2) / np.array(
    options_dict['stride'])
  output_size = output_size.astype(np.uint8)

  # add border
  top, bottom, left, right = options_dict['borders']
  image_with_border = cv2.copyMakeBorder(image_x, top, bottom, left, right,
                                         cv2.BORDER_CONSTANT, None, options_dict['background_value'])

  # split the image
  images_x, rects_x = image_split(image_with_border,
                                  add_extra_end_crop=True,
                                  shape=options_dict['shape'],
                                  stride=options_dict['stride'],
                                  padding=options_dict['padding'])

  stride_x, stride_y = options_dict['stride']
  y_r, x_r = image_letters.shape[:2]
  reduced_image_letter = np.full((int(y_r / stride_y), int(x_r / stride_x),), options_dict['background_value'])

  for y_index, y in enumerate(range(0, y_r, stride_y)):
    for x_index, x in enumerate(range(0, x_r, stride_x)):
      reduced_image_letter[y_index, x_index] = classify_fun(get_rect(image_letters, (x, y, stride_x, stride_y)))

  # px, py = options_dict['padding']
  y_out = [
    get_rect(reduced_image_letter, (int(x / stride_x), int(y / stride_y), output_size[0], output_size[1]))
    for x, y, _, _ in rects_x
  ]

  return np.array(images_x), np.array(y_out)


def split_and_classify_image(image_x, image_letters, split_fun, classify_fun):
  sub_images = split_fun(image_letters)
  X = split_fun(image_x)
  image_classes = [classify_fun(sub_image) for sub_image in sub_images]
  return X, image_classes
