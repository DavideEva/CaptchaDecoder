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

  res_images = []
  rects = []
  for x, y in tuples:
    rect = (x, y, shape_w, shape_h)
    res_image = get_rect(image, rect)
    res_images.append(res_image.copy())
    rects.append(rect)

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
    'shape'                : (36, 36),
    'stride'               : (5, 5),
    'padding'              : (0, 0),
    'shape_prediction'     : (10, 10),
    'stride_prediction'    : (5, 5),
    'padding_prediction'   : (3, 3),
    'fill_image': True,
    **kwargs
  }

  def get_images(image_):
    return image_split(image_,
                       add_extra_end_crop = options_dict['fill_image'],
                       stride=options_dict['stride'],
                       shape=options_dict['shape'],
                       padding=options_dict['padding'],
                       )[0]

  def get_class(image_):
    sub_images, rects = image_split(image_,
                                    add_extra_end_crop=options_dict['fill_image'],
                                    shape=options_dict['shape_prediction'],
                                    stride=options_dict['stride_prediction'],
                                    padding=options_dict['padding_prediction'])

    output_shape = (image_.shape -
                    np.array(options_dict['padding_prediction']) * 2 -
                    np.array(options_dict['shape_prediction'])) / np.array(options_dict['stride_prediction'])
    output_shape += [1, 1]
    output_shape = np.vectorize(ceil)(output_shape)
    output = np.full(output_shape.astype(np.uint8)[:2], 0)
    # print(output_shape, output)
    for sub_image, rect in zip(sub_images, rects):
      x = int((rect[0] - options_dict['padding_prediction'][0]) / options_dict['stride_prediction'][0])
      y = int((rect[1] - options_dict['padding_prediction'][1]) / options_dict['stride_prediction'][1])
      rect_new = check_rect(
        ((rect[0] - options_dict['padding_prediction'][0]),                                 # x
         (rect[1] - options_dict['padding_prediction'][1]),                                 # y
         options_dict['shape_prediction'][0] + options_dict['padding_prediction'][0] * 2,   # w
         options_dict['shape_prediction'][1] + options_dict['padding_prediction'][1] * 2),  # h
        image_.shape)
      # print(rect, rect_new)
      image_to_evaluate = get_rect(image_, rect_new)
      output[y, x] = classify_fun(image_to_evaluate)
    return output

  return split_and_classify_image(image_x, image_letters, get_images, get_class)


def split_and_classify_image(image_x, image_letters, split_fun, classify_fun):
  sub_images = split_fun(image_letters)
  X = split_fun(image_x)
  image_classes = [classify_fun(sub_image) for sub_image in sub_images]
  return X, image_classes
