import cv2
import numpy as np
from math import ceil

from utils.Rectangles import get_rect
from utils.Rectangles import check_rect


class LabelBinarizer:

  def __init__(self, labels):
    self.labels = list(sorted(labels))
    self._size = len(labels)

  def convert_to_binary(self, label):
    def _binary(label_):
      val = np.zeros((self._size,))
      val[self.index(label_)] = 1
      return val

    if len(np.shape(label)) > 0:
      return [self.convert_to_binary(item) for item in label]
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

  def index_to_binary(self, index):
    if len(np.shape(index)) > 0:
      return [self.index_to_binary(i) for i in index]
    return self.convert_to_binary(self.get_label(index))

  def get_size(self):
    return self._size


def image_split_coordinate(shape, shape_rect, stride, add_last_rect=True):
  """
  :param shape:
  :param shape_rect:
  :param stride:
  :param add_last_rect:
  :return: a list of rectangle of type (x, y, w, h) represent rectangles in an image of shape=shape
  """
  image_w, image_h = shape
  rect_w, rect_h = shape_rect
  stride_x, stride_y = stride
  extra_w, extra_h = [], []
  if add_last_rect:
    extra_h = [image_h - rect_h, ]
    extra_w = [image_w - rect_w, ]

  output_rect = []
  for y in list(range(0, image_h - rect_h + 1, stride_y)) + extra_h:
    for x in list(range(0, image_w - rect_w + 1, stride_x)) + extra_w:
      output_rect.append((x, y))

  tuples = np.unique(output_rect, axis=0)

  return [(x, y, rect_w, rect_h) for (x, y) in tuples]


def image_split(image, add_extra_end_crop=True, **kwargs):
  """
    return an array of tuple crop_images, rects
    where rect contain the rectangle information relative to the cropped image

    :shape: tuple w, h of split shape
    :stride: tuple w, h
    :padding: value

    if add_extra_end_crop is True create also the last rect matching end of rows and/or columns

    example: in a image(10, 9) with shape(7,7) and steps(2, 2) creates:
      (0,0,7,7)
      (2,0,7,7)
      (0,2,7,7)
      (2,2,7,7)
      and (3,2,7,7) that not respect the steps(2, 2) but includes all the points in image
  """
  kwargs = {
    "shape"  : (36, 36),
    "stride" : (1, 1),
    **kwargs
  }

  tuples = image_split_coordinate(image.shape[:2], kwargs['shape'], kwargs['stride'], add_last_rect=add_extra_end_crop)

  rects, res_images = zip(*[((x, y, w, h), get_rect(image, (x, y, w, h))) for x, y, w, h in tuples])

  return np.array(res_images), np.array(rects)


def add_constant_border(image, border_value, background_value=255):
  return cv2.copyMakeBorder(image,
                            border_value,  # top
                            border_value,  # bottom
                            border_value,  # left
                            border_value,  # right
                            cv2.BORDER_REFLECT, None)


def mix_multilayer_image(image_shape, values, rects):
  assert len(image_shape) == 3
  output_image = np.full(image_shape, .0)
  for value, rect in zip(values, rects):
    x, y, w, h = rect
    sub_image = np.full((h, w, image_shape[2]), value)
    output_image[y:y + h, x:x + w] += sub_image
  return output_image


def image_preparation(image, image_letters, classify_fun, get_rects=False, **kwargs):
  kwargs = {
    'shape'            : (20, 20),
    'stride'           : (5, 5),
    'padding'          : 8,
    'background_value' : 255,
    'background_letter': '_',
    **kwargs
  }

  rects = image_split_coordinate(image.shape[2::-1], kwargs['shape'], kwargs['stride'])

  # add border
  image_with_border = add_constant_border(image,  # image
                                          kwargs['padding'],  # border size
                                          background_value=kwargs['background_value'])  # border value

  # in image_with_border the image start at point (pad, pad) and end at point (h-pad, w-pad)

  X_images = []
  y = []
  pad = kwargs['padding']
  for (x_, y_, w, h) in rects:
    image_w = w + pad*2
    image_h = h + pad*2
    X_images.append(get_rect(image_with_border, (x_, y_, image_w, image_h)))
    y.append(classify_fun(get_rect(image_letters, (x_, y_, w, h))))

  if get_rects:
    return np.array(X_images), np.array(y), rects
  return np.array(X_images), np.array(y)


def split_and_classify_image(image_x, image_letters, split_fun, classify_fun):
  sub_images = split_fun(image_letters)
  X = split_fun(image_x)
  image_classes = [classify_fun(sub_image) for sub_image in sub_images]
  return X, image_classes
