from typing import Any, Callable

import cv2
import random
import numpy as np
from PIL import ImageFont, ImageDraw
from PIL import Image
from utils.Rectangles import area


def threshold(img):
  return cv2.adaptiveThreshold(img.copy(), 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 4)


def rotation(img, angle):
  angle = int(random.uniform(-angle, angle))
  h, w = img.shape[:2]
  rm = cv2.getRotationMatrix2D((int(w / 2), int(h / 2)), angle, 1)
  img = cv2.bitwise_not(img)
  img = cv2.warpAffine(img, rm, (w, h))
  img = cv2.bitwise_not(img)
  return img


def fill(img, h, w):
  img = cv2.resize(img, (h, w), cv2.INTER_CUBIC)
  return img


def vertical_shift(img, ratio=0.0):
  if ratio > 1 or ratio < 0:
    print('Value should be less than 1 and greater than 0')
    return img
  ratio = random.uniform(-ratio, ratio)
  h, w = img.shape[:2]
  to_shift = h * ratio
  if ratio > 0:
    img = img[:int(h - to_shift), :, :]
  if ratio < 0:
    img = img[int(-1 * to_shift):, :, :]
  img = fill(img, h, w)
  return img


def horizontal_shift(img, ratio=0.0):
  if ratio > 1 or ratio < 0:
    print('Value should be less than 1 and greater than 0')
    return img
  ratio = random.uniform(-ratio, ratio)
  h, w = img.shape[:2]
  to_shift = w * ratio
  if ratio > 0:
    img = img[:, :int(w - to_shift), :]
  if ratio < 0:
    img = img[:, int(-1 * to_shift):, :]
  img = fill(img, h, w)
  return img


def rescale_image(img, scale_x=1.0, scale_y=1.0):
  width = int(img.shape[1] * scale_x)
  height = int(img.shape[0] * scale_y)
  dim = (width, height)
  # resize image
  return cv2.resize(img, dim, interpolation=cv2.INTER_AREA)


def filter_threshold(image, rect_position, template, threshold_vale=50, inplace=True, background_value=255):
  new_image = image if inplace else image.copy()
  x, y, w, h = rect_position
  new_image[y:y + h, x:x + w][template <= threshold_vale] = background_value

  if not inplace:
    return new_image


def erode_option(img):
  return cv2.erode(img, np.ones((3, 3)))


def templates_generator(image, optional_transformations=None, return_options=False):
  if optional_transformations is None:
    optional_transformations = []

  def default(x):
    return x

  transform_list = [default, ]
  for e in optional_transformations:
    transform_list.append(e)

  rotation_range = 16
  scale_min_range = 80
  scale_max_range = 115
  for i in range(-rotation_range, rotation_range + 1, 2):
    rotated_img = rotation(image, i)
    for scaleX in range(scale_min_range, scale_max_range + 1, 3):
      for scaleY in range(scale_min_range, scale_max_range + 1, 3):
        for idx, transform in enumerate(transform_list):
          transformed_image = transform(rescale_image(rotated_img, scaleX / 100, scaleY / 100))
          options = (i, scaleX, scaleY, idx)
          if return_options:
            yield transformed_image, options
          else:
            yield transformed_image


def generate_letter_image(label, w=50, h=50, font_path='./resources/Arial.ttf'):
  assert len(list(label)) == 1
  img = Image.new('L', (w, h), color=255)
  fnt = ImageFont.truetype(font_path, 40)
  d = ImageDraw.Draw(img)
  d.text((0, 0), label, fill=0, font=fnt)  # Selezionare la lettera
  img_array = np.array(img)
  imgGray = 255 - img_array
  outputImage = img_array.copy()
  npaContours, npaHierarchy = cv2.findContours(imgGray.copy(),
                                               cv2.RETR_EXTERNAL,
                                               cv2.CHAIN_APPROX_SIMPLE)

  rects = list(sorted([cv2.boundingRect(x) for x in npaContours], key=lambda x: area(x), reverse=True))

  if len(rects) > 0:
    [intX, intY, intW, intH] = rects[0]
  else:
    [intX, intY, intW, intH] = [0, 0, w, h]
  intY = max(0, intY - 1)
  intX = max(0, intX - 1)
  value = 255 - imgGray[intY:intY + intH + 2, intX:intX + intW + 2]
  return value
