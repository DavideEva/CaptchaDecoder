import cv2
import numpy as np
from matplotlib import pyplot as plt
from time import time

from MainUtils import *
from utils.LettersImage import ColorMap, LetterImage
from utils.Rectangles import get_rect, rect_intersection, rect_intersection_percent
from utils.TemplateMatching import find_letter_position
from utils.vr_utilities import get_file_paths
from utils.ImageLetterPreparation import image_preparation, LabelBinarizer, mix_multilayer_image


def main():
  files = get_file_paths(folder_path='samples/letters/train', search_pattern='*.npy')
  letters_list = np.unique(list(map(lambda x: list(get_title_from_path(x)), files)))
  letters_list = np.append(letters_list, '_')
  lb = LabelBinarizer(letters_list)
  test_index = 112
  images_filename = [x.replace('letters', 'samples').replace('.npy', '.png') for x in files]
  image_letters = np.load(files[test_index])
  image = load_image(images_filename[test_index])
  balance = np.ones((lb.get_size(),))
  balance[lb.index('_')] = 3e-1

  stride = 5

  def image_letter_to_value(image_, balancer=balance):
    if image_.shape != (20,20):
      print(image_.shape)
    out = np.full((4,4), 0)
    for j, y in enumerate(range(0, 20, stride)):
      for i, x in enumerate(range(0, 20, stride)):
        out[j, i] = np.argmax(np.sum(np.sum(lb.convert_to_binary(get_rect(image_, (x, y, stride, stride))), axis=0), axis=0) * balancer)
    return out

  # print(lb.get_label_from_binary(np.sum(np.sum(lb.convert_binary(image), axis=0), axis=0)))
  options = {
    'shape'  : (20, 20),
    'stride' : (stride, stride),
    'padding': 8,
    'borders': 8,
  }
  start = time()
  x, y, rects = image_preparation(image, image_letters, classify_fun=image_letter_to_value, get_rects=True, **options)
  print(np.shape(x), np.shape(y))
  # x, y = np.reshape(x, (-1, *np.shape(x)[2:])), np.reshape(y, (-1, *np.shape(y)[2:]))
  end = time()
  total_time = end - start
  print(', time for image ', total_time)

  options['padding'] = (0, 0)
  # images_letters_originals, _ = image_split(il, add_extra_end_crop=True, **options)

  y_binary = lb.index_to_binary(y)
  rects = [(x//stride, y//stride, 4, 4) for (x, y, _, _) in rects]
  output_multilayer = mix_multilayer_image((image.shape[0]//stride, image.shape[1]//stride, 20), y_binary, rects)
  o = lb.get_label_from_binary(output_multilayer)
  output = np.array(lb.get_label_from_binary(output_multilayer))

  # output[2:5, 8:10] = '5' # add some noise for test

  cm = ColorMap(output)
  plt.imshow(cm.get_rgb_image())
  plt.legend(handles=cm.get_legend_elements(), loc='right')
  plt.show()

  # now analyze the image

  expanded_image = output.repeat(5, axis=0).T.repeat(5, axis=0).T

  li = LetterImage(image, expanded_image)
  print(f"real captcha: '{get_title_from_path(images_filename[test_index])}'", f"predicted captcha: '{li.get_title()}'")


if __name__ == '__main__':
  main()
