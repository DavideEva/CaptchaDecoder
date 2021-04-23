import cv2
import numpy as np
from matplotlib import pyplot as plt
from time import time

from matplotlib.lines import Line2D

from utils.LettersImage import ColorMap
from utils.Rectangles import get_rect
from utils.vr_utilities import get_file_paths
from utils.ImageLetterPreparation import image_preparation, LabelBinarizer, image_split, add_constant_border


def get_title_from_path(image_path):
  if image_path.endswith('.npy'):
    return image_path.replace(".npy", "").split("/")[-1]
  return image_path.replace(".png", "").split("/")[-1]


def load_image(image_path, load_gray=True):
  if load_gray:
    return cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
  else:
    return cv2.cvtColor(cv2.imread(image_path, cv2.IMREAD_GRAYSCALE), cv2.COLOR_GRAY2BGR)


def main():
  files = get_file_paths(folder_path='../samples/letters/train', search_pattern='*.npy')
  letters_list = np.unique(list(map(lambda x: list(get_title_from_path(x)), files)))
  letters_list = np.append(letters_list, '_')
  lb = LabelBinarizer(letters_list)
  test_size = 2
  images_filename = [x.replace('letters', 'samples').replace('.npy', '.png') for x in files]
  images_letters = [np.load(x) for x in files[:test_size]]
  images = [load_image(x) for x in images_filename]
  ez = np.ones((lb.get_size(),))
  ez[lb.index('_')] = 3e-1

  def image_letter_to_value(image_, balancer=ez):
    if image_.shape != (20,20):
      print(image_.shape)
    out = np.full((4,4), 0)
    for j, y in enumerate(range(0, 20, 5)):
      for i, x in enumerate(range(0, 20, 5)):
        out[j, i] = np.argmax(np.sum(np.sum(lb.convert_binary(get_rect(image_, (x, y, 5, 5))), axis=0), axis=0) * balancer)
    return out

  # print(lb.get_label_from_binary(np.sum(np.sum(lb.convert_binary(image), axis=0), axis=0)))
  options = {
    'shape'  : (20, 20),
    'stride' : (5, 5),
    'padding': 8,
    'borders': 8,
  }
  start = time()
  X_y = []
  real_images_letter = []
  for image, image_letters in zip(images, images_letters):
    X_y.append(image_preparation(image, image_letters, classify_fun=image_letter_to_value, **options))
    real_images_letter.append(image_letters)
  x, y = zip(*X_y)
  x, y = np.reshape(x, (-1, *np.shape(x)[2:])), np.reshape(y, (-1, *np.shape(y)[2:]))
  end = time()
  total_time = end - start
  print('total time ', total_time, ', time for image ', total_time / test_size)
  print(np.shape(x), np.shape(y))

  il = real_images_letter[0]
  options['padding'] = (0, 0)
  images_letters_originals, _ = image_split(il, add_extra_end_crop=True, **options)

  for x_, y_ in zip(x, y):
    fig, axs = plt.subplots(3, figsize=(7, 7))
    # print(x_, y_, lb.get_label(y_))
    # plt.title(lb.get_label(y_))
    cv2.rectangle(x_, (8, 8), (28, 28), 125)
    base_line_x, base_line_y = 8, 8
    for line in range(4):
      cv2.line(x_, (base_line_y + line * 5, base_line_x), (base_line_y + line * 5, base_line_x + 20), 125)
      cv2.line(x_, (base_line_y, base_line_x + line * 5), (base_line_y + 20, base_line_x + line * 5), 125)
    axs[0].imshow(x_)
    cm = ColorMap(np.vectorize(lambda x: lb.get_label(x))(y_))
    axs[1].imshow(cm.get_rgb_image())
    keys = list(cm.color_dict.keys() - '_')
    values = [cm.color_dict[k] for k in keys]
    legend_elements = list(map(
      lambda d: Line2D([0], [0],
                       color=tuple(np.array(d[1]) / 255),
                       linewidth=5,
                       label=d[0], ),
      zip(keys, values)))
    axs[1].legend(handles=legend_elements,
                     loc="right", )
    plt.show()



if __name__ == '__main__':
  main()
