import cv2
import numpy as np
from matplotlib import pyplot as plt

from utils.vr_utilities import get_file_paths
from utils.ImageLetterPreparation import image_preparation, LabelBinarizer


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
  files = get_file_paths(folder_path='./samples/letters/train', search_pattern='*.npy')
  letters_list = np.unique(list(map(lambda x: list(get_title_from_path(x)), files)))
  letters_list = np.append(letters_list, '_')
  lb = LabelBinarizer(letters_list)
  image_letters = np.load('./samples/letters/train/5mfff.npy')
  image = load_image('./samples/samples/train/5mfff.png')
  ez = np.ones((lb.get_size(),))
  ez[lb.index('_')] = 7e-1

  def image_letter_to_value(image_, balancer=ez):
    return np.argmax(np.sum(np.sum(lb.convert_binary(image_), axis=0), axis=0) * balancer)

  # print(lb.get_label_from_binary(np.sum(np.sum(lb.convert_binary(image), axis=0), axis=0)))
  options = {
    'shape'             : (36, 36),
    'step'              : (5, 5),
    'shape_prediction'  : (10, 10),
    'stride_prediction' : (5, 5),
    'padding_prediction': (3, 3),
  }
  x, y = image_preparation(image, image_letters, classify_fun=image_letter_to_value, **options)
  print(np.shape(x), np.shape(y))
  for x_, y_ in zip(x, y):
    print(x_, y_, lb.get_label(y_))
    plt.subplot(211).imshow(x_)
    plt.subplot(212).imshow(y_)
    plt.show()


if __name__ == '__main__':
  main()
