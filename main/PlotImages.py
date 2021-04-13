import os
from multiprocessing import Pool, cpu_count

import cv2
from matplotlib.lines import Line2D

from utils.LettersImage import ColorMap
from utils.TemplateMatching import parse_known_image, image_to_letters_image
from matplotlib import pyplot as plt
import requests
import numpy as np
import pandas as pd
from utils.vr_utilities import get_file_paths


def load_image(image_path, load_gray=True):
  if load_gray:
    return cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
  else:
    return cv2.cvtColor(cv2.imread(image_path, cv2.IMREAD_GRAYSCALE), cv2.COLOR_GRAY2BGR)


def from_png_to_npy(file_path):
  return file_path.replace('.png', '.npy')


def get_title_from_path(image_path):
  if image_path.endswith('.npy'):
    return image_path.replace(".npy", "").split("/")[-1]
  return image_path.replace(".png", "").split("/")[-1]


def convert_image_and_save(image_file_name, new_file_name, force_update=False):
  if not os.path.exists(new_file_name) or force_update:
    image = load_image(image_file_name)
    title = get_title_from_path(image_file_name)
    try:
      results = parse_known_image(image, title)
      output = image_to_letters_image(image, results)
      np.save(new_file_name, output)
    except Exception as error:
        print('Caught this error: ' + repr(error))


def convert_list_file(file_list):
  for file_n, n_file_n in file_list:
    print('[', os.getpid(), '] ', file_n, '->', n_file_n)
    convert_image_and_save(file_n, n_file_n)


def letter_to_binary(letter):
  return 255 if letter == '_' else 0


def main(sources):
  data_img = np.array(get_file_paths(sources[0], '*.png'))
  labels = list(map(lambda x: get_title_from_path(x), data_img))
  df = pd.DataFrame(data_img, index=labels, columns=[sources[0]])
  for source in sources[1:]:
    files_png = np.array(get_file_paths(source, '*.png'))
    files_npy = np.array(get_file_paths(source, '*.npy'))
    if len(files_png) > 0:
      labels = list(map(lambda x: get_title_from_path(x), files_png))
      df_tmp = pd.DataFrame(files_png, index=labels, columns=[source])
      df = pd.merge(df, df_tmp, left_index=True, right_index=True)
    elif len(files_npy) > 0:
      labels = list(map(lambda x: get_title_from_path(x), files_npy))
      df_tmp = pd.DataFrame(files_npy, index=labels, columns=[source])
      df = pd.merge(df, df_tmp, left_index=True, right_index=True)

  for raw in df.iloc:
    n_image = len(raw)
    fig, axs = plt.subplots(n_image)
    for idx, file in enumerate(raw):
      if file.endswith('.png'):
        axs[idx].imshow(load_image(file))
      elif file.endswith('.npy'):
        l_image = np.load(file)
        c_image = ColorMap(l_image)
        axs[idx].imshow(c_image.get_rgb_image())
        keys = list(c_image.color_dict.keys() - '_')
        values = [c_image.color_dict[k] for k in keys]
        legend_elements = list(map(
          lambda d: Line2D([0], [0],
                           color=tuple(np.array(d[1])/255),
                           linewidth=5,
                           label=d[0],),
          zip(keys, values)))
        axs[idx].legend(handles=legend_elements,
                        loc="right",)

      plt.title(get_title_from_path(file))
    plt.show()


if __name__ == '__main__':
  import sys
  if len(sys.argv) > 1:
    sources = sys.argv[1:]
    main(sources)
  else:
    print("Missing args <source>* ")
