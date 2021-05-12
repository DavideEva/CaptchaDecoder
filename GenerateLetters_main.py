import os
from multiprocessing import Pool

from utils.TemplateMatching import parse_known_image, image_to_letters_image
from matplotlib import pyplot as plt
import requests
import numpy as np
from utils.vr_utilities import get_file_paths
from MainUtils import *


def from_png_to_npy(file_path):
  return file_path.replace('.png', '.npy')


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


def download_font():
  url = "https://www.freebestfonts.com/download?fn=257"
  font_path = 'resources/Arial.ttf'
  r = requests.get(url, allow_redirects=True)
  with open(font_path, 'wb') as af:
    af.write(r.content)
  return font_path


def letter_to_binary(letter):
  return 255 if letter == '_' else 0


def single_sample():
  file_path = 'samples/samples/test/wgmwp.png'
  image = load_image(file_path)
  title = get_title_from_path(file_path)
  results = parse_known_image(image, title)
  output = image_to_letters_image(image, results)
  binary_output = np.vectorize(letter_to_binary)(output)
  plt.subplot(211).imshow(image)
  plt.subplot(212).imshow(binary_output)
  plt.show()


def main(source, dest):
  # fp = download_font()
  files = np.array(get_file_paths(source, '*.png'))
  new_files_name = map(lambda x: from_png_to_npy(x.replace(source, dest)), files)
  data = list(zip(files, new_files_name))
  n = 8 # cpu_count()
  with Pool(processes=n) as p:
    p.map(convert_list_file, np.array_split(data, n))


if __name__ == '__main__':
  import sys
  if len(sys.argv) == 3:
    source, dest = sys.argv[1:3]
    main(source, dest)
  else:
    print("Missing args <source> and <dest>")
