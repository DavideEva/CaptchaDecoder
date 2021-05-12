import random

from matplotlib import pyplot as plt

from model.Letter import Letter
from model.TemplateScore import TemplateScore
from .Rectangles import get_rect, add_rect, set_rect, y_intersection, x_intersection, rect_intersection_percent
from .ImageOperations import filter_threshold, erode_option, generate_letter_image, templates_generator, threshold
import cv2
import math
import numpy as np
import itertools


def find_best_template(image, templates, options=None, invalid_positions=None):
  threshold_boundary = 50

  inverted_image = 255 - image.copy()
  best_template = TemplateScore(score=0, crop_image=image, template_rect=(0, 0, 0, 0), overlap_score=0)
  method = cv2.TM_CCOEFF_NORMED

  if invalid_positions is None:
    invalid_positions = []

  if options is None:
    def none_gen():
      while True:
        yield None

    options = none_gen()

  for template in templates:
    inverted_template = 255 - template
    h, w = template.shape[:2]
    # Apply template Matching
    res = cv2.matchTemplate(inverted_image, inverted_template, method)

    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

    top_left = max_loc
    rect_top_left = (top_left[0], top_left[1], w, h)
    bottom_right = (top_left[0] + w, top_left[1] + h)

    cut_image = image.copy()

    eroded_template = cv2.erode(template, np.ones((3, 3)))
    points_in_common = np.count_nonzero(
      (get_rect(cut_image, rect_top_left) + eroded_template) <= threshold_boundary)

    points_in_common = points_in_common * 10 + 1  # corrector
    filter_threshold(cut_image, rect_top_left, eroded_template, threshold_boundary, background_value=255)

    deficit = 1
    for rect in invalid_positions:
      deficit *= 1 - rect_intersection_percent(rect, rect_top_left)

    current_score = max_val * math.log(points_in_common) * deficit

    if current_score > best_template.score:
      # improvement
      best_rect = rect_top_left
      base_img = np.full(image.shape, 255)
      letter_cut = get_rect(image, best_rect).copy()

      set_rect(base_img, best_rect, letter_cut)
      add_rect(base_img, best_rect, eroded_template)
      base_img[base_img <= 50] = 0
      base_img[base_img > 50] = 255

      # plt.subplot(311).imshow(base_img)
      # plt.subplot(312).imshow(eroded_template)
      # plt.subplot(313).imshow(cut_image)
      # plt.show()

      best_template = TemplateScore(score=current_score, crop_image=cut_image.copy(), template_rect=best_rect,
                                    overlap_score=points_in_common, option_used=None, image_valid_points=base_img)

  return best_template


def parse_known_image(image, title):
  results = []
  new_title = ''
  letters = list(title)
  all_permutations = list(itertools.permutations(letters))
  random.shuffle(all_permutations)
  for new_title, _ in zip(all_permutations, range(15)):
    image_copy = image.copy()
    results = []
    letter_images = []
    for letter in new_title:
      letter_images.append(
        (letter,
         templates_generator(generate_letter_image(letter), optional_transformations=[erode_option, ])
         )
      )
    for letter, templates in letter_images:
      invalid_rects = list(map(lambda x: x[1].rect, results))
      best_template = find_best_template(image_copy, templates, invalid_positions=invalid_rects)
      best_letter = Letter(
        best_template.rect,
        letter,
        best_template.overlap_score,
        letter_image_position=best_template.image_valid_points)
      results.append((best_template.score, best_letter))
      image_copy = best_template.crop_image
    results = sorted(results, key=lambda x: x[1].rect[0])
    new_title = ''.join(list(map(lambda x: x[1].letter, results)))
    if new_title == title:
      # check overlapping
      any_fail = False
      for idx, res1 in enumerate(results[:-1]):
        for res2 in results[idx + 1:]:
          rect1, rect2 = res1[1].rect, res2[1].rect
          y_inter, x_inter = y_intersection(rect1, rect2), x_intersection(rect1, rect2)
          if y_inter > 0.3 and x_inter > 0.3:
            any_fail = True
      if not any_fail:
        break
  if new_title != title:
    raise Exception(f"Can't find correct sequence for {title}.png image")
  return np.array(results)[:, 1]


def image_to_letters_image(image, templates, background='_'):
  output = np.full(image.shape, background)
  for tmp in templates:
    output[tmp.letter_image_position == 0] = tmp.letter
  return output


def find_letter_position(image, letter, background=255):
  """

  :param image: a binary image
  :param letter: the letter to be searched
  :param background: background value
  :return: the position of the letter and a score
  """
  options = {
    'rotation_range' : 8,
    'scale_min_range': 95,
    'scale_max_range': 110,
  }
  templates = templates_generator(generate_letter_image(letter), optional_transformations=[erode_option, ], **options)
  template = find_best_template(image, templates)
  return template.rect, template.score
