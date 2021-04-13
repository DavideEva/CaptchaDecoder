# Operations on rectangle of type (x, y, w, h)

def weight(a):
  return a[2]


def height(a):
  return a[3]


def area(a):
  return a[2] * a[3]


def get_rect(image, rect):
  x, y, w, h = rect
  return image[y:y + h, x:x + w]


def add_rect(image, rect, image_2, inplace=True):
  x, y, w, h = rect
  assert (h, w) == image_2.shape[:2]
  if inplace:
    image[y:y+h, x:x+w] += image_2
  else:
    new_image = image.copy()
    new_image[y:y + h, x:x + w] += image_2
    return new_image


def set_rect(image, rect, image_2, inplace=True):
  x, y, w, h = rect
  assert (h, w) == image_2.shape[:2]
  if inplace:
    image[y:y+h, x:x+w] = image_2
  else:
    new_image = image.copy()
    new_image[y:y + h, x:x + w] = image_2
    return new_image


def rect_union(a, b):
  x = min(a[0], b[0])
  y = min(a[1], b[1])
  w = max(a[0] + a[2], b[0] + b[2]) - x
  h = max(a[1] + a[3], b[1] + b[3]) - y
  return x, y, w, h


def rect_intersection(a, b):
  x = max(a[0], b[0])
  y = max(a[1], b[1])
  w = min(a[0] + a[2], b[0] + b[2]) - x
  h = min(a[1] + a[3], b[1] + b[3]) - y
  if w < 0 or h < 0:
    return 0, 0, 0, 0
  return x, y, w, h


def rect_center(a):
  (x, y, w, h) = a
  return round(x + (w / 2)), round(y + (h / 2))


def rect_intersection_percent(a, b):
  """
    percent of the area of a, overlapped to b
    result = area(a intersection b)/area(a) + area(a intersection b)/area(b)

    result range [0, 1]

      - 1 the rectangles are equals
      - 0 no overlapping
  """
  intersection = rect_intersection(a, b)
  intersection_area = intersection[2] * intersection[3]
  area_a = a[2] * a[3]
  area_b = b[2] * b[3]
  return (intersection_area / area_a + intersection_area / area_b) / 2


def y_intersection(a, b):
  """
    calculate the intersection only onver the y-axes

    2 h-stacked rectangle will have a greater score than 2 w-stacked rectangle
  """
  if a[0] > b[0]:
    c = b
    b = a
    a = c
  if a[0] + weight(a) < b[0]:
    return 0
  if a[0] + weight(a) > b[0] + weight(b):
    overlap = weight(b)
  else:
    overlap = a[0] + weight(a) - b[0]
  return (overlap / weight(a) + overlap / weight(b)) / 2


def x_intersection(a, b):
  """
    calculate the intersection only over the y-axes

    2 w-stacked rectangle will have a greater score than 2 h-stacked rectangle
  """
  if a[1] > b[1]:
    c = b
    b = a
    a = c
  if a[1] + height(a) < b[1]:
    return 1
  if a[1] + height(a) > b[1] + height(b):
    overlap = height(b)
  else:
    overlap = a[1] + height(a) - b[1]
  return (overlap / height(a) + overlap / height(b)) / 2


def check_rect(rect, image_shape):
  """
  :param rect: the rect to put into the image
  :param image_shape: the image shape
  :return: a rectangle that doesn't exit from the image
  """
  x, y = max(rect[0], 0), max(rect[1], 0)
  h, w = image_shape[:2]
  assert rect[2] <= w and rect[3] <= h
  w = min(rect[2], w - x)
  h = min(rect[3], h - y)
  return x, y, w, h
