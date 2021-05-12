import cv2


def get_title_from_path(image_path):
  if image_path.endswith('.npy'):
    return image_path.replace(".npy", "").split("/")[-1]
  return image_path.replace(".png", "").split("/")[-1]


def load_image(image_path, load_gray=True):
  if load_gray:
    return cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
  else:
    return cv2.cvtColor(cv2.imread(image_path, cv2.IMREAD_GRAYSCALE), cv2.COLOR_GRAY2BGR)
