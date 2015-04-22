import border
import color
import cv2
import numpy as np
import os
import symmetry
import sys

def build_feature_vector():
  img_list = import_image_paths()
  feature_vector = []

  for img_filename in img_list:
    img = cv2.imread(img_filename)
    gray = cv2.imread(img_filename,0)
    # img = cv2.resize(img, (300, 300)) 
    # gray = cv2.resize(gray, (300, 300))

    x, y, _ = img.shape

    if x > 1000 or y > 1000:
      continue # hack to avoid large images

    print img_filename
    symmetry_ssim = symmetry.get_symmetry_ssim(img)
    symmetry_mse = symmetry.get_symmetry_mse(img)
    border_gof = border.get_gof(gray)
    #border_gof = 0
    color_contrast = color.color_contrast(img)

    current_vec = [symmetry_ssim, symmetry_mse, border_gof, color_contrast]
    feature_vector.append(current_vec)

  return np.array(feature_vector).shape

def import_image_paths():
  path_list = ["img/benign", "img/malignant"]
  img_path_list = []

  for p in path_list:
    listing = os.listdir(p)
    for img in listing:
      img_path_list.append(p + '/' + img)

  return img_path_list


if __name__ == '__main__':
  print build_feature_vector()

