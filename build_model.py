import border
import color
import cv2
import numpy as np
import os
import symmetry
import sys

from sklearn import cross_validation
from sklearn.ensemble import RandomForestClassifier


def build_random_forest_classfier():
  X, y = build_feature_vector()
  X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size = 0.2,
                                     random_state = 0)

  #print len(X_test)
  clf = RandomForestClassifier(n_estimators = 10)
  clf = clf.fit(X_train, y_train)
  #print clf.score(X_test, y_test)

  scores = cross_validation.cross_val_score(clf, X, y, cv = 6)
  print scores.mean()

def build_feature_vector():
  img_list = import_image_paths()
  feature_vector = []
  target_vector = []

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

    if 'benign' in img_filename:
      target = 0
    if 'malignant' in img_filename:
      target = 1
    target_vector.append(target)

  #print target_vector
  return (np.array(feature_vector), np.array(target_vector))

def import_image_paths():
  path_list = ["img/benign", "img/malignant"]
  img_path_list = []

  for p in path_list:
    counter = 0
    listing = os.listdir(p)
    for img in listing:
      img_path_list.append(p + '/' + img)
      counter += 1

      # if counter == 25:
      #   break

  return img_path_list


if __name__ == '__main__':
  build_random_forest_classfier()

