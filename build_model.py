import border
import color
import cPickle
import cv2
import numpy as np
import os
import symmetry
import sys

from sklearn import cross_validation
from sklearn.cluster import MiniBatchKMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn import svm

total_images = 
color_sum
mse_sum
ssid_sum
border_sum = 

def build_random_forest_classfier():
  X, y = build_feature_vector()
  X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size = 0.1,
                                     random_state = 0)

  #print len(X_test)
  clf = RandomForestClassifier(n_estimators = 10)
  clf = clf.fit(X, y)
  #print clf.score(X_test, y_test)

  scores = cross_validation.cross_val_score(clf, X, y, cv = 5)
  print scores.mean()

  #clf = clf.fit(X, y)
  return clf

def build_linear_svm():
  X, y = build_feature_vector()
  X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size = 0.1,
                                     random_state = 0)

  clf = svm.SVC(kernel = 'linear')
  clf.fit(X_train, y_train)

  scores = cross_validation.cross_val_score(clf, X, y, cv = 5)
  print scores.mean()

def build_poly_svm():
  X, y = build_feature_vector()
  X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size = 0.1,
                                     random_state = 0)

  clf = svm.SVC(kernel = 'poly', degree = 3)
  clf.fit(X_train, y_train)

  scores = cross_validation.cross_val_score(clf, X, y, cv = 5)
  print scores.mean()

def build_gaussian_nb():
  X, y = build_feature_vector()
  X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size = 0.1,
                                     random_state = 0)

  clf = GaussianNB()
  clf.fit(X_train, y_train)

  scores = cross_validation.cross_val_score(clf, X, y, cv = 5)
  print scores.mean()


def build_feature_vector():
  img_list = import_image_paths()
  feature_vector = []
  target_vector = []

  for img_filename in img_list:
    img = cv2.imread(img_filename)
    x, y, _ = img.shape

    if x > 1000 or y > 1000:
      continue # hack to avoid large images

    # image = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    # image = image.reshape((image.shape[0] * image.shape[1], 3))
    # clt = MiniBatchKMeans(n_clusters = 2)
    # labels = clt.fit_predict(image)
    # quant = clt.cluster_centers_.astype("uint8")[labels]
    # quant = quant.reshape((x, y, 3))
    # image = image.reshape((x, y, 3))
    # quant = cv2.cvtColor(quant, cv2.COLOR_LAB2BGR)
    # image = cv2.cvtColor(image, cv2.COLOR_LAB2BGR)
    # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.imread(img_filename,0)
    # img = cv2.resize(img, (300, 300)) 
    # gray = cv2.resize(gray, (300, 300))

    print img_filename
    symmetry_ssim = symmetry.get_symmetry_ssim(img)
    # symmetry_mse = symmetry.get_symmetry_mse(img)
    symmetry_mse = 0
    border_gof = border.get_gof(gray)
    #print border_gof
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

      # if counter == 10:
      #   break

  return img_path_list


if __name__ == '__main__':
  clf = build_random_forest_classfier() # 0.688888888889
  with open('mole_random_forest_classifer.pk1', 'wb') as clf_file:
    cPickle.dump(clf, clf_file)

  #build_linear_svm() # 0.601615384615
  #build_gaussian_nb() # 0.583888888889
  #build_poly_svm()

