import cPickle
import cv2
import numpy as np
import sys

from build_model import extract_image_features


def classify_image(img_path):
  # with open('mole_random_forest_classifer.pk1', 'rb') as fid:
  clf = cPickle.load(open('mole_random_forest_classifer.pk1', 'rb'))
  img = cv2.imread(img_path)

  new_entry = extract_image_features(img)

  prediction = clf.predict(np.array(new_entry))

  if prediction:
    print "malignant"
  else:
    print "benign"
  #(benign_mse, benign_ssim, benign_border, benign_)


if __name__ == '__main__':
  input_filename = sys.argv[1]
  classify_image(input_filename)