import cPickle
import cv2
import numpy as np
import sys

from build_model import extract_image_features


def classify_image(img_path):
  # with open('mole_random_forest_classifer.pk1', 'rb') as fid:
  clf = cPickle.load(open('mole_random_forest_classifer.pk1', 'rb'))
  img = cv2.imread(img_path)

  malig_avgs = []
  benign_avgs = []
  with open('malignant_averages.txt', 'r') as malig_file:
    malignant_avgs = [float(line.rstrip('\n')) for line in malig_file]
  with open('benign_averages.txt', 'r') as benign_file:
    benign_avgs = [float(line.rstrip('\n')) for line in benign_file]

  labels = ['mse symmetry', 'ssim symmetry', 'border_gof', 'color_contrast']
  new_entry = extract_image_features(img)


  prediction = clf.predict(np.array(new_entry))
  output_messages = []

  if prediction:
    #print "malignant"
    for idx, entry in enumerate(new_entry):
      if idx == 1:
        if entry < benign_avgs[idx]:
          print labels[idx] + ' value is lower than a normal benign mole'
      else:
        if entry > benign_avgs[idx]:
          print labels[idx] + ' value is higher than a normal benign mole'

  else:
    print "benign"
  #(benign_mse, benign_ssim, benign_border, benign_)


if __name__ == '__main__':
  input_filename = sys.argv[1]
  classify_image(input_filename)