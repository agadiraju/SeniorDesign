# Image Clustering via KMeans
# Author: Abhishek Gadiraju (abhishek.gadiraju@gmail.com)

import numpy as np
import random
import sys

from PIL import Image


class KMeans:

  def __init__(self, K):
    self.K = K
    self.centroids = []  # list of centroids
    self.clusters = {}  # dictionary of clusters with index of centroid as key
    self.convergence_factor = .0001
    self.max_iterations = 100


  # Input: two rows from feature vector X
  # Output: a distance metric calculated from the last two values in the rows
  def get_distance(self, point1, point2):
    return np.linalg.norm(point1 - point2)

  def cluster(self, X):
    n, d = X.shape

    init_centroids = [X[e] for e in random.sample(range(n), self.K)]  # random sample from entries in X
    self.centroids = init_centroids
    
    num_iterations = 0
    while True:
      num_iterations += 1
      self.clusters = {idx : [] for idx in range(len(self.centroids))}  # empty list of points 
      for idx, entry in enumerate(X):
        current_distance = 99999999  # absurdly high value
        optimal_cluster = None
        for c in self.clusters:
          centroid = self.centroids[c]
          cluster_distance = self.get_distance(centroid, entry)
          if cluster_distance < current_distance:
            current_distance = cluster_distance
            optimal_cluster = c  # point should be assigned to this cluster now
        self.clusters[optimal_cluster].append(entry)  

      # check if clusters are empty
      for (k, v) in self.clusters.iteritems():
        if v == []:
          rand_key = [k for (k, v) in self.clusters.iteritems() if v != []]
          rand_point = self.clusters[rand_key].pop()
          self.clusters[k].append(rand_point)       


      # new centroids are the mean of the points in each cluster
      new_centroids = [np.mean(np.array(self.clusters[i]), axis=0) for i in sorted(self.clusters.keys())]

      # check for convergence
      if self.get_distance(np.array(self.centroids), np.array(new_centroids)) < self.convergence_factor:
        return (self.centroids, self.clusters)
      self.centroids = new_centroids

      if num_iterations == self.max_iterations:  # maximum iteration failsafe
        return (self.centroids, self.clusters)


def generate_features(im):
  #im = Image.open(filename)
  rgb_im = im.convert('RGB')
  n, d = im.size
  feature_vector = np.zeros(shape=(n * d, 5))  # each pixel is one feature
  counter = 0
  for i in xrange(n):
    for j in xrange(d):
      r, g, b = rgb_im.getpixel((i, j))
      feature_vector[counter] = [r, g, b, i, j]
      counter += 1
  return feature_vector 


def standardize_features(X):
  n, d = X.shape

  # calculate all the means and stds
  standardize = []
  for i in xrange(d):
    column_vector = X[:, i]
    column_mean = np.mean(column_vector)
    column_std = np.std(column_vector)
    standardize.append((column_mean, column_std))

  # do the actual standardization by column
  standard = np.zeros(shape = X.shape)
  for i in xrange(X.shape[0]):
    for j in xrange(X.shape[1]):
      (f_mean, f_std) = standardize[j]
      initial_value = X[i][j]
      if (f_std) == 0:
        standard[i][j] = (initial_value - f_mean)
      else:
        standard[i][j] = (initial_value - f_mean) / (f_std)
  return (standard, standardize)  # return mean/std list for later inversion


def invert_standardization(X, s_list):
  normal = np.zeros(shape = X.shape)
  for i in xrange(X.shape[0]):
    for j in xrange(X.shape[1]):
      (f_mean, f_std) = s_list[j]
      standard_value = X[i][j]
      if (f_std) == 0:
        normal[i][j] = standard_value + f_mean
      else:
        normal[i][j] = (f_std * standard_value) + f_mean
  return np.rint(normal)  # get rid of black lines


def plot_image(clusters, size):
  new_im = Image.new('RGB', size)
  pix = new_im.load()
  for c in clusters.iterkeys():
    for row in clusters[c]:
      [r, g, b] = map(int, row[:3])
      [x, y] = map(int, row[-2:])
      pix[x, y] = (r, g, b)
  #new_im.save(output_filename)
  return new_im


def kmeans_exec(K, im):
  # K = int(sys.argv[1])
  # input_filename = sys.argv[2]
  # output_filename = sys.argv[3]

  #im = Image.open(input_filename)
  X = generate_features(im)
  X, standardize_list = standardize_features(X)
  kmeans = KMeans(K)
  centroids, clusters = kmeans.cluster(X)

  for c in sorted(clusters.keys()):
    current_centroid = centroids[c]
    [cent_r, cent_g, cent_b] = current_centroid[:3]
    for v in clusters[c]:
      v[:3] = [cent_r, cent_g, cent_b]

  for c in clusters.keys():
    clusters[c] = invert_standardization(np.array(clusters[c]), standardize_list)

  return clusters


def get_colors(K):
  X = generate_features(im)
  X, standardize_list = standardize_features(X)
  kmeans = KMeans(K)
  centroids, clusters = kmeans.cluster(X)

  return [c[:3] for c in centroids]  # only RGB values


def get_bounded_box_coordinates(clusters, im_size):
  (max_x, max_y) = im_size
  cluster_points = clusters.values()
  min_x_vals = {}
  min_y_vals = {}
  max_x_vals = {}
  max_y_vals = {}
  colors = set()
  for points in cluster_points:
    for p in points:
      # prev_min_x = min_x_vals
      # prev_min_y = min_y_vals
      # prev_max_x = max_x_vals
      # prev_max_y = max_y_vals
      updated = False
      (current_x, current_y) = tuple(p[-2:])
      (r, g, b) = tuple(p[:3])
      colors.add((r, g, b))

      if (r, g, b) not in min_x_vals:
        min_x_vals[(r, g, b)] = current_x
      else:
        if current_x < min_x_vals[(r, g, b)]:
          min_x_vals[(r, g, b)] = current_x
          updated = True

      if (r, g, b) not in min_y_vals:
        min_y_vals[(r, g, b)] = current_y
      else:
        if current_y < min_y_vals[(r, g, b)]:
          min_y_vals[(r, g, b)] = current_y
          updated = True

      if (r, g, b) not in max_x_vals:
        max_x_vals[(r, g, b)] = current_x
      else:
        if current_x > max_x_vals[(r, g, b)]:
          max_x_vals[(r, g, b)] = current_x
          #updated = True      

      if (r, g, b) not in max_y_vals:
        max_y_vals[(r, g, b)] = current_y
      else:
        if current_y > max_y_vals[(r, g, b)]:
          max_y_vals[(r, g, b)] = current_y
          #updated = True 

      # if updated:
      #   print p
      #   print max_x_vals
      #   print max_y_vals
      #   print min_x_vals
      #   print min_y_vals
        #raw_input()

  # Assuming K = 2

  # print (max_x, max_y)
  # raw_input()
  # print max_x_vals
  # print max_y_vals
  # print min_x_vals
  # print min_y_vals
  # raw_input()
  # print colors
  # raw_input()
  mole_color = [k for k in colors if (max_x_vals[k] + 1) != max_x or (max_y_vals[k] + 1) != max_y]
  # print mole_color
  # raw_input()
  if not mole_color:  # current frame is the bounding box
    return map(int, (0, 0, max_x, max_y))
  mole_color = mole_color[0]

  left = min_x_vals[mole_color]
  upper = min_y_vals[mole_color]
  right = max_x_vals[mole_color]
  down = max_y_vals[mole_color]

  return map(int,(left, upper, right, down))


def plot_bounded_box(original_image):
  im = Image.open(original_image)
  clusters = kmeans_exec(2, im)  # K = 2 to segment image into mole and surrounding skin
  coordinates = get_bounded_box_coordinates(clusters, im.size)
  bounded_box_image = im.crop(coordinates)
  return bounded_box_image


def main(input_filename, output_filename, K):
  im = Image.open(input_filename)
  X = generate_features(im)
  X, standardize_list = standardize_features(X)
  kmeans = KMeans(K)
  centroids, clusters = kmeans.cluster(X)

  for c in sorted(clusters.keys()):
    current_centroid = centroids[c]
    [cent_r, cent_g, cent_b] = current_centroid[:3]
    for v in clusters[c]:
      v[:3] = [cent_r, cent_g, cent_b]

  for c in clusters.keys():
    clusters[c] = invert_standardization(np.array(clusters[c]), standardize_list)

  img = plot_image(clusters, im.size)
  img.save(output_filename)

if __name__ == '__main__':
  K = int(sys.argv[1])
  input_filename = sys.argv[2]
  output_filename = sys.argv[3]
  main(input_filename,output_filename,K)
  
