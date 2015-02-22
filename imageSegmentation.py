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
    self.convergence_factor = .001
    self.max_iterations = 50

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


def plot_image(clusters, size, output_filename):
  new_im = Image.new('RGB', size)
  pix = new_im.load()
  for c in clusters.iterkeys():
    for row in clusters[c]:
      [r, g, b] = map(int, row[:3])
      [x, y] = map(int, row[-2:])
      pix[x, y] = (r, g, b)
  new_im.save(output_filename)


def kmeans_exec(K, input_filename):
  # K = int(sys.argv[1])
  # input_filename = sys.argv[2]
  # output_filename = sys.argv[3]

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

  return clusters


def get_colors(K):
  X = generate_features(im)
  X, standardize_list = standardize_features(X)
  kmeans = KMeans(K)
  centroids, clusters = kmeans.cluster(X)

  return [c[:3] for c in centroids]  # only RGB values

  # for c in sorted(clusters.keys()):
  #   current_centroid = centroids[c]
  #   [cent_r, cent_g, cent_b] = current_centroid[:3]
  #   for v in clusters[c]:
  #     v[:3] = [cent_r, cent_g, cent_b]

def get_bounded_box_coordinates(clusters):
  cluster = clusters.values()
  min_vals = {}
  for points in cluster_points:
    for p in points:
      






if __name__ == '__main__':
  K = int(sys.argv[1])
  input_filename = sys.argv[2]
  output_filename = sys.argv[3]

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

  plot_image(clusters, im.size, output_filename)