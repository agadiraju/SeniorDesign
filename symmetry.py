import sys
import math
import numpy as np
from PIL import Image, ImageOps, ImageMath
import imageSegmentation
import cv2

from scipy import ndimage
from skimage.measure import structural_similarity as ssim

def mse(imageA, imageB):
	err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
	err /= float(imageA.shape[0] * imageA.shape[1])
	
	# return the MSE, the lower the error, the more "similar"
	# the two images are
	return err

# quick hack to get even image dimensions
def get_even_image(im_dimensions):
	x, y, _ = im_dimensions

	if x % 2 == 0 and y % 2 == 0:
		return (x, y)
	else:
		return (x, y - 1)

def divide_into_two(img):
	(x, y) = get_even_image(img.shape)
	top_half = img[0:x, 0:y/2]
	bottom_half = ndimage.rotate(img[0:x, y/2:y], 180)
	top_half = cv2.cvtColor(top_half, cv2.COLOR_BGR2GRAY)
	bottom_half = cv2.cvtColor(bottom_half, cv2.COLOR_BGR2GRAY)
	return (top_half, bottom_half)

def get_symmetry(img):
	(top_half, bottom_half) = divide_into_two(img)
	return ssim(top_half, bottom_half)


# def rotateImage(image, angle):
#   image_center = tuple(np.array(image.shape)/2)
#   rot_mat = cv2.getRotationMatrix2D((image_center[0], image_center[1]),angle,1.0)
#   result = cv2.warpAffine(image, rot_mat, image.shape,flags=cv2.INTER_LINEAR)
#   return result

# class Symmetry:

# 	def __init__(self):
# 		print "hello"
		
# 	def preprocess(outputfile):
# 		img = cv2.imread(outputfile)
# 		gray = cv2.imread(outputfile,0)
# 		ret,thresh = cv2.threshold(gray,127,255,1)
# 		contours,h = cv2.findContours(thresh,1,2)
# 		cv2.drawContours(img, contours, -1, (0,255,0), 3)
# 		cv2.imshow("window title", img)



if __name__ == '__main__':
	input_filename = sys.argv[1]
		#outputfile = sys.argv[2]

		# bounded_image = plot_bounded_box(input_filename)
  # 	bounded_image.show()
	# im = Image.open(input_filename).convert('LA')  # convert to grayscale
	# [x, y] = im.size
	# #box_top = (0, 0, x, y / 2)
	# top_half = im.crop((0, 0, x, y / 2))
	# bottom_half = ImageOps.flip(im.crop((0,y/2+1,x,y))) # reflect image vertically 
	# print "here"
	# print mse(top_half, bottom_half)
	# im.show()
	# top_half.show()
	# bottom_half.show()

	img = cv2.imread(input_filename)
	(x, y) = get_even_image(img.shape)
	top_half = img[0:x, 0:y/2]
	bottom_half = ndimage.rotate(img[0:x, y/2:y], 180)
	top_half = cv2.cvtColor(top_half, cv2.COLOR_BGR2GRAY)
	bottom_half = cv2.cvtColor(bottom_half, cv2.COLOR_BGR2GRAY)
	print ssim(top_half, bottom_half)
	cv2.imshow("top half", top_half)
	cv2.imshow("bottom half", bottom_half)
	cv2.waitKey(0)
		#bottom_half
		#imageSegmentation.main(inputfile,outputfile,3)

		#img = Image.open(outputfile)
		#preprocess(outputfile)



