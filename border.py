import sys
import math
import numpy as np
from PIL import Image, ImageOps, ImageMath
import imageSegmentation as IS
import cv2


from scipy import ndimage


def mse(imageA, imageB):
	# the 'Mean Squared Error' between the two images is the
	# sum of the squared difference between the two images;
	# NOTE: the two images must have the same dimension
	err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
	err /= float(imageA.shape[0] * imageA.shape[1])
	
	# return the MSE, the lower the error, the more "similar"
	# the two images are
	return err

def get_gof(img):
	img = cv2.imread(input_filename)
	gray = cv2.imread(input_filename,0)

	#finding contours
	ret,thresh = cv2.threshold(gray,100,100,1) #127,255,0
	contours,h = cv2.findContours(thresh,mode=cv2.RETR_EXTERNAL,method=cv2.CHAIN_APPROX_NONE) #thresh,1,2
	maxContIndex = 0
	maxContCount = 0
	for i in range(0,len(contours)):
		if len(contours[i]) >= maxContCount:
			maxContIndex = i
			maxContCount = len(contours[i])
	#cv2.drawContours(img, contours[maxContIndex], -1, (0,255,0), 3)

	#drawing ellipse
	ellipse = cv2.fitEllipse(contours[maxContIndex])
	# cv2.ellipse(img,ellipse,(0,255,0),2)
	
	center = ellipse[0]
	size = ellipse[1]
	angle = ellipse[2]

	gof = 0.0
	for i in range(1,len(contours[maxContIndex])):
		posx = (contours[maxContIndex][i][0][0] - center[0]) * math.cos(-angle) - (contours[maxContIndex][i][0][1]- center[1]) * math.sin(-angle)
		posy = (contours[maxContIndex][i][0][0] - center[0]) * math.sin(-angle) + (contours[maxContIndex][i][0][1]- center[1]) * math.cos(-angle)
		gof += abs(posx/size[0]*posx/size[0] + posy/size[1]*posy/size[1] - 0.25)
	return gof

if __name__ == '__main__':
	input_filename = sys.argv[1]
	
	print get_gof(input_filename)
	#TODO: need to get k-means / bounded box first
	img = cv2.imread(input_filename)
	gray = cv2.imread(input_filename,0)

	#finding contours
	ret,thresh = cv2.threshold(gray,100,100,1) #127,255,0
	contours,h = cv2.findContours(thresh,mode=cv2.RETR_EXTERNAL,method=cv2.CHAIN_APPROX_NONE) #thresh,1,2
	maxContIndex = 0
	maxContCount = 0
	for i in range(0,len(contours)):
		if len(contours[i]) >= maxContCount:
			maxContIndex = i
			maxContCount = len(contours[i])
	cv2.drawContours(img, contours[maxContIndex], -1, (0,255,0), 3)

	#drawing ellipse
	ellipse = cv2.fitEllipse(contours[maxContIndex])
	cv2.ellipse(img,ellipse,(0,255,0),2)
	
	

	center = ellipse[0]
	size = ellipse[1]
	angle = ellipse[2]

	gof = 0.0
	for i in range(1,len(contours[maxContIndex])):
		posx = (contours[maxContIndex][i][0][0] - center[0]) * math.cos(-angle) - (contours[maxContIndex][i][0][1]- center[1]) * math.sin(-angle)
		posy = (contours[maxContIndex][i][0][0] - center[0]) * math.sin(-angle) + (contours[maxContIndex][i][0][1]- center[1]) * math.cos(-angle)
		gof += abs(posx/size[0]*posx/size[0] + posy/size[1]*posy/size[1] - 0.25)
	#print gof
	cv2.imshow("window title", img)
	cv2.waitKey()


# if __name__ == '__main__':
# 	input_filename = sys.argv[1]
# 	im = Image.open(input_filename)
# 	[x, y] = im.size
# 	img = cv2.imread(input_filename)
# 	gray = cv2.imread(input_filename,0)
# 	ret,thresh = cv2.threshold(gray,127,255,1)
# 	contours,h = cv2.findContours(thresh,1,2)
# # 		cv2.drawContours(img, contours, -1, (0,255,0), 3)
# # 		cv2.imshow("window title", img)
