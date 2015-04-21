import sys
import math
import numpy as np
from PIL import Image, ImageOps, ImageMath
import imageSegmentation
import cv2

if __name__ == '__main__':
	input_filename = sys.argv[1]
	im = Image.open(input_filename)
	[x, y] = im.size
	img = cv2.imread(input_filename)
	gray = cv2.imread(input_filename,0)
	ret,thresh = cv2.threshold(gray,127,255,1)
	contours,h = cv2.findContours(thresh,1,2)
# 		cv2.drawContours(img, contours, -1, (0,255,0), 3)
# 		cv2.imshow("window title", img)