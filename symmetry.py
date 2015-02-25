import sys
import math
import numpy as np
from PIL import Image
import imageSegmentation
import cv2



class Symmetry:

	def __init__(self):
		print "hello"
		
	def preprocess(outputfile):
		img = cv2.imread(outputfile)
		gray = cv2.imread(outputfile,0)
		ret,thresh = cv2.threshold(gray,127,255,1)
		contours,h = cv2.findContours(thresh,1,2)
		cv2.drawContours(img, contours, -1, (0,255,0), 3)
		cv2.imshow("window title", img)



	if __name__ == '__main__':
		inputfile = sys.argv[1]
		outputfile = sys.argv[2]

		imageSegmentation.main(inputfile,outputfile,3)

		img = Image.open(outputfile)
		preprocess(outputfile)

