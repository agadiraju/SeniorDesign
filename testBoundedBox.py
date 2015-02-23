# Test script for bounded box implementation in image segmentation
# To test: 'python testBoundedBox.py [imgpath]'
# Author: Abhishek Gadiraju (abhishek.gadiraju@gmail.com)

import sys

from imageSegmentation import plot_bounded_box


if __name__ == '__main__':
  input_filename = sys.argv[1]
  bounded_image = plot_bounded_box(input_filename)
  bounded_image.show()
  

