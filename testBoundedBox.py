import sys

from imageSegmentation import plot_bounded_box


if __name__ == '__main__':
  input_filename = sys.argv[1]
  bounded_image = plot_bounded_box(input_filename)
  bounded_image.show()
  

