from pickle import load
from sys import argv

import numpy as np
from scipy.misc import imread, imsave

from canny import gradient, thin_nonmaximum
from hough import hough_detect


if __name__ == '__main__':
    img = imread(argv[2], flatten=True, mode='L')
    with open(argv[1]) as f:
        rtable = load(f)
    gradient_img = gradient(img)
    gradient_img = thin_nonmaximum(gradient_img)
    acc = hough_detect(rtable, gradient_img)
    imsave(argv[3], np.sum(acc, axis=(0, 1)))
