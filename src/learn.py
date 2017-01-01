from pickle import dump
from sys import argv

from scipy.misc import imread, imsave

from canny import gradient, thin_nonmaximum
from hough import hough_learn


if __name__ == '__main__':
    img = imread(argv[1], flatten=True, mode='L')
    gradient_img = gradient(img)
    gradient_img = thin_nonmaximum(gradient_img)
    imsave(argv[1] + '.grad.png', gradient_img.magnitudes)
    rtable = hough_learn(gradient_img)
    with open(argv[1] + '.rtable', 'w') as f:
        dump(rtable, f)
