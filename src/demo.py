import sys

from scipy.misc import imread, imsave

from canny import gradient, thin_nonmaximum, thin_hysteresis
from hough import hough_detect, hough_learn


def main():
    img = imread(sys.argv[1], flatten=True, mode='L')
    imsave('1-gray.png', img)

    gradient_ = gradient(img)
    imsave('2-gradient-magnitudes.png', gradient_.magnitudes)
    imsave('3-gradient-angles.png', gradient_.angles)

    thinned = thin_nonmaximum(gradient_)
    imsave('4-thinned.png', thinned.magnitudes)

    thinned_hyst = thin_hysteresis(thinned, 0.3, 0.1)
    imsave('5-thinned_hyst.png', thinned_hyst.magnitudes)

    rtable = hough_learn(thinned_hyst)
    for k, v in rtable.iteritems():
        print k, len(v)
    hough_acc = hough_detect(rtable, thinned_hyst)
    imsave('5-hough-acc.png', hough_acc[0, 0, :, :])


if __name__ == '__main__':
    main()
