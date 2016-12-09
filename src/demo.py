import sys

from scipy.misc import imread, imsave

from canny import gradient, thin_nonmaximum


def main():
    img = imread(sys.argv[1], flatten=True, mode='L')
    imsave('1-gray.png', img)

    gradient_ = gradient(img)
    imsave('2-gradient-magnitudes.png', gradient_.magnitudes)
    imsave('3-gradient-angles.png', gradient_.angles)

    thinned = thin_nonmaximum(gradient_)
    imsave('4-thinned.png', thinned)


if __name__ == '__main__':
    main()
