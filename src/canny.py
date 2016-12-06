import numpy as np
from scipy.signal import convolve2d


class GradientImage(object):
    def __init__(self, dxs, dys):
        self.dxs = dxs
        self.dys = dys
        self.magnitudes = np.sqrt(dxs ** 2 + dys ** 2)
        self.angles = np.arctan(dxs/dys)


SOBEL_X = np.array([
    [-1, 0,  1],
    [-2, 0,  2],
    [-1, 0,  1],
])

SOBEL_Y = np.array([
    [-1, 2, -1],
    [ 0, 0,  0],
    [ 1, 2,  1],
])


def gradient(in_):
    return GradientImage(
        convolve2d(in_, SOBEL_X, 'same'),
        convolve2d(in_, SOBEL_Y, 'same'))


def thin_nonmaximum(gradient_image):
    return gradient_image.dxs  # TODO


def thin_hysteresis(high, low, magnitudes):
    return magnitudes  # TODO
