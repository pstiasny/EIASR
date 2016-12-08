import numpy as np
from scipy.signal import convolve2d


SOBEL_X = np.array([
    [ 1, 0, -1],
    [ 2, 0, -2],
    [ 1, 0, -1],
])

SOBEL_Y = np.array([
    [ 1,  2,  1],
    [ 0,  0,  0],
    [-1, -2, -1],
])


class GradientImage(object):
    def __init__(self, dxs, dys):
        self.dxs = dxs
        self.dys = dys
        self.magnitudes = np.sqrt(dxs ** 2 + dys ** 2)
        self.angles = np.arctan2(dys, dxs)


def gradient(in_):
    return GradientImage(
        convolve2d(SOBEL_X, in_, 'same'),
        convolve2d(SOBEL_Y, in_, 'same'))


def thin_nonmaximum(gradient_image):
    return gradient_image.dxs  # TODO


def thin_hysteresis(high, low, magnitudes):
    return magnitudes  # TODO
