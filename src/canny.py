# coding: utf8
from math import pi

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
    def __init__(self, magnitudes, angles):
        self.magnitudes = magnitudes
        self.angles = angles

    @property
    def w(self):
        return self.magnitudes.shape[0]

    @property
    def h(self):
        return self.magnitudes.shape[1]

    @classmethod
    def from_partials(cls, dxs, dys):
        magnitudes = np.sqrt(dxs ** 2 + dys ** 2)
        angles = np.arctan2(dys, dxs)
        return cls(magnitudes, angles)


def gradient(in_):
    dxs = convolve2d(in_, SOBEL_X, 'same', 'symm')
    dys = convolve2d(in_, SOBEL_Y, 'same', 'symm')
    return GradientImage.from_partials(dxs, dys)


def thin_nonmaximum(gradient_image):
    thinned = np.copy(gradient_image.magnitudes)
    for idx, s in np.ndenumerate(gradient_image.magnitudes):
        s_nl = _neighbour_in_direction(
            gradient_image.magnitudes, idx,
            gradient_image.angles[idx])
        s_nr = _neighbour_in_direction(
            gradient_image.magnitudes, idx,
            gradient_image.angles[idx] + pi)
        # TODO: consider angle at nl, nr
        if s < s_nl or s < s_nr:
            thinned[idx] = 0

    return GradientImage(thinned, gradient_image.angles)


def thin_hysteresis(high, low, magnitudes):
    return magnitudes  # TODO


NEIGHBOURS = [
    ( 0,  1),
    ( 1,  1),
    ( 1,  0),
    ( 1, -1),
    ( 0, -1),
    (-1, -1),
    (-1,  0),
    (-1,  1),
]

def _neighbour_in_direction(a, (x, y), direction):
    w, h = a.shape
    ndir = len(NEIGHBOURS)
    discrete_direction = int((direction / (2*pi) * ndir + 0.5 * ndir) % ndir)
    dx, dy = NEIGHBOURS[discrete_direction]
    nx, ny = x + dx, y + dy

    if not (0 <= nx < w and 0 <= ny < h):
        return 0

    return a[nx, ny]
