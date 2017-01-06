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


def thin_hysteresis(gradient_image, t_high=0.2, t_low=0.1):

    # 8 pixel neighborhood
    x = [-1,  0,  1, -1,  1, -1,  0,  1]
    y = [-1, -1, -1,  0,  0,  1,  1,  1]
    
    magnitudes = gradient_image.magnitudes
    
    # Dimensions
    xdim, ydim = magnitudes.shape
    
    # Max magnitude
    max_magn = magnitudes.max()
    
    # Pixels > t_high are kept automatically 
    thinned = np.where(magnitudes > (t_high * max_magn), magnitudes, 0)

    # Pixels > t_low will be ad ded later if they prove to be    
    # adjacent to another pixel which has been included in the thinned list
    cands = np.where(magnitudes > (t_low * max_magn), magnitudes, 0)

    # Create an initial list of strong edge pixels
    prevx, prevy = thinned.nonzero()

    # If the previous loop of testing found no new pixels to move from
    # the cands list to the edge list, then stop
    while len(prevx) != 0:
        newx, newy = [], []
        # Loop over new edge pixels discovered on previous iteration
        for ii in range(len(prevx)):
            # Loop through 8 pixel neighborhood
            for ij in range(len(x)):
                xidx = prevx[ii] + x[ij]
                yidx = prevy[ii] + y[ij]
                # Check if pixel index falls within image boundary
                if xidx >= 0 and xidx < xdim and yidx >= 0 and yidx < ydim:
                    # Check if pixel is on the cands list but has not yet been added to the thinned list
                    if cands[xidx][yidx] and not thinned[xidx][yidx]:
                        # Transfer to thinned list
                        thinned[xidx][yidx] = cands[xidx][yidx]
                        # Keep track of indices for next loop iteration
                        newx.append(xidx)
                        newy.append(yidx)
        # Update for next iteration
        prevx = newx
        prevy = newy

    return GradientImage(thinned, gradient_image.angles)


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
