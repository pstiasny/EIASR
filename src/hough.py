from collections import defaultdict
from math import cos, sin, ceil, sqrt, pi

import numpy as np


def discrete_direction(ndir, alpha):
    bucket_width = 2*pi/ndir
    return int((alpha / bucket_width + 0.5 * bucket_width) % ndir)


class HoughLine(object):
    """
    A line l given as
    
        l.d = x * cos(l.alpha) + y * sin(l.alpha)

    that received l.votes in the Hough accumulator.
    """
    def __init__(self, d, alpha, votes):
        self.d = d
        self.alpha = alpha
        self.votes = votes


def hough_lines(mag, angles):
    w, h = mag.shape
    ndir = 64

    acc = np.zeros((ndir, int(ceil(sqrt(w*w + h*h)))))

    for (x, y), s in np.ndenumerate(mag):
        if s < 0.01:
            continue
        alpha = angles[(x, y)]
        dscr_alpha = discrete_direction(ndir, alpha)
        d = x * cos(alpha) + y * cos(alpha)
        acc[dscr_alpha, int(d)] += 1

    results = [
        HoughLine(d, dscr_alpha, votes)
        for (dscr_alpha, d), votes in np.ndenumerate(acc)
        if votes > 0
    ]
    results.sort(lambda line: -line.votes)
    return results


#scales = [2**i for i in range(-3, 3)]
#rotations = [i * (pi / 16) for i in range(0, 32)]
scales = [1]
rotations = [0]


def hough_learn(img):
    rtable = defaultdict(list)
    mag_sum = img.magnitudes.sum()
    if mag_sum == 0:
        raise RuntimeError('no edges detected')
    center = (
        sum(
            mag * np.array([x, y])
            for ((x, y), mag) in np.ndenumerate(img.magnitudes)) /
        mag_sum
    )
    print 'center:', center[0], 'x', center[1]

    for scale_idx, scale in enumerate(scales):
        for rot_idx, rot in enumerate(rotations):
            for (x, y), mag in np.ndenumerate(img.magnitudes):
                if mag <= 0:
                    continue
                rx = int(center[0] - x)
                ry = int(center[1] - y)
                phi = discrete_direction(64, img.angles[x, y])
                rtable[scale_idx, rot_idx, phi].append((rx, ry))
    return rtable


def hough_detect(rtable, img):
    acc = np.zeros((len(scales), len(rotations), img.w, img.h))

    for scale_idx, scale in enumerate(scales):
        for rot_idx, rot in enumerate(rotations):
            for (x, y), mag in np.ndenumerate(img.magnitudes):
                if mag < 1:
                    continue
                dalpha = discrete_direction(64, img.angles[x, y])
                for r in rtable[scale_idx, rot_idx, dalpha]:
                    center_x = x + r[0]
                    if not (0 <= center_x < img.w):
                        continue
                    center_y = y + r[1]
                    if not (0 <= center_y < img.h):
                        continue
                    acc[scale_idx, rot_idx, center_x, center_y] += 1
                    # TODO: increase neighbours (smoothing)
    return acc
