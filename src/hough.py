from collections import defaultdict
from math import cos, sin, ceil, sqrt, pi

import numpy as np


def discrete_direction(ndir, alpha):
    bucket_width = 2*pi/ndir
    return int((alpha / bucket_width + 0.5 * bucket_width) % ndir)


scales = [2**i for i in range(-3, 3)]
rotations = [i * (pi / 32) for i in range(0, 64)]


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

    for (x, y), mag in np.ndenumerate(img.magnitudes):
        if mag <= 0:
            continue
        rx = int(center[0] - x)
        ry = int(center[1] - y)
        phi = discrete_direction(64, img.angles[x, y])
        rtable[phi].append((rx, ry))
    return rtable


def hough_detect(rtable, img):
    acc = np.zeros((len(scales), len(rotations), img.w, img.h))

    for (x, y), mag in np.ndenumerate(img.magnitudes):
        if mag < 1:
            continue

        for scale_idx, scale in enumerate(scales):
            for rot_idx, rot in enumerate(rotations):
                c, s = np.cos(rot), np.sin(rot)
                Rot = np.array([[c, -s], [s, c]])

                alpha = (img.angles[x, y] + rot) % (2 * pi)
                rindex = discrete_direction(64, alpha)
                for r in rtable[rindex]:
                    r = np.dot(Rot, r)
                    r = scale * r
                    center_x = int(x + r[0])
                    if not (0 <= center_x < img.w):
                        continue
                    center_y = int(y + r[1])
                    if not (0 <= center_y < img.h):
                        continue
                    acc[scale_idx, rot_idx, center_x, center_y] += 1
                    # TODO: increase neighbours (smoothing)
    return acc
