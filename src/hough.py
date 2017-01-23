from collections import defaultdict, namedtuple
from math import cos, sin, ceil, sqrt, pi

#from numba import jit
import numpy as np
from scipy.ndimage.filters import gaussian_filter
from skimage.feature import peak_local_max


NDIR = 64

bucket_width = 2*pi/NDIR
bias = bucket_width * 0.5
#@jit
def discrete_direction(alpha):
    alpha_plus = alpha + bias
    return int((alpha_plus / bucket_width) % NDIR)


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
        phi = discrete_direction(img.angles[x, y])
        rtable[phi].append((rx, ry))
    return rtable


HoughDetectionResult = namedtuple(
    'HoughDetectionResult',
    ['accumulator', 'candidates'])


#@jit
def _inner_loop(rtable, acc, w, h, angle, x, y, rot_idx, rot):
    c, s = np.cos(rot), np.sin(rot)
    Rot = np.array([[c, -s], [s, c]])

    alpha = (angle + rot) % (2 * pi)
    rindex = discrete_direction(alpha)
    for r in rtable[rindex]:
        r = np.dot(Rot, r)
        for scale_idx, scale in enumerate(scales):
            r_scaled = scale * r
            center_x = int(x + r_scaled[0])
            if not (0 <= center_x < w):
                continue
            center_y = int(y + r_scaled[1])
            if not (0 <= center_y < h):
                continue
            acc[scale_idx, rot_idx, center_x, center_y] += 1


#@profile
def hough_detect(rtable, img, on_progress=None):
    w = img.w
    h = img.h
    acc = np.zeros((len(scales), len(rotations), w, h))
    num_pixels = w * h
    np_zero = np.array([0, 0])
    np_max = np.array([w, h])

    np_rtable = {k: np.array(rtable[k]) for k in xrange(NDIR)}
    np_scales = np.array(scales)
    RMxs = []
    for rot_idx, rot in enumerate(rotations):
        c, s = np.cos(rot), np.sin(rot)
        Rot = np.array([[c, -s], [s, c]])
        RMxs.append(Rot)

    for (x, y), mag in np.ndenumerate(img.magnitudes):
        if mag < 1:
            continue
        pt = np.array([x, y])
        angle = img.angles[x, y]

        if on_progress is not None:
            on_progress(100 * (x * h + y) / num_pixels)

        for rot_idx, rot in enumerate(rotations):
            Rot = RMxs[rot_idx]

            alpha = (angle + rot) % (2 * pi)
            rindex = discrete_direction(alpha)
            rs = np_rtable[rindex]
            if len(rs) == 0:
                continue
            rotated_rs = np.dot(Rot, rs.transpose()).transpose()
            for scale_idx, scale in enumerate(scales):
                scaled_rs = (rotated_rs * scale).astype(int)
                indicies = scaled_rs + pt
                valid_indicies = (
                    (indicies >= np_zero) & (indicies < np_max)
                ).all(axis=1)
                indicies = indicies[valid_indicies]
                acc[scale_idx, rot_idx, indicies[:, 0], indicies[:, 1]] += 1

    on_progress(100)
    acc = gaussian_filter(acc, 2)
    max_scale_idx, max_rot_idx, cx, cy = \
        np.unravel_index(acc.argmax(), acc.shape)
    return HoughDetectionResult(
        acc,
        [
            (scales[max_scale_idx], rotations[max_rot_idx], cx, cy)
        ])
