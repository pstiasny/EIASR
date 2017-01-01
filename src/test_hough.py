import numpy as np

from hough import hough_lines
from canny import gradient


def test_hough_lines():
    mag = np.array([
        [0, 0, 0, 255],
        [0, 0, 255, 0],
        [0, 255, 0, 0],
        [255, 0, 0, 0],
    ])
    angles = 0.25 * np.pi * np.ones((4, 4))

    result = hough_lines(mag, angles)
    assert result[0].d == 2
    assert result[0].alpha == 8
