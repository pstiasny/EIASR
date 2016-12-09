from math import pi

import numpy as np

from canny import GradientImage, gradient, thin_nonmaximum


def test_gradient():
    in1 = np.array([
        [ 0,  0,  0],
        [ 1,  1,  1],
        [ 1,  1,  1],
    ])
    out1 = gradient(in1)
    assert out1.dxs[1, 1] == 0
    assert out1.dys[1, 1] == 4
    assert out1.angles[1, 1] == pi/2
    assert out1.magnitudes[1, 1] == 4

    in2 = np.array([
        [ 0,  1,  1],
        [ 0,  1,  1],
        [ 0,  1,  1],
    ])
    out2 = gradient(in2)
    assert out2.dxs[1, 1] == 4
    assert out2.dys[1, 1] == 0
    assert out2.angles[1, 1] == 0
    assert out2.magnitudes[1, 1] == 4


def test_thin_nonmaximum():
    d = np.array([
        [  5,   0, 255],
        [  0, 255,   0],
        [255,   0,   5],
    ])
    fat_edge1 = GradientImage(d, d)
    out1 = thin_nonmaximum(fat_edge1)
    assert np.array_equal(
        out1 > 0,
        np.array([
            [False, False,  True],
            [False,  True, False],
            [ True, False, False],
        ]))

    dxs = np.array([
        [  5, 255, 5],
        [ 10, 255, 5],
        [  5, 255, 5],
    ])
    fat_edge2 = GradientImage(dxs, np.zeros((3, 3)))
    out2 = thin_nonmaximum(fat_edge2)
    assert np.array_equal(
        out2 > 0,
        np.array([
            [False,  True, False],
            [False,  True, False],
            [False,  True, False],
        ]))


def test_thin_hysteresis():
    assert False
