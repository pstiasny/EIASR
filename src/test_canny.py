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

    in2 = np.array([
        [ 0,  1,  1],
        [ 0,  1,  1],
        [ 0,  1,  1],
    ])
    out2 = gradient(in2)
    assert out2.dxs[1, 1] == 4
    assert out2.dys[1, 1] == 0
    assert out2.angles[1, 1] == 0


def test_thin_nonmaximum():
    in_ = GradientImage(
        np.array([
            [  0,   0,   0,   0],
            [  0, 100,   0, 255],
            [  0,   0, 255,   0],
            [  0, 255,   0,   0],
        ]),
        np.array([
            [  0,   0,   0,   0],
            [  0, 100,   0, 255],
            [  0,   0, 255,   0],
            [  0, 255,   0,   0],
        ]))

    out = thin_nonmaximum(in_)

    assert (out == np.array([
        [  0,   0,   0,   0],
        [  0,   0,   0, 255],
        [  0,   0, 255,   0],
        [  0, 255,   0,   0],
    ])).all()


def test_thin_hysteresis():
    assert False
