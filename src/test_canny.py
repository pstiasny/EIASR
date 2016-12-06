import numpy as np

from canny import GradientImage, gradient, thin_nonmaximum


def test_gradient():
    in_ = np.array([
        [20, 10,  0],
        [30, 20, 10],
        [40, 30, 20],
    ])

    out = gradient(in_)

    assert np.all(out.dxs == np.array([
        [-40,  60,  40],
        [-80,  80,  80],
        [-80,  60,  80],
    ]))

    assert np.all(out.dys == np.array([
        [ 40,   0,   0],
        [100,  40,  20],
        [ 80,  80,  40],
    ]))


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
