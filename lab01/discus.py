import numpy as np


def discus(x):
    # f(x) = 1e6 * x1^2 + x2^2 + ... + xd^2
    x = np.asarray(x, dtype=float)
    return float(1e6 * x[0] ** 2 + np.sum(x[1:] ** 2))
