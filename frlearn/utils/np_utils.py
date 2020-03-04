"""Utility functions for numpy"""

import numpy as np


def argmax_and_max(a, axis):
    ai = np.argmax(a, axis=axis)
    av = np.squeeze(np.take_along_axis(a, np.expand_dims(ai, axis=axis), axis=axis), axis=axis)
    return ai, av


def least(a, k, axis=-1):
    a = np.partition(a, k - 1, axis=axis)
    take_this = np.arange(k)
    a = np.take(a, take_this, axis=axis)
    a = np.sort(a, axis=axis)
    return a


def greatest(a, k, axis=-1):
    a = np.partition(a, -k, axis=axis)
    take_this = np.arange(-k % a.shape[axis], a.shape[axis])
    a = np.take(a, take_this, axis=axis)
    a = np.sort(a, axis=axis)
    a = np.flip(a, axis=axis)
    return a


def div_or(x, y, fallback=np.nan):
    with np.errstate(divide='ignore', invalid='ignore'):
        z = x / y
        z[np.isnan(z)] = fallback
    return z
