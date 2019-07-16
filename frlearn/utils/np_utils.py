import numpy as np


def argmax_and_max(a, axis):
    ai = np.argmax(a, axis=axis)
    av = np.squeeze(np.take_along_axis(a, np.expand_dims(ai, axis=axis), axis=axis), axis=axis)
    return ai, av


def limit_and_sort(a, limit, axis):
    a = np.partition(a, limit, axis=axis)
    take_this = np.arange(limit) if limit > 0 else np.arange(limit % a.shape[axis], a.shape[axis])
    a = np.take(a, take_this, axis=axis)
    a = np.sort(a, axis=axis)
    if limit < 0:
        a = np.flip(a, axis=axis)
    return a
