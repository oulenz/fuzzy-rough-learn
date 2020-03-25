"T-norms"

import numpy as np


def lukasiewicz(a, axis):
    return np.maximum(np.sum(a, axis=axis) - (a.shape[axis] - 1), 0)
