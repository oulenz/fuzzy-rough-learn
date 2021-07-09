"T-norms"

import numpy as np

__all__ = [
    'lukasiewicz',
]


def lukasiewicz(a, axis):
    return np.maximum(np.sum(a, axis=axis) - (a.shape[axis] - 1), 0)
