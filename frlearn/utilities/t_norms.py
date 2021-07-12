"""
Triangular norms (t-norms): commutative, associative, non-decreasing binary operators on `[0, 1]` with identity 1.
T-norms generalise classical conjunction for fuzzy logic.

Each function aggregates the values of the provided array along the specified axis.
"""

import numpy as np

__all__ = [
    'goguen_t_norm', 'heyting_t_norm', 'lukasiewicz_t_norm',
]


def goguen_t_norm(a, axis):
    """
    `x * y`; also known as *product* t-norm.

    Parameters
    ----------
    a: np.array
        Input array of values. All values should lie in `[0, 1]`.

    axis : int, default=-1
        The axis along which values are aggregated.

    Returns
    -------
    b: np.array
        Aggregated values, array with the same shape as `a`, except for the specified axis.
    """
    return np.prod(a, axis=axis)


def heyting_t_norm(a, axis):
    """
    `min(x, y)`; also known as *Gödel* or *minimum* t-norm.

    Parameters
    ----------
    a: np.array
        Input array of values. All values should lie in `[0, 1]`.

    axis : int, default=-1
        The axis along which values are aggregated.

    Returns
    -------
    b: np.array
        Aggregated values, array with the same shape as `a`, except for the specified axis.
    """
    return np.min(a, axis=axis)


def lukasiewicz_t_norm(a, axis):
    """
    `max(x + y - 1, 0)`

    Parameters
    ----------
    a: np.array
        Input array of values. All values should lie in `[0, 1]`.

    axis : int, default=-1
        The axis along which values are aggregated.

    Returns
    -------
    b: np.array
        Aggregated values, array with the same shape as `a`, except for the specified axis.
    """
    return np.maximum(np.sum(a, axis=axis) - (a.shape[axis] - 1), 0)
