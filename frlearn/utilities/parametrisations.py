"""Functions for expressing one value in terms of another."""

import numpy as np

__all__ = [
    'fraction', 'log_units',
]


def fraction(a):
    """
    Creates a function that calculates a positive integer as a fraction of some maximum.

    Parameters
    ----------
    a : float
        The fraction to be used. Should be in [0, 1].

    Returns
    -------
    f : int -> int
        Function that takes a maximum value `x` and returns `a * x`, rounded to the closest integer in `[1, x]`.
    """
    def _f(x):
        return min(max(1, int(a * x)), x)
    return _f


def log_units(a):
    """
    Creates a function that calculates a positive integer as a multiple of the logarithm of some maximum.

    Parameters
    ----------
    a : float
        The multiple to be used. Should be in `[0, ∞)`.

    Returns
    -------
    f : int -> int
        Function that takes a maximum value `x` and returns `a * log x`, rounded to the closest integer in `[1, x]`.
    """
    def _f(x):
        return min(max(1, int(a * np.log(x))), x)
    return _f
