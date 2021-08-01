"""Functions for expressing one value in terms of another."""

import numpy as np

__all__ = [
    'log_multiple', 'multiple',
]


def log_multiple(a):
    """
    Function to obtain multiples of the logarithm of other numbers.

    Parameters
    ----------
    a : float
        The multiple to be used.

    Returns
    -------
    f : int -> int
        Function that takes a number `x` and returns `a * log x`.
    """
    def _f(x):
        return a * np.log(x)
    return _f


def multiple(a: float):
    """
    Function to obtain multiples of other numbers.

    Parameters
    ----------
    a: float
        The multiple to be used.

    Returns
    -------
    f: float -> float
        Function that takes a number `x` and returns `a * x`.
    """
    def _f(x):
        return a * x
    return _f
