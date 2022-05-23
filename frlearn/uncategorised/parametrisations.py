"""Functions for expressing one value in terms of another."""

import math

__all__ = [
    'at_most', 'log_multiple', 'multiple',
]


def at_most(a: float):
    """
    Function to limit other numbers to a certain maximum.

    Parameters
    ----------
    a: float
        The maximum to be used.

    Returns
    -------
    f: float -> float
        Function that takes a number `x` and returns `max(x, a)`.
    """
    def _f(x):
        return min(x, a)
    return _f


def log_multiple(a: float):
    """
    Function to obtain multiples of the logarithm of other numbers.

    Parameters
    ----------
    a : float
        The multiple to be used.

    Returns
    -------
    f : float -> float
        Function that takes a number `x` and returns `a * log x`.
    """
    def _f(x):
        return a * math.log(x)
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
