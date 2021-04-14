"""Utility functions for numpy"""

from typing import Callable, Union

import numpy as np


def argmax_and_max(a, axis):
    ai = np.argmax(a, axis=axis)
    av = np.squeeze(np.take_along_axis(a, np.expand_dims(ai, axis=axis), axis=axis), axis=axis)
    return ai, av


def first(a, k: Union[int, Callable[[int], int]], axis: int = -1):
    """
    Returns the `k` first values of `a` along the specified axis.
    If `k` is in `(0, 1)`, it is interpreted as a fraction of the total number, with a minimum of 1.
    If `k` is None, all values of `a` will be returned.

    Parameters
    ----------
    a : ndarray
        Input array of values.

    k : int or (int -> int)
        Number of values to return.
        Should be either a positive integer not larger than `a` along `axis`,
        or a function that takes the size of `a` along `axis` and returns such an integer.

    axis : int, default=-1
        The axis along which values are selected.

    Returns
    -------
    first_along_axis : ndarray
        An array with the same shape as `a`, with the specified axis reduced according to the value of `k`.
    """
    if callable(k):
        k = k(a.shape[axis])
    if k == a.shape[axis]:
        return a
    slc = [slice(None)] * len(a.shape)
    slc[axis] = slice(0, k)
    return a[tuple(slc)]


def last(a, k: Union[int, Callable[[int], int]], axis: int = -1):
    """
    Returns the `k` last values of `a` along the specified axis, in reverse order.
    If `k` is in `(0, 1)`, it is interpreted as a fraction of the total number, with a minimum of 1.
    If `k` is None, all values of `a` will be returned.

    Parameters
    ----------
    a : ndarray
        Input array of values.

    k : int or (int -> int)
        Number of values to return.
        Should be either a positive integer not larger than `a` along `axis`,
        or a function that takes the size of `a` along `axis` and returns such an integer.

    axis : int, default=-1
        The axis along which values are selected.

    Returns
    -------
    last_along_axis : ndarray
        An array with the same shape as `a`, with the specified axis reduced according to the value of `k`.
    """
    if callable(k):
        k = k(a.shape[axis])
    if k == a.shape[axis]:
        return np.flip(a, axis=axis)
    slc = [slice(None)] * len(a.shape)
    slc[axis] = slice(-1, -k - 1, -1)
    return a[tuple(slc)]


def least(a, k: Union[int, Callable[[int], int]], axis: int = -1):
    """
    Returns the `k` least values of `a` along the specified axis, in order.
    If `k` is in `(0, 1)`, it is interpreted as a fraction of the total number, with a minimum of 1.
    If `k` is None, all values of `a` will be returned.

    Parameters
    ----------
    a : ndarray
        Input array of values.

    k : int or (int -> int)
        Number of values to return.
        Should be either a positive integer not larger than `a` along `axis`,
        or a function that takes the size of `a` along `axis` and returns such an integer.

    axis : int, default=-1
        The axis along which values are selected.

    Returns
    -------
    least_along_axis : ndarray
        An array with the same shape as `a`, with the specified axis reduced according to the value of `k`.
    """
    if callable(k):
        k = k(a.shape[axis])
    if k == a.shape[axis]:
        return np.sort(a, axis=axis)
    a = np.partition(a, k - 1, axis=axis)
    take_this = np.arange(k)
    a = np.take(a, take_this, axis=axis)
    a = np.sort(a, axis=axis)
    return a


def greatest(a, k: Union[int, Callable[[int], int]], axis: int = -1):
    """
    Returns the `k` greatest values of `a` along the specified axis, in order.
    If `k` is in `(0, 1)`, it is interpreted as a fraction of the total number, with a minimum of 1.
    If `k` is None, all values of `a` will be returned.

    Parameters
    ----------
    a : ndarray
        Input array of values.

    k : int or (int -> int)
        Number of values to return.
        Should be either a positive integer not larger than `a` along `axis`,
        or a function that takes the size of `a` along `axis` and returns such an integer.

    axis : int, default=-1
        The axis along which values are selected.

    Returns
    -------
    greatest_along_axis : ndarray
        An array with the same shape as `a`, with the specified axis reduced according to the value of `k`.
    """
    if callable(k):
        k = k(a.shape[axis])
    if k == a.shape[axis]:
        return np.flip(np.sort(a, axis=axis), axis=axis)
    a = np.partition(a, -k, axis=axis)
    take_this = np.arange(-k % a.shape[axis], a.shape[axis])
    a = np.take(a, take_this, axis=axis)
    a = np.flip(np.sort(a, axis=axis), axis=axis)
    return a


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
    return lambda x: min(max(1, int(a * x)), x)


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
    return lambda x: min(max(1, int(a * np.log(x))), x)


def div_or(x, y, fallback=np.nan):
    """
    Divides `x` by `y`, replacing `np.nan` values with `fallback`.

    Parameters
    ----------
    x : ndarray
        Dividend.

    y : ndarray
        Divisor.

    fallback : numerical, default=np.nan
        Fallback value to substitute for `np.nan` after division.

    Returns
    -------
    z : ndarray
        Quotient.
    """
    with np.errstate(divide='ignore', invalid='ignore'):
        z = x / y
        z[np.isnan(z)] = fallback
    return z


def remove_diagonal(a):
    #TODO: parametrise dimensions
    return a[~np.eye(a.shape[0], dtype=bool)].reshape(a.shape[0], -1)


def contract(x, c: float = 1):
    """
    Strictly order-preserving function from `[-∞, ∞]` to `[0, 1]`
    that sends `-∞, -c, 0, c, ∞` to `0, 0.25, 0.5, 0.75, 1`, respectively.

    Parameters
    ----------
    x : float
        Input value. Should be in `[-∞, ∞]`.

    c : float = 1
        The secondary 'central' value that is sent to 0.75 (-c is sent to 0.25). Should be in `(0, ∞)`.

    Returns
    -------
    y : float
        Output value in [0, 1].
    """
    y = x/(2*(abs(x) + c)) + 0.5
    y = np.where(np.isneginf(x), 0, y)
    y = np.where(np.isposinf(x), 1, y)
    return y


def shifted_reciprocal(x, c: float = 1):
    """
    Order-reversing function from [0, ∞) to [0, 1] that sends `x` to `1/(1 + x/c)`.
    Strictly order-reversing, but does not preserve absolute differences.

    Parameters
    ----------
    x : float
        Input value. Should be in `[0, ∞)`.

    c : float = 1
        The 'central' value that is sent to 0.5. Should be in `(0, ∞)`.

    Returns
    -------
    y : float
        Output value in [0, 1].
    """
    return 1/(1 + x/c)


def truncated_complement(x):
    """
    Order-reversing function from [0, ∞) to [0, 1] that sends `x` to `max(0, 1 - x)`.
    Preserves absolute differences for values under 1, but discards all differences for larger values.

    Parameters
    ----------
    x : float
        Input value. Should be in `[0, ∞)`.

    Returns
    -------
    y : float
        Output value in [0, 1].
    """
    return np.maximum(0, 1 - x)
