"""Utility functions for numpy"""

import numpy as np


def argmax_and_max(a, axis):
    ai = np.argmax(a, axis=axis)
    av = np.squeeze(np.take_along_axis(a, np.expand_dims(ai, axis=axis), axis=axis), axis=axis)
    return ai, av


def first(a, k, axis=-1):
    """
    Returns the `k` first values of `a` along the specified axis.
    If `k` is in `(0, 1)`, it is interpreted as a fraction of the total number, with a minimum of 1.
    If `k` is None, all values of `a` will be returned.

    Parameters
    ----------
    a : ndarray
        Input array of values.

    k : int or float or None
        Number of values to return.
        If a float in (0, 1), taken as a fraction of the total number of values.
        If None, all values are returned.

    axis : int, default=-1
        The axis along which values are selected.

    Returns
    -------
    first_along_axis : ndarray
        An array with the same shape as `a`, with the specified axis reduced according to the value of `k`.
    """
    if not k:
        return a
    if 0 < k < 1:
        k = max(int(k * a.shape[axis]), 1)
    slc = [slice(None)] * len(a.shape)
    slc[axis] = slice(0, k)
    return a[tuple(slc)]


def last(a, k, axis=-1):
    """
    Returns the `k` last values of `a` along the specified axis, in reverse order.
    If `k` is in `(0, 1)`, it is interpreted as a fraction of the total number, with a minimum of 1.
    If `k` is None, all values of `a` will be returned.

    Parameters
    ----------
    a : ndarray
        Input array of values.

    k : int or float or None
        Number of values to return.
        If a float in (0, 1), taken as a fraction of the total number of values.
        If None, all values are returned.

    axis : int, default=-1
        The axis along which values are selected.

    Returns
    -------
    last_along_axis : ndarray
        An array with the same shape as `a`, with the specified axis reduced according to the value of `k`.
    """
    if not k:
        return np.flip(a, axis=axis)
    if 0 < k < 1:
        k = max(int(k * a.shape[axis]), 1)
    slc = [slice(None)] * len(a.shape)
    slc[axis] = slice(-1, -k - 1, -1)
    return a[tuple(slc)]


def least(a, k, axis=-1):
    """
    Returns the `k` least values of `a` along the specified axis, in order.
    If `k` is in `(0, 1)`, it is interpreted as a fraction of the total number, with a minimum of 1.
    If `k` is None, all values of `a` will be returned.

    Parameters
    ----------
    a : ndarray
        Input array of values.

    k : int or float or None
        Number of values to return.
        If a float in (0, 1), taken as a fraction of the total number of values.
        If None, all values are returned.

    axis : int, default=-1
        The axis along which values are selected.

    Returns
    -------
    least_along_axis : ndarray
        An array with the same shape as `a`, with the specified axis reduced according to the value of `k`.
    """
    if not k:
        return np.sort(a, axis=axis)
    if 0 < k < 1:
        k = max(int(k * a.shape[axis]), 1)
    a = np.partition(a, k - 1, axis=axis)
    take_this = np.arange(k)
    a = np.take(a, take_this, axis=axis)
    a = np.sort(a, axis=axis)
    return a


def greatest(a, k, axis=-1):
    """
    Returns the `k` greatest values of `a` along the specified axis, in order.
    If `k` is in `(0, 1)`, it is interpreted as a fraction of the total number, with a minimum of 1.
    If `k` is None, all values of `a` will be returned.

    Parameters
    ----------
    a : ndarray
        Input array of values.

    k : int or float or None
        Number of values to return.
        If a float in (0, 1), taken as a fraction of the total number of values.
        If None, all values are returned.

    axis : int, default=-1
        The axis along which values are selected.

    Returns
    -------
    greatest_along_axis : ndarray
        An array with the same shape as `a`, with the specified axis reduced according to the value of `k`.
    """
    if not k:
        return np.flip(np.sort(a, axis=axis), axis=axis)
    if 0 < k < 1:
        k = max(int(k * a.shape[axis]), 1)
    a = np.partition(a, -k, axis=axis)
    take_this = np.arange(-k % a.shape[axis], a.shape[axis])
    a = np.take(a, take_this, axis=axis)
    a = np.flip(np.sort(a, axis=axis), axis=axis)
    return a


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
