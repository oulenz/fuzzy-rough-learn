"""Utility functions for numpy"""

import numpy as np

__all__ = [
    'div_or', 'first', 'greatest', 'last','least', 'remove_diagonal', 'soft_head', 'soft_max', 'soft_min', 'soft_tail',
]


def argmax_and_max(a, axis):
    ai = np.argmax(a, axis=axis)
    av = np.squeeze(np.take_along_axis(a, np.expand_dims(ai, axis=axis), axis=axis), axis=axis)
    return ai, av


def div_or(x: np.array or float, y: np.array or float, fallback: np.array or float = np.nan):
    """
    Divides `x` by `y`, replacing `np.nan` values with `fallback`.

    Parameters
    ----------
    x: np.array or float
        Dividend.

    y: np.array or float
        Divisor.

    fallback: np.array or float, default=np.nan
        Fallback value(s) to substitute for `np.nan` after division.

    Returns
    -------
    z: np.array
        Quotient.

    Notes
    -----
    `x`, `y` and `fallback` should be broadcastable to a single shape.
    """
    with np.errstate(divide='ignore', invalid='ignore'):
        z = x / y
        z = np.where(np.isnan(z), fallback, z)
    return z


def first(a, k: int, axis: int = -1):
    """
    Returns the `k` first values of `a` along the specified axis.

    Parameters
    ----------
    a : ndarray
        Input array of values.

    k: int
        Number of values to return.
        Should be a positive integer not larger than `a` along `axis`.

    axis : int, default=-1
        The axis along which values are selected.

    Returns
    -------
    first_along_axis : ndarray
        An array with the same shape as `a`, with the specified axis reduced according to the value of `k`.
    """
    if k == a.shape[axis]:
        return a
    slc = [slice(None)] * len(a.shape)
    slc[axis] = slice(0, k)
    return a[tuple(slc)]


def greatest(a, k: int, axis: int = -1):
    """
    Returns the `k` greatest values of `a` along the specified axis, in order.

    Parameters
    ----------
    a : ndarray
        Input array of values.

    k: int
        Number of values to return.
        Should be a positive integer not larger than `a` along `axis`.

    axis : int, default=-1
        The axis along which values are selected.

    Returns
    -------
    greatest_along_axis : ndarray
        An array with the same shape as `a`, with the specified axis reduced according to the value of `k`.
    """
    if k == a.shape[axis]:
        return np.flip(np.sort(a, axis=axis), axis=axis)
    a = np.partition(a, -k, axis=axis)
    take_this = np.arange(-k % a.shape[axis], a.shape[axis])
    a = np.take(a, take_this, axis=axis)
    a = np.flip(np.sort(a, axis=axis), axis=axis)
    return a


def last(a, k: int, axis: int = -1):
    """
    Returns the `k` last values of `a` along the specified axis, in reverse order.

    Parameters
    ----------
    a : ndarray
        Input array of values.

    k: int
        Number of values to return.
        Should be a positive integer not larger than `a` along `axis`.

    axis : int, default=-1
        The axis along which values are selected.

    Returns
    -------
    last_along_axis : ndarray
        An array with the same shape as `a`, with the specified axis reduced according to the value of `k`.
    """
    if k == a.shape[axis]:
        return np.flip(a, axis=axis)
    slc = [slice(None)] * len(a.shape)
    slc[axis] = slice(-1, -k - 1, -1)
    return a[tuple(slc)]


def least(a, k: int, axis: int = -1):
    """
    Returns the `k` least values of `a` along the specified axis, in order.

    Parameters
    ----------
    a : ndarray
        Input array of values.

    k: int
        Number of values to return.
        Should be a positive integer not larger than `a` along `axis`.

    axis : int, default=-1
        The axis along which values are selected.

    Returns
    -------
    least_along_axis : ndarray
        An array with the same shape as `a`, with the specified axis reduced according to the value of `k`.
    """
    if k == a.shape[axis]:
        return np.sort(a, axis=axis)
    a = np.partition(a, k - 1, axis=axis)
    take_this = np.arange(k)
    a = np.take(a, take_this, axis=axis)
    a = np.sort(a, axis=axis)
    return a


def remove_diagonal(a):
    # TODO: parametrise dimensions
    """
    Remove the diagonal from a square array.

    Parameters
    ----------
    a: np.array
        Input array of values. Should have shape `(n, n)`.

    Returns
    -------
    b: np.array
        An array of shape `(n, n-1)`, containing the same values as the input array,
        except for the values in the diagonal.
    """
    return a[~np.eye(a.shape[0], dtype=bool)].reshape(a.shape[0], -1)


def soft_head(a, weights, k: int or None, axis=-1, type: str = 'arithmetic'):
    r"""
    Calculates the soft head of an array.

    Parameters
    ----------
    a : ndarray
        Input array of values.

    weights : (k -> np.array) or None
        Weights to apply to the `k` selected values. If None, the `k`\ th value is returned.

    k: int or None
        Number of initial values from which the soft head is calculated.
        Should be either a positive integer not larger than `a` along `axis`,
        or None, which is interpreted as the size of `a` along `axis`.

    axis : int, default=-1
        The axis along which the soft head is calculated.

    type : str {'arithmetic', 'geometric', 'harmonic', }, default='arithmetic'
        Determines the type of weighted average.

    Returns
    -------
    soft_head_along_axis : ndarray
        An array with the same shape as `a`, with the specified
        axis removed. If `a` is a 0-d array, a scalar is returned.
    """
    if k is None:
        k = a.shape[axis]
    a = first(a, k, axis=axis)
    return _weighted_mean(a, weights, axis=axis, type=type)


def soft_max(a, weights, k: int or None, axis=-1, type: str = 'arithmetic'):
    r"""
    Calculates the soft maximum of an array.

    Parameters
    ----------
    a : ndarray
        Input array of values.

    weights : (k -> np.array) or None
        Weights to apply to the `k` selected values. If None, the `k`\ th value is returned.

    k: int or None
        Number of greatest values from which the soft maximum is calculated.
        Should be either a positive integer not larger than `a` along `axis`,
        or None, which is interpreted as the size of `a` along `axis`.

    axis : int, default=-1
        The axis along which the soft maximum is calculated.

    type : str {'arithmetic', 'geometric', 'harmonic', }, default='arithmetic'
        Determines the type of weighted average.

    Returns
    -------
    soft_max_along_axis : ndarray
        An array with the same shape as `a`, with the specified
        axis removed. If `a` is a 0-d array, a scalar is returned.
    """
    if k is None:
        k = a.shape[axis]
    a = greatest(a, k, axis=axis)
    return _weighted_mean(a, weights, axis=axis, type=type)


def soft_min(a, weights, k: int or None, axis=-1, type: str = 'arithmetic'):
    r"""
    Calculates the soft minimum of an array.

    Parameters
    ----------
    a : ndarray
        Input array of values.

    weights : (k -> np.array) or None
        Weights to apply to the `k` selected values. If None, the `k`\ th value is returned.

    k: int or None
        Number of least values from which the soft minimum is calculated.
        Should be either a positive integer not larger than `a` along `axis`,
        or None, which is interpreted as the size of `a` along `axis`.

    axis : int, default=-1
        The axis along which the soft minimum is calculated.

    type : str {'arithmetic', 'geometric', 'harmonic', }, default='arithmetic'
        Determines the type of weighted average.

    Returns
    -------
    soft_min_along_axis : ndarray
        An array with the same shape as `a`, with the specified
        axis removed. If `a` is a 0-d array, a scalar is returned.
    """
    if k is None:
        k = a.shape[axis]
    a = least(a, k, axis=axis)
    return _weighted_mean(a, weights, axis=axis, type=type)


def soft_tail(a, weights, k: int or None, axis=-1, type: str = 'arithmetic'):
    r"""
    Calculates the soft tail of an array.

    Parameters
    ----------
    a : ndarray
        Input array of values.

    weights : (k -> np.array) or None
        Weights to apply to the `k` selected values. If None, the `k`\ th value is returned.

    k: int or None
        Number of terminal values from which the soft tail is calculated.
        Should be either a positive integer not larger than `a` along `axis`,
        or None, which is interpreted as the size of `a` along `axis`.

    axis : int, default=-1
        The axis along which the soft tail is calculated.

    type : str {'arithmetic', 'geometric', 'harmonic', }, default='arithmetic'
        Determines the type of weighted average.

    Returns
    -------
    soft_tail_along_axis : ndarray
        An array with the same shape as `a`, with the specified
        axis removed. If `a` is a 0-d array, a scalar is returned.
    """
    if k is None:
        k = a.shape[axis]
    a = last(a, k, axis=axis)
    return _weighted_mean(a, weights, axis=axis, type=type)


def _weighted_mean(a, weights, axis, type):
    if weights is None:
        return np.take(a, -1, axis=axis)
    w = weights(a.shape[axis])
    w = np.reshape(w, [-1] + ((len(a.shape) - axis - 1) % len(a.shape)) * [1])
    if type == 'arithmetic':
        return np.sum(w * a, axis=axis)
    if type == 'geometric':
        return np.exp(np.sum(w * np.log(a), axis=axis))
    if type == 'harmonic':
        return 1 / np.sum(w / a, axis=axis)
