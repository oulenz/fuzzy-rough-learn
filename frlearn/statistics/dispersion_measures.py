"""Measures of dispersion of datasets"""

import numpy as np

__all__ = [
    'interquartile_range', 'maximum_absolute_value', 'standard_deviation', 'total_range',
]


def interquartile_range(X):
    """
    Distance between the first and the third quartile; range of the central half of the data.

    Parameters
    ----------
    X : np.array
        Dataset. Should be a two-dimensional array.

    Returns
    -------
    a: np.array
        One-dimensional array that contains the interquartile range for each feature.
    """
    quartiles = np.nanpercentile(X, [25, 75], axis=0)
    return quartiles[1, :] - quartiles[0, :]


def maximum_absolute_value(X):
    """
    Maximum distance from 0.

    Parameters
    ----------
    X : np.array
        Dataset. Should be a two-dimensional array.

    Returns
    -------
    a: np.array
        One-dimensional array that contains the maximum absolute value for each feature.
    """
    return np.nanmax(np.abs(X), axis=0)


def standard_deviation(X):
    """
    Square root of the sum of the squared distances to the mean.

    Parameters
    ----------
    X : np.array
        Dataset. Should be a two-dimensional array.

    Returns
    -------
    a: np.array
        One-dimensional array that contains the standard deviation for each feature.
    """
    return np.nanstd(X, axis=0)


def total_range(X):
    """
    Distance between the smallest and largest value.

    Parameters
    ----------
    X : np.array
        Dataset. Should be a two-dimensional array.

    Returns
    -------
    a: np.array
        One-dimensional array that contains the ranges for each feature.

    Notes
    -----
    This function is not called `range` so as not to overwrite the built-in function `range`.
    """
    return np.nanmax(X, axis=0) - np.nanmin(X, axis=0)
