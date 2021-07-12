"""Measures of location of datasets"""

import numpy as np

__all__ = [
    'maximum', 'mean', 'median', 'midhinge', 'midrange', 'minimum',
]


def maximum(X):
    """
    Greatest value.

    Parameters
    ----------
    X : np.array
        Dataset. Should be a two-dimensional array.

    Returns
    -------
    a: np.array
        One-dimensional array that contains the maximum for each feature.
    """
    return np.nanmax(X, axis=0)


def mean(X):
    """
    Sum of all values divided by the number of values.

    Parameters
    ----------
    X : np.array
        Dataset. Should be a two-dimensional array.

    Returns
    -------
    a: np.array
        One-dimensional array that contains the mean for each feature.
    """
    return np.nanmean(X, axis=0)


def median(X):
    """
    Middle value after sorting all values by size, or mean of the two middle values.

    Parameters
    ----------
    X : np.array
        Dataset. Should be a two-dimensional array.

    Returns
    -------
    a: np.array
        One-dimensional array that contains the median for each feature.
    """
    return np.nanmedian(X, axis=0)


def midhinge(X):
    """
    Mean of the first and third quartiles. Midpoint of the interquartile range.

    Parameters
    ----------
    X : np.array
        Dataset. Should be a two-dimensional array.

    Returns
    -------
    a: np.array
        One-dimensional array that contains the midhinge for each feature.
    """
    quartiles = np.nanpercentile(X, [25, 75], axis=0)
    return np.nansum(quartiles, axis=0)/2


def midrange(X):
    """
    Mean of the minimum and maximum. Midpoint of the range.

    Parameters
    ----------
    X : np.array
        Dataset. Should be a two-dimensional array.

    Returns
    -------
    a: np.array
        One-dimensional array that contains the midrange for each feature.
    """
    return (np.nanmax(X, axis=0) + np.nanmin(X, axis=0))/2


def minimum(X):
    """
    Least value.

    Parameters
    ----------
    X : np.array
        Dataset. Should be a two-dimensional array.

    Returns
    -------
    a: np.array
        One-dimensional array that contains the minimum for each feature.
    """
    return np.nanmin(X, axis=0)
