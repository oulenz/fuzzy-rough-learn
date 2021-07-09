"""Measures of location of datasets"""

import numpy as np

__all__ = [
    'maximum', 'mean', 'median', 'midhinge', 'midrange', 'minimum',
]


def maximum(X):
    return np.nanmax(X, axis=0)


def mean(X):
    return np.nanmean(X, axis=0)


def median(X):
    return np.nanmedian(X, axis=0)


def midhinge(X):
    quartiles = np.nanpercentile(X, [25, 75], axis=0)
    return np.nansum(quartiles, axis=0)/2


def midrange(X):
    return (np.nanmax(X, axis=0) + np.nanmin(X, axis=0))/2


def minimum(X):
    return np.nanmin(X, axis=0)
