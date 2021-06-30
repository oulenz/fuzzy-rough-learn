"""Measures of dispersion of datasets"""

import numpy as np


def interquartile_range(X):
    quartiles = np.nanpercentile(X, [25, 75], axis=0)
    return quartiles[1, :] - quartiles[0, :]


def maximum_absolute_value(X):
    return np.nanmax(np.abs(X), axis=0)


def standard_deviation(X):
    return np.nanstd(X, axis=0)


def total_range(X):
    return np.nanmax(X, axis=0) - np.nanmin(X, axis=0)
