"""Central points of datasets"""

import numpy as np


def centroid(X):
    return np.nanmean(X, axis=0)


def marginal_median(X):
    return np.nanmedian(X, axis=0)
