from inspect import signature, Parameter

import numpy as np
from scipy.spatial.distance import cdist

from frlearn.vector_size_measures import MinkowskiSize


def resolve_dissimilarity(dissimilarity, scale_by_dimensionality=False):
    """
    Resolves a dissimilarity or vector size measure from a string or a float.
    Passes through values that are already callable.
    Raises an exception for invalid values.

    Floats are resolved as Minkowski size with corresponding value for `p`.

    Parameters
    ----------
    dissimilarity: callable or str or float
        The input value to resolve.

    scale_by_dimensionality: bool=False
        Option passed through when initialising the dissimilarity measures returned
        from a string or float input.

    Returns
    -------
    measure: (np.array -> float) or ((np.array, np.array) -> float)
        The resolved dissimilarity or vector size measure.
        Either a callable that takes two vectors `x` and `y` and returns their dissimilarity,
        or a callable that takes a single vector and returns its size,
        which induces a dissimilarity measure through application to `y - x`.

    Raises
    ------
    ValueError if the input value is not a string, float, or callable,
    or if the input value is an unrecognised string.
    """
    if callable(dissimilarity):
        return dissimilarity
    if isinstance(dissimilarity, str):
        dissimilarity = dissimilarity.lower()
        if dissimilarity in ['hamming', ]:
            p = 0
            unrooted = True
        elif dissimilarity in ['boscovich', 'cityblock', 'manhattan', 'taxicab', ]:
            p = 1
            unrooted = False
        elif dissimilarity in ['euclidean', 'pythagorean', ]:
            p = 2
            unrooted = False
        elif dissimilarity in ['squared_euclidean', 'squared_pythagorean', ]:
            p = 2
            unrooted = True
        elif dissimilarity in ['chebyshev', 'chessboard', 'maximum']:
            p = np.inf
            unrooted = False
        else:
            raise ValueError(f'Unknown dissimilarity measure: \'{dissimilarity}\'')
        return MinkowskiSize(p=p, unrooted=unrooted, scale_by_dimensionality=scale_by_dimensionality)
    if isinstance(dissimilarity, (int, float)):
        return MinkowskiSize(p=dissimilarity, unrooted=False, scale_by_dimensionality=scale_by_dimensionality)
    raise ValueError(f'Parameter `dissimilarity` must be a function or a callable class, a string, or a float.')


def apply_dissimilarity(u, v, measure):
    """
    Calculates the dissimilarity of `u` and `v`,
    or each pair of vectors if `u` and/or `v` is a two-dimensional array.

    Parameters
    ----------
    u: np.array
        Array of 1 or 2 dimensions, corresponding to a single vector or a collection of vectors.
        The length of the vectors in `u` and `v` should match.

    v: np.array
        Array of 1 or 2 dimensions, corresponding to a single vector or a collection of vectors.
        The length of the vectors in `u` and `v` should match.

    measure: (np.array -> float) or ((np.array, np.array) -> float)
        The dissimilarity or vector size measure to apply.
        If this is a vector size measure `np.array -> float` (like a norm),
        it is applied to the difference `v - u`.

    Returns
    -------
    dissimilarity: np.array
        The dissimilarity of `u` and `v`.
        If `u` and `v` are vectors, this is a single value.
        Otherwise, it is an array corresponding to the first dimensions of `u` and/or `v`.

    """
    assert (len(u.shape), len(v.shape) <= 2) and u.shape[-1] == v.shape[-1],\
        'Arrays should not be more than two-dimensional, and the size of their last dimension should match.'
    new_shape = u.shape[:-1] + v.shape[:-1]
    params = signature(measure.__call__).parameters.values()
    num_args = len([p for p in params if p.default == Parameter.empty])
    if num_args == 1:
        return np.reshape(measure(np.atleast_2d(v) - np.atleast_2d(u)[:, None, :]), new_shape)
    if num_args == 2:
        return np.reshape(cdist(np.atleast_2d(u), np.atleast_2d(v), measure), new_shape)
    raise ValueError('Measure should be a callable with one or two arguments without default values.')
