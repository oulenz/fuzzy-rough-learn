from __future__ import annotations

import numpy as np
from scipy.spatial.distance import cdist

from frlearn.base import Regressor
from frlearn.feature_preprocessors import RangeNormaliser
from frlearn.neighbours.utilities import resolve_k


class FRNN(Regressor):
    """
    Implementation of the Fuzzy Rough Nearest Neighbour (FRNN) regressor [1]_.

    Predicts an output value of a test instance `y` on the basis of the output values
    of the `k` nearest neighbours of `y`, similar to KNN regression.
    The difference is that the output value is calculated as a weighted mean,
    with the weights corresponding to membership degrees in the upper and lower approximations
    of the tolerance sets of the output values of the neighbours.

    Parameters
    ----------
    k: int or (int -> float) = 10
        Number of neighbours to consider.
        Should be either a positive integer,
        or a function that takes the training set size `n` and returns a float.
        All such values are rounded to the nearest integer in `[1, n]`.
        Due to the computational complexity of this algorithm,
        `k` should not be chosen too large.

    dissimilarity: str = 'chebyshev'
        The dissimilarity measure to use.

    preprocessors : iterable = (RangeNormaliser(), )
        Preprocessors to apply. The default range normaliser ensures that all features have range 1.,
        To simulate a tolerance relation R that is the mean of the per-atribute tolerance relations,
        `dissimilarity` should be set to `'manhattan'` and `RangeNormaliser(normalise_dimensionality=True)`
        should be used as preprocessor, to ensure that the ranges sum to 1.

    Notes
    -----
    Although proposed in the same paper [1]_, FRNN regression and FRNN classification are different algorithms.

    [1]_ does not recommend any specific value for `k`, but seems to use `k = 10` for its experiments.

    References
    ----------
    .. [1] `Jensen R, Cornelis C (2011).
       Fuzzy-rough nearest neighbour classification and prediction.
       Theoretical Computer Science, vol 412, pp 5871–5884.
       doi: 10.1016/j.tcs.2011.05.040
       <https://www.sciencedirect.com/science/article/pii/S0304397511004580>`_
    """
    def __init__(
            self,
            k: int = 10,
            dissimilarity: str = 'chebyshev',
            preprocessors=(RangeNormaliser(), )
    ):
        super().__init__(preprocessors=preprocessors)
        self.k = k
        self.dissimilarity = dissimilarity

    def _construct(self, X, y) -> Model:
        model: FRNN.Model = super()._construct(X, y)
        model.k = resolve_k(self.k, model.n)
        model.dissimilarity = self.dissimilarity
        model.X = X
        model.y_range = np.max(y) - np.min(y)
        model.y = y
        return model

    class Model(Regressor.Model):

        k: int
        dissimilarity: str
        X: np.array
        y: np.array
        y_range: float

        def _query(self, X):
            distances = cdist(X, self.X, self.dissimilarity)
            neighbour_indices = np.argpartition(distances, kth=self.k - 1, axis=-1)[:, :self.k]
            neighbour_vals = self.y[neighbour_indices]
            neighbour_vals_sims = 1 - np.abs((neighbour_vals/self.y_range)[..., None] - self.y / self.y_range)
            lower_approx_vals = np.min(np.maximum(neighbour_vals_sims, distances[:, None, :]), axis=-1)
            upper_approx_vals = np.max(np.minimum(neighbour_vals_sims, 1 - distances[:, None, :]), axis=-1)
            combined_vals = (lower_approx_vals + upper_approx_vals)/2
            return np.sum(combined_vals*neighbour_vals, axis=-1)/np.sum(combined_vals, axis=-1)
