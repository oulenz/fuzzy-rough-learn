from __future__ import annotations

from typing import Callable

import numpy as np

from frlearn.base import Regressor
from frlearn.feature_preprocessors import RangeNormaliser
from frlearn.neighbours.utilities import resolve_k
from frlearn.parametrisations import at_most
from frlearn.uncategorised.utilities import apply_dissimilarity, resolve_dissimilarity


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
    k: int or (int -> float) = at_most(10)
        Number of neighbours to consider.
        Should be either a positive integer,
        or a function that takes the training set size `n` and returns a float.
        All such values are rounded to the nearest integer in `[1, n]`.
        Due to the computational complexity of this algorithm,
        `k` should not be chosen too large.

    dissimilarity: str or float or (np.array -> float) or ((np.array, np.array) -> float) = 'chebyshev'
        The dissimilarity measure to use.
        The similarity between two instances is calculated as 1 minus their dissimilarity.

        A vector size measure `np.array -> float` induces a dissimilarity measure through application to `y - x`.
        A float is interpreted as Minkowski size with the corresponding value for `p`.
        For convenience, a number of popular measures can be referred to by name.

        When a float or string is passed, the corresponding dissimilarity measure is automatically scaled
        to ensure that the dissimilarity of `[1, 1, ..., 1]` with `[0, 0, ..., 0]` is 1.

        For the default Chebyshev norm, this is already the case,
        since it assigns the maximum of the per-attribute differences,
        but e.g. the Boscovich norm normally amounts to the sum of the per-attribute differences.
        In this case, the scaling step divides by the number of dimensions,
        and we obtain a dissimilarity that is the mean of the per-attribute differences.

        This can be prevented by explicitly passing a dissimilarity measure without scaling.

    preprocessors : iterable = (RangeNormaliser(), )
        Preprocessors to apply. The default range normaliser ensures that all features have range 1.

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
            k: int = at_most(10),
            dissimilarity: str or float or Callable[[np.array], float] or Callable[[np.array, np.array], float] = 'chebyshev',
            preprocessors=(RangeNormaliser(), )
    ):
        super().__init__(preprocessors=preprocessors)
        self.k = k
        self.dissimilarity = resolve_dissimilarity(dissimilarity, scale_by_dimensionality=True)

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
        dissimilarity: Callable[[np.array], float] or Callable[[np.array, np.array], float]
        X: np.array
        y: np.array
        y_range: float

        def _query(self, X):
            #distances = cdist(X, self.X, self.dissimilarity)
            distances = apply_dissimilarity(X, self.X, self.dissimilarity)
            neighbour_indices = np.argpartition(distances, kth=self.k - 1, axis=-1)[:, :self.k]
            neighbour_vals = self.y[neighbour_indices]
            neighbour_vals_sims = 1 - np.abs((neighbour_vals/self.y_range)[..., None] - self.y / self.y_range)
            lower_approx_vals = np.min(np.maximum(neighbour_vals_sims, distances[:, None, :]), axis=-1)
            upper_approx_vals = np.max(np.minimum(neighbour_vals_sims, 1 - distances[:, None, :]), axis=-1)
            combined_vals = (lower_approx_vals + upper_approx_vals)/2
            return np.sum(combined_vals*neighbour_vals, axis=-1)/np.sum(combined_vals, axis=-1)
