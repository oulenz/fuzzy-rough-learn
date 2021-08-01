"""Nearest neighbour feature preprocessors"""
from __future__ import annotations

from typing import Callable

import numpy as np

from frlearn.base import FeatureSelector, ClassSupervised
from frlearn.statistics.feature_preprocessors import Standardiser
from frlearn.array_functions import soft_min
from frlearn.t_norms import lukasiewicz_t_norm
from frlearn.uncategorised.quantifiers import QuadraticSigmoid
from frlearn.weights import QuantifierWeights


class FRFS(ClassSupervised, FeatureSelector):
    """
    Implementation of the Fuzzy Rough Feature Selection (FRFS) preprocessor.

    Greedily selects features that induce the greatest increase in the size of the positive region,
    until it matches the size of the positive region with all features,
    or until the required number of features is selected.

    The positive region is defined as the union of the lower approximations of the decision classes in `X`.
    Its size is the sum of its membership values.

    Parameters
    ----------
    n_features : int or None, default=None
        Number of features to select. If None, will continue to add features until positive region size becomes maximal.

    owa_weights: (int -> np.array) = QuantifierWeights(QuadraticSigmoid(0.2, 1))
        OWA weights to use for calculation of the soft minimum in the positive regions.

    t_norm : (ndarray, int, ) -> ndarray, default=lukasiewicz_t_norm
        Function that takes an ndarray and a keyword argument `axis`,
        and returns an ndarray with the corresponding axis removed.
        Used to define the similarity relation `R` from the per-attribute similarities.
        Should be a t-norm, or else the size of the positive region may decrease as features are added.


    References
    ----------
    .. [1] `Cornelis C, Verbiest N, Jensen R (2011).
       Ordered Weighted Average Based Fuzzy Rough Sets
       In: Yu J, Greco S, Lingras P, Wang G, Skowron A (eds). Rough Set and Knowledge Technology. RSKT 2010.
       Lecture Notes in Computer Science, vol 6401. Springer, Berlin, Heidelberg.
       doi: 10.1007/978-3-642-16248-0_16
       <https://link.springer.com/chapter/10.1007/978-3-642-16248-0_16>`_
    """

    def __init__(
            self, n_features=None,
            owa_weights: Callable[[int], np.array] = QuantifierWeights(QuadraticSigmoid(0.2, 1)),
            t_norm=lukasiewicz_t_norm,
    ):
        super().__init__()
        self.n_features = n_features
        self.owa_weights = owa_weights
        self.t_norm = t_norm

    def _construct(self, X, y):
        model = super()._construct(X, y)
        X_scaled = Standardiser()(X)(X)
        R_a = np.minimum(np.maximum(1 - np.abs(X_scaled[:, None, :] - X_scaled), 0), y[:, None, None] != y[:, None])
        POS_A_size = self._POS_size(R_a)
        selected_attributes = np.full(X.shape[-1], False)
        remaining_attributes = set(range(X.shape[-1]))
        best_size = 0
        condition = (lambda: np.sum(selected_attributes) < self.n_features) if self.n_features else (lambda: best_size < POS_A_size)
        while condition():
            best_size = 0
            for i in remaining_attributes:
                candidate = selected_attributes.copy()
                candidate[i] = True
                candidate_size = self._POS_size(R_a[..., candidate])
                if candidate_size > best_size:
                    best_size = candidate_size
                    new_attribute = i
            selected_attributes[new_attribute] = True
            remaining_attributes.remove(new_attribute)
        model.selection = selected_attributes
        return model

    def _POS_size(self, R_a):
        R = self.t_norm(R_a, axis=-1)
        return np.sum(soft_min(1 - R, self.owa_weights, k=None, axis=-1))

    class Model(ClassSupervised.Model, FeatureSelector.Model):
        pass
