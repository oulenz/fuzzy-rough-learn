"""Uncategorised preprocessors"""
from __future__ import annotations

from typing import Callable

import numpy as np

from frlearn.array_functions import div_or
from frlearn.base import FeaturePreprocessor, Unsupervised
from frlearn.uncategorised.utilities import resolve_dissimilarity


class VectorSizeNormaliser(Unsupervised, FeaturePreprocessor):
    """
    Rescales each instance (seen as a vector) to size 1.
    Typically used on datasets of frequency counts,
    when only the relative frequencies are considered important,
    e.g. token counts of texts in NLP.

    Parameters
    ----------
    measure: str or float or (np.array -> float) = 'boscovich'
        The vector size measure to use.
        A float is interpreted as Minkowski size with the corresponding value for `p`.
        For convenience, a number of popular measures can be referred to by name.

    Notes
    -----
    If the size of an instance is 0, it will be left unscaled.
    If the size of an instance is ∞, it will be scaled to 0.

    """
    # TODO: this doesn't need to be a ModelFactory

    def __init__(self, measure: str or float or Callable[[np.array], float] = 'boscovich', ):
        super().__init__()
        # TODO: resolve vector size measures separately
        self.measure = resolve_dissimilarity(measure)

    def _construct(self, X, ) -> Model:
        model = super()._construct(X)
        model.measure = self.measure
        return model

    class Model(Unsupervised.Model, FeaturePreprocessor.Model):

        measure: Callable[[np.array], float]

        def _query(self, X):
            return div_or(X, self.measure(X)[:, None], X)
