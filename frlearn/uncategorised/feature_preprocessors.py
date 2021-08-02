"""Uncategorised preprocessors"""
from __future__ import annotations

from typing import Callable

import numpy as np

from frlearn.array_functions import div_or
from frlearn.base import FeaturePreprocessor, Unsupervised
from frlearn.uncategorised.utilities import resolve_dissimilarity


class VectorSizeNormaliser(Unsupervised, FeaturePreprocessor):
    """
    Rescales each instance (seen as a vector) to a fixed size.
    Typically used on datasets of frequency counts,
    when only the relative frequencies are considered important,
    e.g. token counts of texts in NLP.

    Parameters
    ----------
    measure: str or float or (np.array -> float) = 'boscovich'
        The vector size measure to use. Must be positively homogeneous.
        A float is interpreted as Minkowski size with the corresponding value for `p`.
        For convenience, a number of popular measures can be referred to by name.

    target_size: float = 0.5
        The size that all vectors will be rescaled to.
        The default, 0.5, ensures that for Minkowski sizes,
        the maximum distance in the resulting dataset is 1.
        A more typical choice is to set this value to 1,
        so that all instances end up on the unit hypersphere.

    Notes
    -----
    If the size of an instance is 0, it will be left unscaled.
    If the size of an instance is ∞, it will be scaled to 0.

    """
    # TODO: this doesn't need to be a ModelFactory

    def __init__(
            self,
            measure: str or float or Callable[[np.array], float] = 'boscovich',
            target_size: float = 0.5,
    ):
        super().__init__()
        # TODO: resolve vector size measures separately
        self.measure = resolve_dissimilarity(measure)
        self.target_size = target_size

    def _construct(self, X, ) -> Model:
        model = super()._construct(X)
        model.measure = self.measure
        model.target_size = self.target_size
        return model

    class Model(Unsupervised.Model, FeaturePreprocessor.Model):

        measure: Callable[[np.array], float]
        target_size: float

        def _query(self, X):
            return self.target_size * div_or(X, self.measure(X)[:, None], X)
