"""Statistical preprocessors"""
from __future__ import annotations

import numpy as np

from frlearn.array_functions import div_or
from frlearn.base import FeaturePreprocessor, Unsupervised


class NormNormaliser(Unsupervised, FeaturePreprocessor):
    """
    Rescales each instance to unit norm. Typically used on datasets of frequency counts,
    when only the relative frequencies are considered important,
    e.g. token counts of texts in NLP.

    Parameters
    ----------
    p: float = 1
        Order of the norm to use. Can also be `-np.inf` or `np.inf`.

    Notes
    -----
    If the norm of an instance is 0, it will be left unscaled.
    If the norm of an instance is ∞, it will be scaled to 0.

    """

    def __init__(self, p: float = 1, ):
        super().__init__()
        self.p = p

    def _construct(self, X, ) -> Model:
        model = super()._construct(X)
        model.p = self.p
        return model

    class Model(Unsupervised.Model, FeaturePreprocessor.Model):

        p: float

        def _query(self, X):
            if self.p == 0:
                norm = np.where(np.count_nonzero(X, axis=-1) <= 1, np.sum(np.abs(X), axis=-1), np.inf)
            else:
                norm = np.linalg.norm(X, ord=self.p, axis=-1)
            return div_or(X, norm[:, None], X)
