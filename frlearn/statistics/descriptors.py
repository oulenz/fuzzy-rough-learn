"""Statistical data descriptors"""
from __future__ import annotations

import numpy as np
from scipy import linalg
from scipy.stats import chi2

from frlearn.base import Descriptor


class MD(Descriptor):
    """
    Implementation of the Mahalanobis Distance (MD) data descriptor [1]_.
    Mahalanobis distance is the multivariate generalisation of distance to the mean in terms of σ,
    in a Gaussian distribution. This data descriptor simply assumes that the target class is normally distributed,
    and uses the pseudo-inverse of its covariance matrix to transform a vector with deviations from the mean
    in each dimension into a single distance value.
    Squared Mahalanobis distance is χ²-distributed, the corresponding p-value is the confidence score.

    References
    ----------

    .. [1] `Mahalanobis PC (1936).
       On the generalized distance in statistics.
       Proceedings of the National Institute of Sciences of India, vol 2, no 1, pp 49–55.
       <http://insa.nic.in/writereaddata/UpLoadedFiles/PINSA/Vol02_1936_1_Art05.pdf>`_
    """

    def construct(self, X) -> Model:
        model: MD.Model = super().construct(X)
        model.mean = X.mean(axis=0)
        model.covar_inv = linalg.pinvh(np.cov(X.T, bias=True))
        return model

    class Model(Descriptor.Model):

        mean: np.array
        covar_inv: np.array

        def query(self, X):
            d = X - self.mean
            D2 = (d[..., None, :] @ self.covar_inv @ d[..., None]).squeeze(axis=(-2, -1))
            return 1 - chi2.cdf(D2, df=self.m)
