"""Statistical data descriptors"""
from __future__ import annotations

import numpy as np
from scipy import linalg, spatial
from scipy.stats import chi2

from .centres import centroid
from ..base import Descriptor
from ..utils.np_utils import shifted_reciprocal


class CD(Descriptor):
    """
    Implementation of the Centre Distance (CD) data descriptor.
    Calculates a score based on the distance to a central point of the target data.

    Parameters
    ----------
    centre : ndarray -> ndarray, default=centroid
        Central point definition to use.
        Should be a function that takes an array of shape (n, m, )
        and returns an array of shape (m, )
    metric: str, default='euclidean'
        Metric to use for distance calculations.
    threshold_perc : int or None, default=80
        Threshold percentile for normal instances. Should be in (0, 100] or None.
        All distances below the distance value in the target set corresponding to this percentile
        result in a final score above 0.5. If None, 1 is used as the threshold instead.
    """

    def __init__(self, centre=centroid, metric: str = 'euclidean', threshold_perc: int | None = 80):
        self.centre = centre
        self.metric = metric
        self.threshold_perc = threshold_perc

    def construct(self, X) -> Model:
        model: CD.Model = super().construct(X)
        model.centre = self.centre(X)
        model.metric = self.metric
        if self.threshold_perc:
            distances = model._distances(X)
            model.threshold = np.percentile(distances, self.threshold_perc)
        else:
            model.threshold = 1
        return model

    class Model(Descriptor.Model):

        centre: np.ndarray
        metric: str
        threshold: float

        def query(self, X):
            distances = self._distances(X)
            return shifted_reciprocal(distances, self.threshold)

        def _distances(self, X):
            # cdist expects two two-dimensional arrays, and returns the two-dimensional array of pairwise distances
            return np.squeeze(spatial.distance.cdist(X, self.centre[None, :], metric=self.metric), axis=1)


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
