"""Statistical data descriptors"""
from __future__ import annotations

from typing import Callable

import numpy as np
from scipy import linalg
from scipy.stats import chi2

from frlearn.base import DataDescriptor
from frlearn.feature_preprocessors import Standardiser
from frlearn.transformations import shifted_reciprocal
from frlearn.uncategorised.utilities import resolve_dissimilarity


class CD(DataDescriptor):
    """
    Implementation of the Centre Distance (CD) data descriptor.
    Calculates a score based on the distance to a central point of the target data.

    This is implemented simply as the vector size of each element (the distance to the origin),
    with the expectation that the given preprocessor normalises the data in such a way that
    a suitable central value of the data is located at the origin,
    and that all features have the same scale.
    The drawback of this approach is that it does not allow
    dissimilarity measures to be used that are not induced by a vector size measure.

    By default (standardisation) this is euclidean centroid distance.

    Parameters
    ----------
    measure: str or float or (np.array -> float) = 'euclidean'
        The vector size measure to use.
        A float is interpreted as Minkowski size with the corresponding value for `p`.
        For convenience, a number of popular measures can be referred to by name.

    threshold_perc : int or None, default=80
        Threshold percentile for normal instances. Should be in (0, 100] or None.
        All distances below the distance value in the target set corresponding to this percentile
        result in a final score above 0.5. If None, 1 is used as the threshold instead.

    preprocessors : iterable = (Standardiser(), )
        Preprocessors to apply. The default standardiser places the centroid of the data at the origin,
        and ensures that all features have the same standard deviation.
    """

    def __init__(
            self,
            measure: str or float or Callable[[np.array], float] = 'euclidean',
            threshold_perc: int or None = 80,
            preprocessors=(Standardiser(), )
    ):
        super().__init__(preprocessors=preprocessors)
        # TODO: resolve vector size measures separately
        self.measure = resolve_dissimilarity(measure)
        self.threshold_perc = threshold_perc

    def _construct(self, X) -> Model:
        model: CD.Model = super()._construct(X)
        model.measure = self.measure
        if self.threshold_perc:
            distances = model.measure(X)
            model.threshold = np.percentile(distances, self.threshold_perc)
        else:
            model.threshold = 1
        return model

    class Model(DataDescriptor.Model):

        measure: Callable[[np.array], float]
        threshold: float

        def _query(self, X):
            distances = self.measure(X)
            return shifted_reciprocal(distances, self.threshold)


class MD(DataDescriptor):
    """
    Implementation of the Mahalanobis Distance (MD) data descriptor [1]_.
    Mahalanobis distance is the multivariate generalisation of distance to the mean in terms of σ,
    in a Gaussian distribution. This data descriptor simply assumes that the target class is normally distributed,
    and uses the pseudo-inverse of its covariance matrix to transform a vector with deviations from the mean
    in each dimension into a single distance value.
    Squared Mahalanobis distance is χ²-distributed, the corresponding p-value is the confidence score.

    Parameters
    ----------
    preprocessors : iterable = ()
        Preprocessors to apply.

    References
    ----------

    .. [1] `Mahalanobis PC (1936).
       On the generalized distance in statistics.
       Proceedings of the National Institute of Sciences of India, vol 2, no 1, pp 49–55.
       <http://insa.nic.in/writereaddata/UpLoadedFiles/PINSA/Vol02_1936_1_Art05.pdf>`_
    """

    def __init__(self, preprocessors=()):
        super().__init__(preprocessors=preprocessors)

    def _construct(self, X) -> Model:
        model: MD.Model = super()._construct(X)
        model.mean = X.mean(axis=0)
        model.covar_inv = linalg.pinvh(np.cov(X.T, bias=True))
        return model

    class Model(DataDescriptor.Model):

        mean: np.array
        covar_inv: np.array

        def _query(self, X):
            d = X - self.mean
            D2 = (d[..., None, :] @ self.covar_inv @ d[..., None]).squeeze(axis=(-2, -1))
            return 1 - chi2.cdf(D2, df=self.m)
