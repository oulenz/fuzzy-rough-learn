"""Support vector data descriptors"""
from __future__ import annotations

from typing import Callable

from sklearn.svm import OneClassSVM

from frlearn.base import DataDescriptor
from frlearn.feature_preprocessors import IQRNormaliser
from frlearn.parametrisations import multiple
from frlearn.transformations import contract


class SVM(DataDescriptor):
    """
    Wrapper for the Support Vector Machine (SVM) data descriptor [1]_ with gaussian kernel, implemented in scikit-learn.
    Expresses the signed distance to the separating hyperplane, scaled to `[0, 1]`.

    Parameters
    ----------
    nu : float = 0.20
        How many nearest neighbour distances / localised proximities to consider.
        Corresponds to the scale at which proximity is evaluated.
        Should be either a positive integer not larger than the target class size,
        or a function that takes the size of the target class and returns such an integer.

    c : float or (int -> float) = 0.25 * m
        Kernel width.
        Should be either a positive float
        or a function that takes the dimensionality of the target class and returns such a float.

    preprocessors : iterable = (IQRNormaliser(), )
        Preprocessors to apply. The default interquartile range normaliser rescales all features
        to ensure that they all have the same interquartile range.

    sklearn_params
        Additional keyword parameters will be passed on as-is to scikit-learn's OneClassSVM constructor.

    Notes
    -----
    `nu` and `c` are the two principal hyperparameters that can be tuned to increase performance.
    Its default values are based on the empirical evaluation in [2]_.

    References
    ----------
    .. [1] `Schölkopf B, Platt JC, Shawe-Taylor J, Smola AJ, Williamson RC (1999).
       Estimating the support of a high-dimensional distribution.
       MSR-TR-99-87, Microsoft Research.
       <https://www.microsoft.com/en-us/research/publication/estimating-the-support-of-a-high-dimensional-distribution/>`_
    .. [2] `Lenz OU, Peralta D, Cornelis C (2021).
       Average Localised Proximity: A new data descriptor with good default one-class classification performance.
       Pattern Recognition, vol 118, no 107991.
       doi: 10.1016/j.patcog.2021.107991
       <https://www.sciencedirect.com/science/article/abs/pii/S0031320321001783>`_
    """

    def __init__(
            self,
            nu: float = 0.20,
            c: float | Callable[[int], float] = multiple(0.25),
            preprocessors=(IQRNormaliser(), ),
            **sklearn_params,
    ):
        super().__init__(preprocessors=preprocessors)
        self.nu = nu
        self.c = c
        self.sklearn_params = sklearn_params

    def _construct(self, X, ):
        model = super()._construct(X)
        model.nu = self.nu
        model.c = self.c(X.shape[1]) if callable(self.c) else self.c
        model.svm = OneClassSVM(nu=model.nu, gamma=1/model.c, **self.sklearn_params).fit(X)
        return model

    class Model(DataDescriptor.Model):

        nu: float
        c: float
        svm: OneClassSVM

        def _query(self, X):
            signed_distance = self.svm.decision_function(X)
            # scale signed distance from [-∞, ∞] to (0, 1)
            score = contract(signed_distance)
            return score
