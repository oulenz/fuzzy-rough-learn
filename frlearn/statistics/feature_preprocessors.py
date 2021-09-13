"""Statistical preprocessors"""
from __future__ import annotations

import numpy as np

from frlearn.base import FeaturePreprocessor, Unsupervised
from frlearn.dispersion_measures import interquartile_range, maximum_absolute_value, standard_deviation, total_range
from frlearn.location_measures import mean, midhinge, midrange


class LinearNormaliser(Unsupervised, FeaturePreprocessor):
    """
    Linearly transforms all features by normalising a measure of dispersion and a measure of location,
    ensuring that for each feature, that measure of dispersion becomes 1 and that measure of location becomes 0.

    Parameters
    ----------
    dispersion: (np.array -> np.array) or None = None
        The measure of dispersion to normalise.

    location: (np.array -> np.array) or None = None
        The measure of location to normalise.

    Notes
    -----
    If the measure of dispersion is 0 for some feature, it will be left unnormalised.

    """

    def __init__(self, dispersion=None, location=None, ):
        super().__init__()
        self.dispersion = dispersion
        self.location = location

    def _construct(self, X, ) -> Model:
        model = super()._construct(X)
        if self.dispersion is not None:
            divisor = self.dispersion(X)
            divisor = np.where(divisor == 0 | np.isnan(divisor), 1, divisor)
        else:
            divisor = 1
        if self.location is not None:
            subtrahend = self.location(X)
            subtrahend = np.where(np.isnan(subtrahend), 0, subtrahend)
        else:
            subtrahend = 0
        model.divisor = divisor
        model.subtrahend = subtrahend
        return model

    class Model(Unsupervised.Model, FeaturePreprocessor.Model):

        divisor: np.array
        subtrahend: np.array

        def _query(self, X):
            return (X - self.subtrahend)/self.divisor


class IQRNormaliser(LinearNormaliser):
    """
    Implementation of the interquartile range (IQR) normaliser.
    Ensures that for each feature, [-0.5, 0.5] contains the central half of all data.

    Notes
    -----
    If the interquartile range of a feature is 0, that feature is left unscaled.

    """

    def __init__(self):
        super().__init__(dispersion=interquartile_range, location=midhinge)


class MaxAbsNormaliser(LinearNormaliser):
    """
    Implementation of the maximum absolute value normaliser.
    Rescales all features by dividing through their maximum absolute value,
    ensuring that the values of each feature lie in [-1, 1],
    although the range of feature will in general be less than 2.

    Notes
    -----
    If the maximum absolute value of a feature is 0, that feature is left unscaled.

    """

    def __init__(self):
        super().__init__(dispersion=maximum_absolute_value)


class RangeNormaliser(LinearNormaliser):
    """
    Implementation of the range normaliser.
    Rescales all features by dividing through their total range,
    ensuring that the values of each feature lie in [-0.5, 0.5].

    Notes
    -----
    If the range of a feature is 0, that feature is left unscaled.

    """

    def __init__(self):
        super().__init__(dispersion=total_range, location=midrange, )


class Standardiser(LinearNormaliser):
    """
    Implementation of the standard deviation normaliser, or standardiser.
    Rescales all features by dividing through their standard deviation,
    ensuring that each feature has a standard deviation of 1.

    Notes
    -----
    If the standard deviation of a feature is 0, that feature is left unscaled.

    """

    def __init__(self):
        super().__init__(dispersion=standard_deviation, location=mean)
