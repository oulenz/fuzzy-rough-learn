"""Ensemble classifiers"""
from __future__ import annotations

import numpy as np

from ..base import Classifier
from ..neighbours.approximators import ComplementedDistance
from frlearn.base import Approximator
from ..neighbours.neighbour_search import KDTree, NNSearch


class FuzzyRoughClassifier(Classifier):
    def __init__(
            self,
            upper_approximator: Approximator = ComplementedDistance(),
            lower_approximator: Approximator = ComplementedDistance(),
            nn_search: NNSearch = KDTree(),
    ):
        self.upper_approximator = upper_approximator
        self.lower_approximator = lower_approximator
        self.nn_search = nn_search

    class Model(Classifier.Model):
        def __init__(self, classifier, X, y):
            super().__init__(classifier, X, y)

            Cs = [X[np.where(y == c)] for c in self.classes]
            indices = [classifier.nn_search.construct(C) for C in Cs]
            self.upper_approximations = [classifier.upper_approximator.construct(index) for index in indices]

            co_Cs = [X[np.where(y != c)] for c in self.classes]
            co_indices = [classifier.nn_search.construct(co_C) for co_C in co_Cs]
            self.lower_approximations = [classifier.lower_approximator.construct(co_index) for co_index in co_indices]

        def query(self, X):
            vals = []
            if self.upper_approximations:
                vals.append(np.stack([approximation.query(X) for approximation in self.upper_approximations], axis=1))
            if self.lower_approximations:
                vals.append(
                    1 - np.stack([approximation.query(X) for approximation in self.lower_approximations], axis=1))
            if len(vals) == 2:
                return sum(vals) / 2
            return vals[0]