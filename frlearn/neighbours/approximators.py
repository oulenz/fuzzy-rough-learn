"""Nearest neighbour approximators"""
from __future__ import annotations

from abc import abstractmethod

import numpy as np

from ..base import Approximator
from ..utils.owa_operators import OWAOperator, trimmed


class NNApproximator(Approximator):

    @abstractmethod
    def __init__(self, k, owa: OWAOperator, *args, **kwargs):
        self.k = k
        self.owa = owa

    class Approximation(Approximator.Approximation):

        def __init__(self, approximator, index):
            self.index = index
            self.k = approximator.k
            self.owa = approximator.owa
            self.neighbours, self.distances = index.query_self(self.k + 1)

        def query(self, X):
            q_neighbours, q_distances = self.index.query(X, self.k)
            return self._query(q_neighbours, q_distances)

        @abstractmethod
        def _query(self, q_neighbours, q_distances):
            pass

        @abstractmethod
        def query_self(self):
            pass

        def _query_self_naive(self):
            q_neighbours = self.neighbours[..., :self.k]
            q_distances = self.distances[..., :self.k]
            return self._query(q_neighbours, q_distances)

        def copy(self, **attribute_values):
            other = super().copy(**attribute_values)
            if 'k' not in attribute_values:
                return other
            if other.k + 1 > other.neighbours.shape[-1]:
                other.neighbours, other.distances = other.index.query_self(other.k + 1)
            else:
                other.neighbours = other.neighbours[..., :other.k + 1]
                other.distances = other.distances[..., :other.k + 1]
            return other


class ComplementedDistance(NNApproximator):

    def __init__(self, k: int = 40, owa: OWAOperator = trimmed):
        super().__init__(k=k, owa=owa)

    class Approximation(NNApproximator.Approximation):

        def _query(self, q_neighbours, q_distances):
            indiscernibilities = np.maximum(0, 1 - q_distances)
            score = self.owa.soft_max(indiscernibilities, self.k)
            return score

        def query_self(self):
            return self._query_self_naive()
