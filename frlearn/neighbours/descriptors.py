"""Nearest neighbour data descriptors"""
from __future__ import annotations

from abc import abstractmethod
from typing import Union

import numpy as np

from ..base import Descriptor
from ..utils.owa_operators import OWAOperator, trimmed


class NNDescriptor(Descriptor):

    @abstractmethod
    def __init__(self, k: Union[int, float, None], owa: OWAOperator, *args, **kwargs):
        self.k = k
        self.owa = owa

    class Description(Descriptor.Description):

        def __init__(self, descriptor, index):
            self.index = index
            self.owa = descriptor.owa
            k = descriptor.k
            if k and 0 < k < 1:
                self.k = max(int(k * len(index)), 1)
            elif not k:
                self.k = len(index)
            else:
                self.k = k
            self.neighbours, self.distances = index.query_self(self.k if k else len(index) - 1)

        def query(self, X):
            q_neighbours, q_distances = self.index.query(X, self.k)
            return self._query(q_neighbours, q_distances)

        @abstractmethod
        def _query(self, q_neighbours, q_distances):
            pass

        def copy(self, **attribute_values):
            other = super().copy(**attribute_values)
            if 'k' not in attribute_values:
                return other
            if other.k > other.neighbours.shape[-1]:
                other.neighbours, other.distances = other.index.query_self(other.k)
            else:
                other.neighbours = other.neighbours[..., :other.k]
                other.distances = other.distances[..., :other.k]
            return other


class ComplementedDistance(NNDescriptor):

    def __init__(self, k: Union[int, float, None] = 40, owa: OWAOperator = trimmed):
        super().__init__(k=k, owa=owa)

    class Description(NNDescriptor.Description):

        def _query(self, q_neighbours, q_distances):
            indiscernibilities = np.maximum(0, 1 - q_distances)
            score = self.owa.soft_max(indiscernibilities, self.k)
            return score

        def query_self(self):
            return self._query_self_naive()
