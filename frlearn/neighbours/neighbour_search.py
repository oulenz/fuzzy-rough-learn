"""Nearest neighbour searches"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable, Union

import numpy as np
from sklearn.neighbors._unsupervised import NearestNeighbors

from frlearn.base import ModelFactory


class NNSearch(ModelFactory):
    """
    Abstract base class for nearest neighbour searches. Subclasses must
    implement Model._query (and typically __init__ and _construct).
    """

    def __call__(self, X) -> Model:
        """
        Construct the model based on the data X.

        Parameters
        ----------
        X : array shape=(n, m, )
            Construction instances.

        Returns
        -------
        M : Model
            Constructed model
        """
        return super().__call__(X)

    def _construct(self, X) -> Model:
        model = super()._construct(X)
        model._X = X
        return model

    class Model(ModelFactory.Model):

        _X: np.array

        def query_self(self, k: Union[int, float, None]):
            if callable(k):
                k = k(self.n - 1)
            return [a[:, 1:] for a in self(self._X, k + 1)]

        def __call__(self, X, k: Union[int, Callable[[int], int]]):
            """
            Identify the k nearest neighbours for each of the instances in X.

            Parameters
            ----------
            X : array shape=(n, m, )
                Query instances.

            k : int or (int -> int)
                Number of neighbours to return. Should be either a positive integer not larger than the model size,
                or a function that takes the size of the model and returns such an integer.

            Returns
            -------
            I : array shape=(n, k, )
                Indices of the k nearest neighbours among the construction
                instances for each query instance.

            D : array shape=(n, k, )
                Distances to the k nearest neighbours among the construction
                instances for each query instance.
            """
            if callable(k):
                k = k(self.n)
            return self._query(X, k)

        @property
        def query(self):
            return self.__call__

        @abstractmethod
        def _query(self, X, k: int):
            pass


class BallTree(NNSearch):
    """
    Nearest neighbour search with a Ball tree.

    Parameters
    ----------
    metric : str, default='manhattan'
        The metric through which distances are defined.

    leaf_size : int, default=30
        The leaf size to be used for the Ball tree.

    n_jobs : int, default=1
        The number of parallel jobs to run for neighbour search. -1 means using
        all processors.
    """

    def __init__(self, *, metric: str = 'manhattan', leaf_size: int = 30,
                 n_jobs: int = 1, preprocessors=()):
        super().__init__(preprocessors=preprocessors)
        self.construction_params = {
            'algorithm': 'ball_tree',
            'metric': metric,
            'leaf_size': leaf_size,
            'n_jobs': n_jobs,
        }

    def _construct(self, X) -> Model:
        model = super()._construct(X)
        model.tree = NearestNeighbors(**self.construction_params).fit(X)
        return model

    class Model(NNSearch.Model):

        tree: NearestNeighbors

        def _query(self, X, k: int):
            return self.tree.kneighbors(X, n_neighbors=k)[::-1]


class KDTree(NNSearch):
    """
    Nearest neighbour search with a KD-tree.

    Parameters
    ----------
    metric : str, default='manhattan'
        The metric through which distances are defined.

    leaf_size : int, default=30
        The leaf size to be used for the KD-tree.

    n_jobs : int, default=1
        The number of parallel jobs to run for neighbour search. -1 means using
        all processors.
    """

    def __init__(self, *, metric: str = 'manhattan', leaf_size: int = 30,
                 n_jobs: int = 1, preprocessors=()):
        super().__init__(preprocessors=preprocessors)
        self.construction_params = {
            'algorithm': 'kd_tree',
            'metric': metric,
            'leaf_size': leaf_size,
            'n_jobs': n_jobs,
        }

    def _construct(self, X) -> Model:
        model = super()._construct(X)
        model.tree = NearestNeighbors(**self.construction_params).fit(X)
        return model

    class Model(NNSearch.Model):

        tree: NearestNeighbors

        def _query(self, X, k: int):
            return self.tree.kneighbors(X, n_neighbors=k)[::-1]
