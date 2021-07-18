"""Nearest neighbour search methods"""
from __future__ import annotations

from abc import abstractmethod

import numpy as np
from sklearn.neighbors._unsupervised import NearestNeighbors

from frlearn.base import ModelFactory
from frlearn.feature_preprocessors import NormNormaliser


class NeighbourSearchMethod(ModelFactory):
    """
    Abstract base class for nearest neighbour searches. Subclasses must
    implement Model._query (and typically __init__ and _construct).
    """

    def __call__(self, X, metric='manhattan', *, preprocessors=()) -> Model:
        """
        Construct the model based on the data X.

        Parameters
        ----------
        X: array shape=(n, m, )
            Construction instances.

        metric: str = 'manhattan'
            The metric through which distances are defined.

        Returns
        -------
        M: Model
            Constructed model
        """
        return super().__call__(X, metric=metric, preprocessors=preprocessors)

    def _construct(self, X, metric) -> Model:
        model = super()._construct(X)
        model._X = X
        model.metric = metric
        return model

    class Model(ModelFactory.Model):

        _X: np.array
        metric: str

        def query_self(self, k: int):
            return [a[:, 1:] for a in self(self._X, k + 1)]

        def __call__(self, X, k: int):
            """
            Identify the k nearest neighbours for each of the instances in X.

            Parameters
            ----------
            X: array shape=(n, m, )
                Query instances.

            k: int
                Number of neighbours to return. Should be a positive integer not larger than the model size.

            Returns
            -------
            I: array shape=(n, k, )
                Indices of the k nearest neighbours among the construction
                instances for each query instance.

            D: array shape=(n, k, )
                Distances to the k nearest neighbours among the construction
                instances for each query instance.
            """
            return super().__call__(X, k)

        @property
        def query(self):
            return self.__call__

        @abstractmethod
        def _query(self, X, k: int):
            pass


class _SKLearnTree(NeighbourSearchMethod):
    """
    Abstract base wrapper for the tree classes implemented by scikit-learn.

    Parameters
    ----------
    algorithm: str
        scikit-learn algorithm.

    leaf_size: int = 30
        The leaf size to be used for the Ball tree.

    n_jobs: int = 1
        The number of parallel jobs to run for neighbour search. -1 means using
        all processors.

    preprocessors: iterable = ()
        Preprocessors to apply.
    """

    @abstractmethod
    def __init__(self, *, algorithm: str, leaf_size: int,
                 n_jobs: int, preprocessors: tuple):
        super().__init__(preprocessors=preprocessors)
        self.construction_params = {
            'algorithm': algorithm,
            'leaf_size': leaf_size,
            'n_jobs': n_jobs,
        }

    def __call__(self, X, metric='manhattan'):
        if metric == 'cosine':
            return super().__call__(X, metric=metric, preprocessors=(NormNormaliser(p=2), ))
        return super().__call__(X, metric=metric, )

    def _construct(self, X, metric) -> Model:
        model = super()._construct(X, metric)
        if metric == 'cosine':
            metric = 'euclidean'
        model.tree = NearestNeighbors(metric=metric, **self.construction_params).fit(X)
        return model

    class Model(NeighbourSearchMethod.Model):

        tree: NearestNeighbors

        def _query(self, X, k: int):
            indices, distances = self.tree.kneighbors(X, n_neighbors=k)[::-1]
            if self.metric == 'cosine':
                distances = 0.25 * distances**2
            return indices, distances


class BallTree(_SKLearnTree):
    """
    Nearest neighbour search with a Ball tree.

    Parameters
    ----------
    leaf_size: int = 30
        The leaf size to be used for the Ball tree.

    n_jobs: int = 1
        The number of parallel jobs to run for neighbour search. -1 means using
        all processors.

    preprocessors: iterable = ()
        Preprocessors to apply.
    """

    def __init__(self, *, leaf_size: int = 30,
                 n_jobs: int = 1, preprocessors=()):
        super().__init__(
            algorithm='ball_tree', leaf_size=leaf_size, n_jobs=n_jobs,
            preprocessors=preprocessors
        )


class KDTree(_SKLearnTree):
    """
    Nearest neighbour search with a KD-tree.

    Parameters
    ----------
    leaf_size: int = 30
        The leaf size to be used for the KD-tree.

    n_jobs: int = 1
        The number of parallel jobs to run for neighbour search. -1 means using
        all processors.

    preprocessors: iterable = ()
        Preprocessors to apply.
    """

    def __init__(self, *, leaf_size: int = 30,
                 n_jobs: int = 1, preprocessors=()):
        super().__init__(
            algorithm='kd_tree', leaf_size=leaf_size, n_jobs=n_jobs,
            preprocessors=preprocessors
        )
