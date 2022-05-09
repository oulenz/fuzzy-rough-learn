"""Nearest neighbour search methods"""
from __future__ import annotations

from abc import abstractmethod
from typing import Callable

import numpy as np
from sklearn.neighbors import NearestNeighbors

from frlearn.base import SoftMachine
from frlearn.vector_size_measures import MinkowskiSize


class NeighbourSearchMethod(SoftMachine):
    """
    Abstract base class for nearest neighbour searches. Subclasses must
    implement Model._query (and typically __init__ and _construct).
    """

    def __call__(
            self, X,
            dissimilarity: Callable[[np.array], float] or Callable[[np.array, np.array], float] = MinkowskiSize(p=1),
    ) -> Model:
        """
        Construct the model based on the data X.

        Parameters
        ----------
        X: array shape=(n, m, )
            Construction instances.

        dissimilarity: (np.array -> float) or ((np.array, np.array) -> float) = MinkowskiSize(p=1)
            The dissimilarity measure used to calculate distances.
            A vector size measure `np.array -> float` induces a dissimilarity measure through application to `y - x`.

        Returns
        -------
        M: Model
            Constructed model
        """
        return super().__call__(X, dissimilarity=dissimilarity)

    def _construct(self, X, dissimilarity) -> Model:
        model = super()._construct(X)
        model._X = X
        model.dissimilarity = dissimilarity
        return model

    class Model(SoftMachine.Model):

        _X: np.array
        dissimilarity: Callable[[np.array], float] or Callable[[np.array, np.array], float]

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

    def _construct(self, X, dissimilarity) -> Model:
        model = super()._construct(X, dissimilarity)
        params = self.construction_params
        if isinstance(dissimilarity, MinkowskiSize):
            if dissimilarity.p == 0:
                if dissimilarity.unrooted:
                    params['metric'] = 'hamming'
                else:
                    raise ValueError('Rooted Hamming size is not supported by the scikit-learn implementations of the KDTree and BallTree algorithms.')
            elif 0 < dissimilarity.p < 1:
                raise ValueError('Minkowski size with `0 < p < 1` is not supported by the scikit-learn implementations of the KDTree and BallTree algorithms.')
            elif dissimilarity.p == np.inf and dissimilarity.unrooted:
                raise ValueError('Unrooted Chebyshev size is not supported by the scikit-learn implementations of the KDTree and BallTree algorithms.')
            else:
                params['metric'] = 'minkowski'
                params['p'] = dissimilarity.p
        else:
            params['metric'] = dissimilarity
        model.tree = NearestNeighbors(**params).fit(X)
        return model

    class Model(NeighbourSearchMethod.Model):

        tree: NearestNeighbors

        def _query(self, X, k: int):
            indices, distances = self.tree.kneighbors(X, n_neighbors=k)[::-1]
            if isinstance(self.dissimilarity, MinkowskiSize):
                if self.dissimilarity.scale_by_dimensionality:
                    distances = distances/(self.m**(1/self.dissimilarity.p))
                if self.dissimilarity.unrooted and self.dissimilarity.p != 0:
                    distances = distances**self.dissimilarity.p
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
