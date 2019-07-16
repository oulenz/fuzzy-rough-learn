from __future__ import annotations
from abc import ABC, abstractmethod

from sklearn.neighbors.unsupervised import NearestNeighbors


class NNSearch(ABC):
    """
    Abstract base class for nearest neighbour searches. Subclasses must
    implement __init__ and Index.
    """

    @abstractmethod
    def __init__(self, **kwargs):
        pass

    def construct(self, X) -> Index:
        """
        Construct the index based on the data X.

        Parameters
        ----------
        X : array shape=(n_instances, n_features, )
            Construction instances.

        Returns
        -------
        I : Index
            Constructed index
        """
        return self.Index(self, X)

    class Index(ABC):

        """
        Abstract base class for the index object created by NNSearch.construct.
        Subclasses must implement __init__ and query.

        Parameters
        ----------
        search : NNSearch
            The search object that contains all the relevant parametre values.

        X : array shape=(n_instances, n_features, )
            Construction instances.
        """

        @abstractmethod
        def __init__(self, search: NNSearch, X):
            pass

        @abstractmethod
        def query(self, X, k: int):
            """
            Identify the k nearest neighbours for each of the instances in X.

            Parameters
            ----------
            X : array shape=(n_instances, n_features, )
                Query instances.

            k : int
                Number of neighbours to return

            Returns
            -------
            N : array shape=(n_instances, k, )
                Indices of the k nearest neighbours among the construction
                instances for each query instance.
            """
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
                 n_jobs: int = 1):
        self.construction_params = {
            'algorithm': 'ball_tree',
            'metric': metric,
            'leaf_size': leaf_size,
            'n_jobs': n_jobs,
        }

    class Index(NNSearch.Index):

        def __init__(self, search: BallTree, X):
            self.tree = NearestNeighbors(**search.construction_params).fit(X)

        def query(self, X, k: int):
            return self.tree.kneighbors(X, n_neighbors=k)[0]


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
                 n_jobs: int = 1):
        self.construction_params = {
            'algorithm': 'kd_tree',
            'metric': metric,
            'leaf_size': leaf_size,
            'n_jobs': n_jobs,
        }

    class Index(NNSearch.Index):

        def __init__(self, search: KDTree, X):
            self.tree = NearestNeighbors(**search.construction_params).fit(X)

        def query(self, X, k: int):
            return self.tree.kneighbors(X, n_neighbors=k)[0]
