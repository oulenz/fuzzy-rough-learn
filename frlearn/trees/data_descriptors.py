"""Decision tree data descriptors"""
from __future__ import annotations

from typing import Callable

from sklearn.ensemble import IsolationForest

from frlearn.base import DataDescriptor


class EIF(DataDescriptor):
    """
    Wrapper for the Extended Isolation Forest (IF) data descriptor [1]_.
    Requires the eif library, which is not automatically installed.
    Expresses the effort required to isolate a query instance from the target data
    by separating instances with random hyperplanes.

    Parameters
    ----------
    psi : int or (int -> int) = 256
        Sub-sampling size. Number of training instances to use for each random tree.
        Should be either a positive integer,
        or a function that takes the size of the target class and returns such an integer.
        If the size of the target class is a smaller number, that will be used instead.

    t : int = 100
        Number of random trees.

    random_state : int = 0
        Random state to use.

    eif_params
        additional keyword parameters will be passed on as-is to eif's iForest constructor.

    preprocessors : iterable = ()
        Preprocessors to apply.

    Notes
    -----
    Scores are the complement of the anomaly scores in [1]_.
    `psi` and `t` are two hyperparameters that can potentially be tuned,
    but the default values should be good enough [2]_.

    References
    ----------
    .. [1] `Hariri S, Carrasco Kind M, Brunner RJ (2021).
       Extended Isolation Forest.
       IEEE Transactions on Knowledge and Data Engineering, vol 33, no 4, pp 1479–1489.
       doi: 10.1109/TKDE.2019.2947676
       <https://ieeexplore.ieee.org/document/8888179>`_
    .. [2] `Liu FT, Ting KM, Zhou Z-H (2008).
       Isolation Forest.
       ICDM 2008: Proceedings of the Eighth IEEE International Conference on Data Mining, pp 413–422.
       IEEE.
       doi: 10.1109/ICDM.2008.17
       <https://ieeexplore.ieee.org/document/4781136>`_
    """

    def __init__(
            self,
            psi: int | Callable[[int], int] = 256,
            t: int = 100,
            random_state: int = 0,
            preprocessors=(),
            **eif_params
    ):
        super().__init__(preprocessors=preprocessors)
        try:
            import eif
        except ImportError:
            raise ImportError('EIF data descriptor requires the eif library.') from None
        self.psi = psi
        self.t = t
        self.random_state = random_state
        self.eif_params = eif_params

    def _construct(self, X):
        import eif
        model = super()._construct(X)
        model.psi = min(self.psi, X.shape[0])
        model.t = self.t
        model.random_state = self.random_state
        model.forest = eif.iForest(
            X,
            ntrees=model.t, sample_size=model.psi, seed=model.random_state,
            ExtensionLevel=X.shape[1] - 1, **self.eif_params
        )
        return model

    class Model(DataDescriptor.Model):

        psi: int
        t: int
        random_state: int
        forest: ...

        def _query(self, X):
            # convert anomaly scores to normality scores
            return 1 - self.forest.compute_paths(X_in=X)


class IF(DataDescriptor):
    """
    Wrapper for the Isolation Forest (IF) data descriptor [1]_ implemented in scikit-learn.
    Expresses the effort required to isolate a query instance from the target data
    by random splits on attribute values.

    Parameters
    ----------
    psi : int or (int -> int) = 256
        Sub-sampling size. Number of training instances to use for each random tree.
        Should be either a positive integer,
        or a function that takes the size of the target class and returns such an integer.
        If the size of the target class is a smaller number, that will be used instead.

    t : int = 100
        Number of random trees.

    random_state : int = 0
        Random state to use.

    preprocessors : iterable = ()
        Preprocessors to apply.

    sklearn_params
        Additional keyword parameters will be passed on as-is to scikit-learn's IsolationForest constructor.

    Notes
    -----
    Scores are the complement of the anomaly scores in [1]_.
    `psi` and `t` are two hyperparameters that can potentially be tuned,
    but the default values should be good enough [1]_.

    References
    ----------
    .. [1] `Liu FT, Ting KM, Zhou Z-H (2008).
       Isolation Forest.
       ICDM 2008: Proceedings of the Eighth IEEE International Conference on Data Mining, pp 413–422.
       IEEE.
       doi: 10.1109/ICDM.2008.17
       <https://ieeexplore.ieee.org/document/4781136>`_
    """

    def __init__(
            self,
            psi: int | Callable[[int], int] = 256,
            t: int = 100,
            random_state: int = 0,
            preprocessors=(),
            **sklearn_params,
    ):
        super().__init__(preprocessors=preprocessors)
        self.psi = psi
        self.t = t
        self.random_state = random_state
        self.sklearn_params = sklearn_params

    def _construct(self, X):
        model = super()._construct(X)
        model.psi = min(self.psi, X.shape[0])
        model.t = self.t
        model.random_state = self.random_state
        model.forest = IsolationForest(
            max_samples=model.psi, n_estimators=model.t, random_state=self.random_state,
            **self.sklearn_params
        ).fit(X)
        return model

    class Model(DataDescriptor.Model):

        psi: int
        t: int
        random_state: int
        forest: IsolationForest

        def _query(self, X):
            # map from [-1, 0] to [0, 1]
            return 1 + self.forest.score_samples(X)
