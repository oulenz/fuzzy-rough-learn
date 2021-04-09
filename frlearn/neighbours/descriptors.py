"""Nearest neighbour data descriptors"""
from __future__ import annotations

from abc import abstractmethod
from typing import Callable, Union

import numpy as np

from .neighbour_search import KDTree, NNSearch
from ..base import Descriptor
from ..utils.np_utils import div_or, shifted_reciprocal
from ..utils.owa_operators import OWAOperator, trimmed


class NNDescriptor(Descriptor):

    @abstractmethod
    def __init__(self, nn_search: NNSearch, k: Union[int, Callable[[int], int]], *args, **kwargs):
        self.nn_search = nn_search
        self.k = k

    @abstractmethod
    def construct(self, X) -> Model:
        model = super().construct(X)
        nn_model = self.nn_search.construct(X)
        model.nn_model = nn_model
        model.k = self.k(len(nn_model)) if callable(self.k) else self.k
        return model

    class Model(Descriptor.Model):

        nn_model: NNSearch.Model
        k: int

        def query(self, X):
            q_neighbours, q_distances = self.nn_model.query(X, self.k)
            return self._query(q_neighbours, q_distances)

        @abstractmethod
        def _query(self, q_neighbours, q_distances):
            pass


class LNND(NNDescriptor):
    """
    Implementation of the Localised Nearest Neighbour Distance (LNND) data descriptor [1]_[2]_.

    Parameters
    ----------
    k : int or (int -> int) = 1
        Which nearest neighbour to consider.
        Should be either a positive integer not larger than the target class size,
        or a function that takes the size of the target class and returns such an integer.

    Notes
    -----
    The scores are derived with 1/(1 + l_distances).

    References
    ----------

    .. [1] `de Ridder D, Tax DMJ, Duin RPW (1998).
       An experimental comparison of one-class classification methods.
       ASCI`98: Proceedings of the Fourth Annual Conference of the Advanced School for Computing and Imaging, 213–218.
       ASCI.
       <http://rduin.nl/papers/asci_98.html>`_
    .. [2] `Tax DMJ, Duin RPW (1998).
       Outlier detection using classifier instability.
       SSPR/SPR 1998: Joint IAPR International Workshops on Statistical Techniques in Pattern Recognition and Structural and Syntactic Pattern Recognition, 593--601.
       Springer.
       doi: 10.1007/BFb0033283
       <https://link.springer.com/chapter/10.1007/BFb0033283>`_
    """

    def __init__(self, nn_search: NNSearch = KDTree(), k: Union[int, Callable[[int], int]] = 1):
        super().__init__(nn_search=nn_search, k=k)

    def construct(self, X) -> Model:
        model: LNND.Model = super().construct(X)
        _, distances = model.nn_model.query_self(model.k)
        model.distances = distances[:, -1]
        return model

    class Model(NNDescriptor.Model):

        distances: np.ndarray

        def _query(self, q_neighbours, q_distances):
            # if both distances are zero, default to 1
            l_distances = div_or(q_distances[:, self.k-1], self.distances[q_neighbours[:, self.k-1]], 1)
            # replace infinites with very large numbers, but keep nans (which shouldn't be here) to flag problems
            l_distances = np.nan_to_num(l_distances, nan=np.nan)
            return shifted_reciprocal(l_distances)


class LOF(NNDescriptor):
    """
    Implementation of the Local Outlier Factor (LOF) data descriptor [1]_.

    Parameters
    ----------
    k : int or (int -> int) = 1
        How many nearest neighbours to consider.
        Should be either a positive integer not larger than the target class size,
        or a function that takes the size of the target class and returns such an integer.

    Notes
    -----
    The scores are derived with 1/(1 + lof).

    References
    ----------

    .. [1] `Breunig MM, Kriegel H-P, Ng RT, Sander J (2000).
       LOF: identifying density-based local outliers.
       SIGMOD 2000: ACM international conference on Management of data, 93–104.
       ACM.
       doi: 10.1145/342009.335388
       <https://dl.acm.org/doi/abs/10.1145/342009.335388>`_
    """

    def __init__(self, nn_search: NNSearch = KDTree(), k: Union[int, Callable[[int], int]] = 1):
        super().__init__(nn_search=nn_search, k=k)

    def construct(self, X) -> Model:
        model: LOF.Model = super().construct(X)
        neighbours, distances = model.nn_model.query_self(model.k)
        model.distances = distances[:, -1]
        model.lrd = model._get_lrd(neighbours, distances)
        return model

    class Model(NNDescriptor.Model):

        distances: np.ndarray
        lrd: np.ndarray

        def _get_lrd(self, q_neighbours, q_distances):
            r_distances = np.maximum(q_distances, self.distances[q_neighbours])
            return 1/np.mean(r_distances, axis=-1)

        def _query(self, q_neighbours, q_distances):
            q_lrd = self._get_lrd(q_neighbours, q_distances)
            lof = np.mean(self.lrd[q_neighbours], axis=-1) / q_lrd
            # handle nan, which comes from inf/inf
            lof[np.isnan(lof)] = 1
            return shifted_reciprocal(lof)


class NND(NNDescriptor):
    """
    Implementation of the Nearest Neighbour Distance (NND) data descriptor, which goes back to at least [1]_.
    It has also been used to calculate upper and lower approximations of fuzzy rough sets,
    where the addition of aggregation with OWA operators is due to [2]_.

    Parameters
    ----------
    k : int or (int -> int) = 1
        Which nearest neighbour(s) to consider.
        Should be either a positive integer not larger than the target class size,
        or a function that takes the size of the target class and returns such an integer.
        If `owa = trimmed()`, only the kth neighbour is used,
        otherwise closer neighbours are also taken into account.

    proximity : float -> float = np_utils.shifted_reciprocal
        The function used to convert distance values to proximity values.
        It should be be an order-reversing map from `[0, ∞)` to `[0, 1]`.

    owa : OWAOperator = trimmed()
        How to aggregate the proximity values from the `k` nearest neighbours.
        The default is to only consider the kth nearest neighbour distance.

    References
    ----------

    .. [1] `Knorr EM, Ng RT (1997).
       A Unified Notion of Outliers: Properties and Computation.
       KDD-97: Proceedings of the Third International Conference on Knowledge Discovery and Data Mining, pp 219–222.
       AAAI.
       doi: 10.5555/3001392.3001438
       <https://www.aaai.org/Library/KDD/1997/kdd97-044.php>`_
    .. [2] `Cornelis C, Verbiest N, Jensen R (2010).
       Ordered weighted average based fuzzy rough sets.
       RSKT 2010: Proceedings of the 5th International Conference on Rough Set and Knowledge Technology, pp 78--85.
       Springer, Lecture Notes in Artificial Intelligence 6401.
       doi: 10.1007/978-3-642-16248-0_16
       <https://link.springer.com/chapter/10.1007/978-3-642-16248-0_16>`_
    """

    def __init__(
            self,
            nn_search: NNSearch = KDTree(),
            k: Union[int, Callable[[int], int]] = 1,
            proximity: Callable[[float], float] = shifted_reciprocal,
            owa: OWAOperator = trimmed(),
    ):
        super().__init__(nn_search=nn_search, k=k)
        self.proximity = proximity
        self.owa = owa

    def construct(self, X) -> Model:
        model: NND.Model = super().construct(X)
        model.proximity = self.proximity
        model.owa = self.owa
        return model

    class Model(NNDescriptor.Model):

        proximity: Callable[[float], float]
        owa: OWAOperator

        def _query(self, q_neighbours, q_distances):
            proximities = self.proximity(q_distances)
            score = self.owa.soft_max(proximities, self.k)
            return score
