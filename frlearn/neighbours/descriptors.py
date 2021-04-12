"""Nearest neighbour data descriptors"""
from __future__ import annotations

from abc import abstractmethod
from typing import Callable, Union

import numpy as np

from .neighbour_search import KDTree, NNSearch
from ..base import Descriptor
from ..utils.np_utils import div_or, log_based_k, shifted_reciprocal
from ..utils.owa_operators import OWAOperator, additive, trimmed


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


class ALP(NNDescriptor):
    """
    Implementation of the Average Localised Proximity (ALP) data descriptor [1]_.
    Expresses the proximity of a query instance to the target data,
    by localising its nearest neighbour distances against the local nearest neighbour distances in the target data.

    Parameters
    ----------
    k : int or (int -> int) = 5.5 * log n
        How many nearest neighbour distances / localised proximities to consider.
        Corresponds to the scale at which proximity is evaluated.
        Should be either a positive integer not larger than the target class size,
        or a function that takes the size of the target class and returns such an integer.

    l : int or (int -> int) = 6 * log n
        How many nearest neighbours to use for determining the local ith nearest neighbour distance, for each `i <= k`.
        Lower values correspond to more localisation.
        Should be either a positive integer not larger than the target class size,
        or a function that takes the size of the target class and returns such an integer.

    scale_weights : OWAOperator = additive()
        Weights to use for calculating the soft maximum of localised proximities.
        Determines to which extent scales with high localised proximity are emphasised.

    localisation_weights : OWAOperator = additive()
        Weights to use for calculating the local ith nearest neighbour distance, for each `i <= k`.
        Determines to which extent nearer neighbours dominate.

    max_array_size : int = 2**26
        Maximum array size to use. For a query set of size `q`,
        calculating local distances requires an array of size `[q, l, k]`,
        which can be too large to fit in memory. If the size of this array is larger than `max_array_size`,
        a query set is batch-processed, which is slower.
        TODO: determine maximum array size dynamically, investigate lowering float precision

    Notes
    -----
    `k` and `l` are the two principal hyperparameters that can be tuned to increase performance.
    Its default values are based on the empirical evaluation in [1]_.

    References
    ----------
    .. [1] `Lenz OU, Peralta D, Cornelis C (2021).
       Average Localised Proximity: a new data descriptor with good default one-class classification performance.
       Pattern Recognition, to appear.
       <https://arxiv.org/abs/2101.11037>`_
    """

    def __init__(
            self,
            k: int | Callable[[int], int] = log_based_k(5.5),
            l: int | Callable[[int], int] = log_based_k(6),
            scale_weights: OWAOperator = additive(),
            localisation_weights: OWAOperator = additive(),
            nn_search: NNSearch = KDTree(),
            max_array_size: int = 2**26,
    ):
        super().__init__(nn_search=nn_search, k=k, )
        self.l = l
        self.scale_weights = scale_weights
        self.localisation_weights = localisation_weights
        self.max_array_size = max_array_size

    def construct(self, X):
        model: ALP.Model = super().construct(X)
        model.l = self.l(len(model.nn_model)) if callable(self.l) else self.l
        model._kl = max(model.k, model.l)
        _, model.distances = model.nn_model.query_self(model._kl)
        model.scale_weights = self.scale_weights
        model.localisation_weights = self.localisation_weights
        return model

    class Model(NNDescriptor.Model):

        l: int
        _kl: int
        distances: np.ndarray
        scale_weights: OWAOperator
        localisation_weights: OWAOperator

        def query(self, X):
            q_neighbours, q_distances = self.nn_model.query(X, self._kl)
            return self._query(q_neighbours[..., :self.l], q_distances[..., :self.k])

        def _query(self, q_neighbours, q_distances):
            batch_size = 2**26 // (self.k * self.l)
            local_distances = []
            for i in range(0, q_neighbours.shape[0], batch_size):
                local_distances.append(self.localisation_weights.soft_head(
                    self.distances[q_neighbours[i:i+batch_size], :self.k], self.l, axis=-2
                ))
            local_distances = np.concatenate(local_distances, axis=0)
            
            # if both distances are zero, default to 1
            localised_distances = div_or(q_distances, local_distances, 1)
            localised_proximities = shifted_reciprocal(localised_distances)
            return self.scale_weights.soft_max(localised_proximities, self.k)


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
