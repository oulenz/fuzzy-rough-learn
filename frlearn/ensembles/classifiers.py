"""Ensemble classifiers"""
from __future__ import annotations

import numpy as np

from ..base import Approximator, MultiClassClassifier, MultiLabelClassifier
from ..neighbours.approximators import ComplementedDistance
from ..neighbours.neighbour_search import KDTree, NNSearch
from ..utils.np_utils import div_or
from ..utils.owa_operators import OWAOperator, additive, exponential


class FuzzyRoughEnsemble(MultiClassClassifier):
    def __init__(
            self,
            upper_approximator: Approximator = ComplementedDistance(),
            lower_approximator: Approximator = ComplementedDistance(),
            nn_search: NNSearch = KDTree(),
    ):
        self.upper_approximator = upper_approximator
        self.lower_approximator = lower_approximator
        self.nn_search = nn_search

    class Model(MultiClassClassifier.Model):
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


class FRNN(FuzzyRoughEnsemble):
    """
    Implementation of Fuzzy Rough Nearest Neighbour (FRNN) classification
    (FRNN).

    Parameters
    ----------
    upper_weights : OWAOperator, default=additive(20)
        OWA operator to use in calculation of upper approximation of
        decision classes.

    lower_weights : OWAOperator, default=None
        OWA operator to use in calculation of lower approximation of
        decision classes.

    nn_search : NNSearch, default=KDTree()
        Nearest neighbour search algorithm to use.

    Notes
    -----
    With strict upper_weights and lower_weights, this is FRNN classification
    as presented in [1]_. The use of OWA operators for the calculation of
    fuzzy rough sets was proposed in [2]_, and OWA operators were first
    explicitly combined with FRNN in [3]_.

    References
    ----------

    .. [1] `Jensen R, Cornelis C (2008).
       A New Approach to Fuzzy-Rough Nearest Neighbour Classification.
       In: Chan CC, Grzymala-Busse JW, Ziarko WP (eds). Rough Sets and Current Trends in Computing. RSCTC 2008.
       Lecture Notes in Computer Science, vol 5306. Springer, Berlin, Heidelberg.
       doi: 10.1007/978-3-540-88425-5_32
       <https://link.springer.com/chapter/10.1007/978-3-540-88425-5_32>`_
    .. [2] `Cornelis C, Verbiest N, Jensen R (2010).
       Ordered Weighted Average Based Fuzzy Rough Sets.
       In: Yu J, Greco S, Lingras P, Wang G, Skowron A (eds). Rough Set and Knowledge Technology. RSKT 2010.
       Lecture Notes in Computer Science, vol 6401. Springer, Berlin, Heidelberg.
       doi: 10.1007/978-3-642-16248-0_16
       <https://link.springer.com/chapter/10.1007/978-3-642-16248-0_16>`_
    .. [3] `E. Ramentol et al.,
       IFROWANN: Imbalanced Fuzzy-Rough Ordered Weighted Average Nearest Neighbor Classification.
       IEEE Transactions on Fuzzy Systems, vol 23, no 5, pp 1622-1637, Oct 2015.
       doi: 10.1109/TFUZZ.2014.2371472
       <https://ieeexplore.ieee.org/document/6960859>`_
    """
    def __init__(self, *, upper_weights: OWAOperator = additive(), upper_k: int = 20,
                 lower_weights: OWAOperator = additive(), lower_k: int = 20,
                 nn_search: NNSearch = KDTree()):
        upper_approximator = ComplementedDistance(owa=upper_weights, k=upper_k) if upper_weights else None
        lower_approximator = ComplementedDistance(owa=lower_weights, k=lower_k) if lower_weights else None
        super().__init__(upper_approximator, lower_approximator, nn_search)


class FROVOCO(MultiClassClassifier):
    """
    Implementation of the Fuzzy Rough OVO COmbination (FROVOCO) ensemble classifier.

    Parameters
    ----------
    nn_search : NNSearch, default=KDTree()
        Nearest neighbour search algorithm to use.

    References
    ----------

    .. [1] `Vluymans S, Fernández A, Saeys Y, Cornelis C, Herrera F (2018).
       Dynamic affinity-based classification of multi-class imbalanced data with one-versus-one decomposition:
       a fuzzy rough set approach.
       Knowledge and Information Systems, vol 56, pp 55–84.
       doi: 10.1007/s10115-017-1126-1
       <https://link.springer.com/article/10.1007/s10115-017-1126-1>`_
    """
    def __init__(
            self,
            nn_search: NNSearch = KDTree(),
    ):
        self.exponential_approximator = ComplementedDistance(owa=exponential(), k=None)
        self.additive_approximator = ComplementedDistance(owa=additive(), k=.1)
        self.nn_search = nn_search

    class Model(MultiClassClassifier.Model):
        def __init__(self, classifier, X, y):
            super().__init__(classifier, X, y)
            
            self.scale = (np.max(X, axis=0) - np.min(X, axis=0)) * self.n_attributes
            X = X.copy() / self.scale

            Cs = [X[np.where(y == c)] for c in self.classes]
            co_Cs = [X[np.where(y != c)] for c in self.classes]

            class_sizes = np.array([len(C) for C in Cs])
            self.ovo_ir = (class_sizes[:, None] / class_sizes)
            self.ova_ir = np.array([c_n / (len(X) - c_n) for c_n in class_sizes])
            max_ir = np.max(self.ovo_ir, axis=1)

            indices = [classifier.nn_search.construct(C) for C in Cs]
            co_indices = [classifier.nn_search.construct(co_C) for co_C in co_Cs]

            add_construct = classifier.additive_approximator.construct
            exp_construct = classifier.exponential_approximator.construct
            self.add_approx = [add_construct(index) if ir > 9 else None for ir, index in zip(max_ir, indices)]
            self.exp_approx = [exp_construct(index) if ir <= 9 else None for ir, index in zip(self.ova_ir, indices)]
            self.co_approx = [add_construct(co_index) if 1/ir > 9 else exp_construct(co_index) for ir, co_index in zip(self.ova_ir, co_indices)]

            self.sig = np.array([self._sig(C) for C in Cs])

        def _sig(self, C):
            approx = [a if ir > 9 else e for ir, a, e in zip(self.ova_ir, self.add_approx, self.exp_approx)]
            vals_C = np.array([np.mean(a.query(C)) for a in approx])
            co_vals_C = np.array([np.mean(co_a.query(C)) for co_a in self.co_approx])
            return (vals_C + 1 - co_vals_C)/2

        def query(self, X):
            X = X.copy() / self.scale
            # The values in the else clause are just placeholders. But we can't use `None`, because that will force
            # the dtype of the resulting array to become `object`, which will in turn lead to 0/0 producing
            # ZeroDivisionError rather than np.nan
            additive_vals_X = np.stack(np.broadcast_arrays(*[a.query(X) if a else -np.inf for a in self.add_approx])).transpose()
            exponential_vals_X = np.stack(np.broadcast_arrays(*[a.query(X) if a else -np.inf for a in self.exp_approx])).transpose()
            co_vals_X = np.array([a.query(X) for a in self.co_approx]).transpose()

            mem = self._mem(additive_vals_X, exponential_vals_X, co_vals_X)

            mse = np.mean((mem[:, None, :] - self.sig) ** 2, axis=-1)
            mse_n = mse/np.sum(mse, axis=-1, keepdims=True)

            wv = self._wv(additive_vals_X, exponential_vals_X)

            return (wv + mem)/2 - mse_n/self.n_classes

        def _mem(self, additive_vals, exponential_vals, co_approximation_vals):
            approximation_vals = np.where(self.ova_ir > 9, additive_vals, exponential_vals)
            return (approximation_vals + 1 - co_approximation_vals) / 2

        def _wv(self, additive_vals, exponential_vals):
            vals = np.where(self.ovo_ir > 9, additive_vals[..., None], exponential_vals[..., None])
            tot_vals = vals + vals.transpose(0, 2, 1)
            vals = div_or(vals, tot_vals, 0.5)
            # Exclude comparisons of a class with itself.
            vals[:, range(self.n_classes), range(self.n_classes)] = 0
            return np.sum(vals, axis=2)


class FRONEC(MultiLabelClassifier):
    """
    Implementation of the Fuzzy ROugh NEighbourhood Consensus (FRONEC) multilabel classifier.

    Parameters
    ----------
    Q_type : int {1, 2, 3, }, default=2
        Quality measure to use for identifying most relevant instances.
        Q^1 uses lower approximation, Q^2 uses upper approximation, Q^3 is the mean of Q^1 and Q^2.
    R_d_type : int {1, 2, }, default=1
        Label similarity relation to use.
        R_d^1 is simple Hamming similarity. R_d^2 is similar, but takes the prior label probabilities into account.
    k : int, default=20
        Number of neighbours to consider for neighbourhood consensus.
    weights: OWAOperator, default=additive()
        OWA weights to use for calculation of soft maximum and/or minimum.
    nn_search : NNSearch, default=KDTree()
        Nearest neighbour search algorithm to use.

    References
    ----------

    .. [1] `Vluymans S, Cornelis C, Herrera F, Saeys Y (2018).
       Multi-label classification using a fuzzy rough neighborhood consensus.
       Information Sciences, vol 433, pp 96–114.
       doi: 10.1016/j.ins.2017.12.034
       <https://www.sciencedirect.com/science/article/pii/S002002551731157X>`_
    """

    def __init__(self, Q_type: int = 2, R_d_type: int = 1,
                 k: int = 20, weights: OWAOperator = additive(), nn_search: NNSearch = KDTree()):
        self.Q_type = Q_type
        self.R_d_type = R_d_type
        self.k = k
        self.weights = weights
        self.nn_search = nn_search

    class Model(MultiLabelClassifier.Model):

        def __init__(self, classifier, X, Y):
            super().__init__(classifier, X, Y)

            self.scale = (np.max(X, axis=0) - np.min(X, axis=0)) * self.n_attributes
            X = X.copy() / self.scale
            self.Q_type = classifier.Q_type
            self.R_d = self._R_d_2(Y) if classifier.R_d_type == 2 else self._R_d_1(Y)
            self.k = classifier.k
            self.weights = classifier.weights
            self.index = classifier.nn_search.construct(X)
            self.Y = Y

        @staticmethod
        def _R_d_1(Y):
            return np.sum(Y[:, None, :] == Y, axis=-1)

        @staticmethod
        def _R_d_2(Y):
            p = np.sum(Y, axis=0)/len(Y)

            both = np.minimum(Y[:, None, :], Y)
            either = np.maximum(Y[:, None, :], Y)
            neither = 1 - either
            xeither = either - both

            numerator = both * (1 - p) + neither * p
            divisor = numerator + xeither * 0.5
            return np.sum(numerator, axis=-1)/np.sum(divisor, axis=-1)

        def query(self, X):
            X = X.copy() / self.scale
            neighbours, distances = self.index.query(X, self.k)
            R = np.maximum(1 - distances, 0)
            if self.Q_type == 1:
                Q = self._Q_1(neighbours, R)
            elif self.Q_type == 2:
                Q = self._Q_2(neighbours, R)
            else:
                Q = self._Q_1(neighbours, R) + self._Q_2(neighbours, R)
            Q_max = np.max(Q, axis=-1, keepdims=True)
            Q = Q == Q_max
            return np.sum(np.minimum(self.Y, Q[..., None]), axis=1) / np.sum(Q, axis=-1, keepdims=True)

        def _Q_1(self, neighbours, R):
            return self.weights.soft_min(np.minimum(1 - R[..., None] + self.R_d[neighbours, :] - 1, 1), axis=1)

        def _Q_2(self, neighbours, R):
            return self.weights.soft_max(np.maximum(R[..., None] + self.R_d[neighbours, :] - 1, 0), axis=1)
