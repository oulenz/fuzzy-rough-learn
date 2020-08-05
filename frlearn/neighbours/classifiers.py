"""Nearest neighbour classifiers"""
from __future__ import annotations

from typing import List, Optional

import numpy as np

from frlearn.base import Descriptor, MultiClassClassifier, MultiLabelClassifier
from frlearn.neighbours.descriptors import NND
from frlearn.neighbours.neighbour_search import KDTree, NNSearch
from frlearn.utils.np_utils import div_or, fractional_k, truncated_complement
from frlearn.utils.owa_operators import OWAOperator, additive, exponential


class FuzzyRoughEnsemble(MultiClassClassifier):
    def __init__(
            self,
            upper_approximator: Descriptor = NND(k=40, owa=additive(), proximity=truncated_complement),
            lower_approximator: Descriptor = NND(k=40, owa=additive(), proximity=truncated_complement),
    ):
        self.upper_approximator = upper_approximator
        self.lower_approximator = lower_approximator

    def construct(self, X, y):
        model = super().construct(X, y)
        Cs = [X[np.where(y == c)] for c in model.classes]
        model.upper_approximations = [self.upper_approximator.construct(C) for C in Cs]
        co_Cs = [X[np.where(y != c)] for c in model.classes]
        model.lower_approximations = [self.lower_approximator.construct(co_C) for co_C in co_Cs]
        return model

    class Model(MultiClassClassifier.Model):

        upper_approximations: List[Descriptor.Description]
        lower_approximations: List[Descriptor.Description]

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
    upper_weights : OWAOperator, default=additive()
        OWA weights to use in calculation of upper approximation of decision classes.

    upper_k : int, default = 20
        Effective length of upper weights vector (number of nearest neighbours to consider).

    lower_weights : OWAOperator, default=additive()
        OWA weights to use in calculation of lower approximation of decision classes.

    lower_k : int, default = 20
        Effective length of lower weights vector (number of nearest neighbours to consider).

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
        upper_approximator = NND(owa=upper_weights, k=upper_k, proximity=truncated_complement, nn_search=nn_search) if upper_weights else None
        lower_approximator = NND(owa=lower_weights, k=lower_k, proximity=truncated_complement, nn_search=nn_search) if lower_weights else None
        super().__init__(upper_approximator, lower_approximator)


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
        self.exponential_approximator = NND(owa=exponential(), k=fractional_k(1), proximity=truncated_complement, nn_search=nn_search)
        self.additive_approximator = NND(owa=additive(), k=fractional_k(.1), proximity=truncated_complement, nn_search=nn_search)

    def construct(self, X, y):
        model = super().construct(X, y)

        model.scale = (np.max(X, axis=0) - np.min(X, axis=0)) * model.n_attributes
        X = X.copy() / model.scale

        Cs = [X[np.where(y == c)] for c in model.classes]
        co_Cs = [X[np.where(y != c)] for c in model.classes]

        class_sizes = np.array([len(C) for C in Cs])
        model.ovo_ir = (class_sizes[:, None] / class_sizes)
        model.ova_ir = np.array([c_n / (len(X) - c_n) for c_n in class_sizes])
        max_ir = np.max(model.ovo_ir, axis=1)

        add_costr = self.additive_approximator.construct
        exp_costr = self.exponential_approximator.construct
        model.add_approx = [add_costr(C) if ir > 9 else None for ir, C in zip(max_ir, Cs)]
        model.exp_approx = [exp_costr(C) if ir <= 9 else None for ir, C in zip(model.ova_ir, Cs)]
        model.co_approx = [(add_costr if 1/ir > 9 else exp_costr)(co_C) for ir, co_C in zip(model.ova_ir, co_Cs)]

        model.sig = np.array([model._sig(C) for C in Cs])
        return model


    class Model(MultiClassClassifier.Model):

        scale: np.array
        ovo_ir: np.array
        ova_ir: np.array
        add_approx: List[Optional[Descriptor.Description]]
        exp_approx: List[Optional[Descriptor.Description]]
        co_approx: List[Descriptor.Description]
        sig: np.array

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
            # Subtract from 1 because we're using lower approximations.
            vals = 1 - np.where(self.ovo_ir > 9, additive_vals[..., None], exponential_vals[..., None])
            tot_vals = vals + vals.transpose(0, 2, 1)
            vals = div_or(vals, tot_vals, 0.5)
            # Exclude comparisons of a class with itself.
            vals[:, range(self.n_classes), range(self.n_classes)] = 0
            # Sum along axis 1 (rather than 2) because we're using lower approximations.
            return np.sum(vals, axis=1)


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
    owa_weights: OWAOperator, default=additive()
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
                 k: int = 20, owa_weights: OWAOperator = additive(), nn_search: NNSearch = KDTree()):
        self.Q_type = Q_type
        self.R_d_type = R_d_type
        self.k = k
        self.owa_weights = owa_weights
        self.nn_search = nn_search

    def construct(self, X, Y):
        model = super().construct(X, Y)
        model.scale = (np.max(X, axis=0) - np.min(X, axis=0)) * model.n_attributes
        X = X.copy() / model.scale
        model.Q_type = self.Q_type
        model.R_d = model._R_d_2(Y) if self.R_d_type == 2 else model._R_d_1(Y)
        model.k = self.k
        model.owa_weights = self.owa_weights
        model.index = self.nn_search.construct(X)
        model.Y = Y
        return model


    class Model(MultiLabelClassifier.Model):

        scale: np.array
        Q_type: int
        R_d: np.array
        k: int
        owa_weights: OWAOperator
        index: NNSearch.Index
        Y: np.array

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
            vals = np.minimum(1 - R[..., None] + self.R_d[neighbours, :] - 1, 1)
            return self.owa_weights.soft_min(vals, k=fractional_k(1), axis=1)

        def _Q_2(self, neighbours, R):
            vals = np.maximum(R[..., None] + self.R_d[neighbours, :] - 1, 0)
            return self.owa_weights.soft_max(vals, k=fractional_k(1), axis=1)
