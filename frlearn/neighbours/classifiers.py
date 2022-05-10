"""Nearest neighbour classifiers"""
from __future__ import annotations

from typing import Callable, List

import numpy as np

from frlearn.base import DataDescriptor, MultiClassClassifier, MultiLabelClassifier
from frlearn.data_descriptors import NND
from frlearn.neighbour_search_methods import NeighbourSearchMethod, KDTree
from frlearn.neighbours.utilities import resolve_k
from frlearn.statistics.feature_preprocessors import RangeNormaliser
from frlearn.array_functions import div_or, soft_max, soft_min
from frlearn.parametrisations import at_most, multiple
from frlearn.transformations import truncated_complement
from frlearn.uncategorised.utilities import resolve_dissimilarity
from frlearn.weights import ExponentialWeights, LinearWeights


class FuzzyRoughEnsemble(MultiClassClassifier):
    def __init__(
            self,
            upper_approximator: DataDescriptor | None,
            lower_approximator: DataDescriptor | None,
            preprocessors=()
    ):
        super().__init__(preprocessors=preprocessors)
        self.upper_approximator = upper_approximator
        self.lower_approximator = lower_approximator

    def _construct(self, X, y) -> Model:
        model: FuzzyRoughEnsemble.Model = super()._construct(X, y)
        Cs = [X[np.where(y == c)] for c in model.classes]
        model.upper_approximations = self.upper_approximator and [self.upper_approximator(C) for C in Cs]
        co_Cs = [X[np.where(y != c)] for c in model.classes]
        model.lower_approximations = self.lower_approximator and [self.lower_approximator(co_C) for co_C in co_Cs]
        return model

    class Model(MultiClassClassifier.Model):

        upper_approximations: List[DataDescriptor.Model] | None
        lower_approximations: List[DataDescriptor.Model] | None

        def _query(self, X):
            vals = []
            if self.upper_approximations:
                vals.append(np.stack([approximation(X) for approximation in self.upper_approximations], axis=1))
            if self.lower_approximations:
                vals.append(
                    1 - np.stack([approximation(X) for approximation in self.lower_approximations], axis=1))
            if len(vals) == 2:
                return sum(vals) / 2
            return vals[0]


class FRNN(FuzzyRoughEnsemble):
    r"""
    Implementation of Fuzzy Rough Nearest Neighbour (FRNN) classification.

    Parameters
    ----------
    upper_weights : (int -> np.array) or None = LinearWeights()
        OWA weights to use in calculation of upper approximation of decision classes.
        If `None`, only the `upper_k`\ th neighbour is used.

    upper_k: int or (int -> float) or None = at_most(20)
        Effective length of upper weights vector (number of nearest neighbours to consider).
        Should be either a positive integer,
        or a function that takes the class size `n` and returns a float,
        or None, which is resolved as `n`.
        All such values are rounded to the nearest integer in `[1, n]`.
        Alternatively, if this is 0, only the lower approximation is used.

    lower_weights : (int -> np.array) or None = LinearWeights()
        OWA weights to use in calculation of lower approximation of decision classes.
        If `None`, only the `lower_k`\ th neighbour is used.

    lower_k: int or (int -> float) or None = at_most(20)
        Effective length of lower weights vector (number of nearest neighbours to consider).
        Should be either a positive integer,
        or a function that takes the size `n` of the complement of the class and returns a float,
        or None, which is resolved as `n`.
        All such values are rounded to the nearest integer in `[1, n]`.
        Alternatively, if this is 0, only the upper approximation is used.

    dissimilarity: str or float or (np.array -> float) or ((np.array, np.array) -> float) = 'boscovich'
        The dissimilarity measure to use.
        The similarity between two instances is calculated as 1 minus their dissimilarity.

        A vector size measure `np.array -> float` induces a dissimilarity measure through application to `y - x`.
        A float is interpreted as Minkowski size with the corresponding value for `p`.
        For convenience, a number of popular measures can be referred to by name.

        When a float or string is passed, the corresponding dissimilarity measure is automatically scaled
        to ensure that the dissimilarity of `[1, 1, ..., 1]` with `[0, 0, ..., 0]` is 1.

        For instance, the default Boscovich norm (also known as cityblock, Manhattan or taxicab norm)
        normally assigns a dissimilarity that is the sum of the per-attribute differences.
        In this case, the scaling step divides by the number of dimensions,
        and we obtain a dissimilarity that is the mean of the per-attribute differences.

        This can be prevented by explicitly passing a dissimilarity measure without scaling.

    nn_search : NeighbourSearchMethod = KDTree()
        Nearest neighbour search algorithm to use.

    preprocessors : iterable = (RangeNormaliser(), )
        Preprocessors to apply. The default range normaliser ensures that all features have range 1.

    Notes
    -----
    If `upper_weights = lower_weights = None` and `upper_k = lower_k = 1`,
    this is the original strict FRNN classification as presented in [1]_.
    The use of OWA operators for the calculation of fuzzy rough sets was proposed in [2]_,
    and OWA operators were first explicitly combined with FRNN in [3]_.

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
    .. [3] `Ramentol E, Vluymans S, Verbiest N, Caballero Y, Bello R, Cornelis C, Herrera F (2015).
       IFROWANN: Imbalanced Fuzzy-Rough Ordered Weighted Average Nearest Neighbor Classification.
       IEEE Transactions on Fuzzy Systems, vol 23, no 5, pp 1622–1637.
       doi: 10.1109/TFUZZ.2014.2371472
       <https://ieeexplore.ieee.org/document/6960859>`_
    """
    def __init__(
            self, *,
            upper_weights: Callable[[int], np.array] or None = LinearWeights(),
            upper_k: int or Callable[[int], float] or None = at_most(20),
            lower_weights: Callable[[int], np.array] or None = LinearWeights(),
            lower_k: int or Callable[[int], float] or None = at_most(20),
            dissimilarity: str or float or Callable[[np.array], float] or Callable[[np.array, np.array], float] = 'boscovich',
            nn_search: NeighbourSearchMethod = KDTree(),
            preprocessors=(RangeNormaliser(), )
    ):
        dissimilarity = resolve_dissimilarity(dissimilarity, scale_by_dimensionality=True)
        upper_approximator = upper_k != 0 and NND(
            weights=upper_weights, k=upper_k, proximity=truncated_complement, dissimilarity=dissimilarity, nn_search=nn_search, preprocessors=()
        )
        lower_approximator = lower_k != 0 and NND(
            weights=lower_weights, k=lower_k, proximity=truncated_complement, dissimilarity=dissimilarity, nn_search=nn_search, preprocessors=()
        )
        super().__init__(upper_approximator, lower_approximator, preprocessors=preprocessors, )

    def _construct(self, X, y) -> Model:
        model = super()._construct(X, y)
        return model

    class Model(FuzzyRoughEnsemble.Model):
        pass


class FROVOCO(MultiClassClassifier):
    r"""
    Implementation of the Fuzzy Rough OVO COmbination (FROVOCO) ensemble classifier [1]_.

    FROVOCO decomposes muliclass classification into a number of one-vs-one and one-vs-rest comparisons.
    Depending on the imbalance ratio (ir) of each comparison,
    it uses different weights, following the IFROWANN scheme [2]_.

    Parameters
    ----------
    balanced_weights : (int -> np.array) or None = ExponentialWeights(base=2)
        OWA weights to use when the imbalance ratio is not larger than `ir_threshold`.
        If `None`, only the `balanced_k`\ th neighbour is used.

    balanced_k: int or (int -> float) or None = at_most(16)
        Length of the weights vector when the imbalance ratio is not larger than `ir_threshold`.
        Should be either a positive integer,
        or a function that takes the class size `n` and returns a float,
        or None, which is resolved as `n`.
        All such values are rounded to the nearest integer in `[1, n]`.
        Alternatively, if this is 0, only the lower approximation is used.

    imbalanced_weights : (int -> np.array) or None = LinearWeights()
        OWA weights to use when the imbalance ratio is larger than `ir_threshold`.
        If `None`, only the `imbalanced_k`\ th neighbour is used.

    imbalanced_k: int or (int -> float) or None = 0.1 * n
        Length of the weights vector when the imbalance ratio is larger than `ir_threshold`.
        Effective length of lower weights vector (number of nearest neighbours to consider).
        Should be either a positive integer,
        or a function that takes the size `n` of the complement of the class and returns a float,
        or None, which is resolved as `n`.
        All such values are rounded to the nearest integer in `[1, n]`.
        Alternatively, if this is 0, only the upper approximation is used.

    ir_threshold: float or None = 9
        Imbalance ratio threshold.  Above this value, `imbalanced_weights` and `imbalanced_k` are used,
        otherwise `balanced_weights` and `balanced_k`.
        Set to `None` to only use `balanced_weights` and `balanced_k`.

    dissimilarity: str or float or (np.array -> float) or ((np.array, np.array) -> float) = 'boscovich'
        The dissimilarity measure to use.
        The similarity between two instances is calculated as 1 minus their dissimilarity.

        A vector size measure `np.array -> float` induces a dissimilarity measure through application to `y - x`.
        A float is interpreted as Minkowski size with the corresponding value for `p`.
        For convenience, a number of popular measures can be referred to by name.

        When a float or string is passed, the corresponding dissimilarity measure is automatically scaled
        to ensure that the dissimilarity of `[1, 1, ..., 1]` with `[0, 0, ..., 0]` is 1.

        For instance, the default Boscovich norm (also known as cityblock, Manhattan or taxicab norm)
        normally assigns a dissimilarity that is the sum of the per-attribute differences.
        In this case, the scaling step divides by the number of dimensions,
        and we obtain a dissimilarity that is the mean of the per-attribute differences.

        This can be prevented by explicitly passing a dissimilarity measure without scaling.

    nn_search: NeighbourSearchMethod = KDTree()
        Nearest neighbour search algorithm to use.

    preprocessors : iterable = (RangeNormaliser(), )
        Preprocessors to apply. The default range normaliser ensures that all features have range 1.

    Notes
    -----
    The original proposal uses full length exponential weight vectors,
    but since exponential weights decrease exponentially,
    the contribution of these additional weights to the classification result
    also decreases exponentially, and they are liable to corrupt the calculation
    due to their small size.
    Therefore, the default length of the exponential weight vector is 16.
    The original behaviour can be obtained by setting `balanced_k` to `None`.

    References
    ----------
    .. [1] `Vluymans S, Fernández A, Saeys Y, Cornelis C, Herrera F (2018).
       Dynamic affinity-based classification of multi-class imbalanced data with one-versus-one decomposition:
       a fuzzy rough set approach.
       Knowledge and Information Systems, vol 56, pp 55–84.
       doi: 10.1007/s10115-017-1126-1
       <https://link.springer.com/article/10.1007/s10115-017-1126-1>`_
    .. [2] `Ramentol E, Vluymans S, Verbiest N, Caballero Y, Bello R, Cornelis C, Herrera F (2015).
       IFROWANN: Imbalanced Fuzzy-Rough Ordered Weighted Average Nearest Neighbor Classification.
       IEEE Transactions on Fuzzy Systems, vol 23, no 5, pp 1622–1637.
       doi: 10.1109/TFUZZ.2014.2371472
       <https://ieeexplore.ieee.org/document/6960859>`_
    """
    def __init__(
            self,
            balanced_weights: Callable[[int], np.array] or None = ExponentialWeights(base=2),
            balanced_k: int or Callable[[int], float] or None = at_most(16),
            imbalanced_weights: Callable[[int], np.array] or None = LinearWeights(),
            imbalanced_k: int or Callable[[int], float] or None = multiple(0.1),
            ir_threshold: float or None = 9,
            dissimilarity: str or float or Callable[[np.array], float] or Callable[[np.array, np.array], float] = 'boscovich',
            nn_search: NeighbourSearchMethod = KDTree(),
            preprocessors=(RangeNormaliser(), )
    ):
        super().__init__(preprocessors=preprocessors)
        dissimilarity = resolve_dissimilarity(dissimilarity, scale_by_dimensionality=True)
        self.ir_threshold = ir_threshold if ir_threshold is not None else np.inf
        self.balanced_approximator = NND(
            dissimilarity=dissimilarity, k=balanced_k, weights=balanced_weights, proximity=truncated_complement,
            nn_search=nn_search, preprocessors=()
        )
        self.imbalanced_approximator = NND(
            dissimilarity=dissimilarity, k=imbalanced_k, weights=imbalanced_weights, proximity=truncated_complement,
            nn_search=nn_search, preprocessors=()
        )

    def _construct(self, X, y) -> Model:
        model: FROVOCO.Model = super()._construct(X, y)
        model.ir_threshold = self.ir_threshold

        Cs = [X[np.where(y == c)] for c in model.classes]
        co_Cs = [X[np.where(y != c)] for c in model.classes]

        class_sizes = np.array([len(C) for C in Cs])
        model.ovo_ir = (class_sizes[:, None] / class_sizes)
        model.ovr_ir = np.array([c_n / (len(X) - c_n) for c_n in class_sizes])
        max_ir = np.max(model.ovo_ir, axis=1)

        model.imb_approxs = [
            self.imbalanced_approximator(C) if ir > self.ir_threshold else None for ir, C in zip(max_ir, Cs)
        ]
        model.bal_approxs = [
            self.balanced_approximator(C) if ir <= self.ir_threshold else None for ir, C in zip(model.ovr_ir, Cs)
        ]
        model.co_approxs = [
            (self.imbalanced_approximator if 1 / ir > self.ir_threshold else self.balanced_approximator)(co_C)
            for ir, co_C in zip(model.ovr_ir, co_Cs)
        ]

        model.sig = np.array([model._sig(C) for C in Cs])
        return model


    class Model(MultiClassClassifier.Model):

        ir_threshold: float
        ovo_ir: np.array
        ovr_ir: np.array
        imb_approxs: List[DataDescriptor.Model or None]
        bal_approxs: List[DataDescriptor.Model or None]
        co_approxs: List[DataDescriptor.Model]
        sig: np.array

        def _sig(self, C):
            approxs = [
                imb if ir > self.ir_threshold else bal
                for ir, imb, bal in zip(self.ovr_ir, self.imb_approxs, self.bal_approxs)
            ]
            vals_C = np.array([np.mean(a(C)) for a in approxs])
            co_vals_C = np.array([np.mean(co_a(C)) for co_a in self.co_approxs])
            return (vals_C + 1 - co_vals_C)/2

        def _query(self, X):
            # The values in the else clause are just placeholders. But we can't use `None`, because that will force
            # the dtype of the resulting array to become `object`, which will in turn lead to 0/0 producing
            # ZeroDivisionError rather than np.nan
            imb_vals_X = np.stack(np.broadcast_arrays(*[a(X) if a else -np.inf for a in self.imb_approxs])).transpose()
            bal_vals_X = np.stack(np.broadcast_arrays(*[a(X) if a else -np.inf for a in self.bal_approxs])).transpose()
            co_vals_X = np.array([a(X) for a in self.co_approxs]).transpose()

            mem = self._mem(imb_vals_X, bal_vals_X, co_vals_X)

            mse = np.mean((mem[:, None, :] - self.sig) ** 2, axis=-1)
            mse_n = mse/np.sum(mse, axis=-1, keepdims=True)

            wv = self._wv(imb_vals_X, bal_vals_X)

            return (wv + mem)/2 - mse_n/self.n_classes

        def _mem(self, imb_vals, bal_vals, co_vals):
            vals = np.where(self.ovr_ir > self.ir_threshold, imb_vals, bal_vals)
            return (vals + 1 - co_vals) / 2

        def _wv(self, imb_vals, bal_vals):
            # Subtract from 1 because we're using lower approximations.
            vals = 1 - np.where(self.ovo_ir > self.ir_threshold, imb_vals[..., None], bal_vals[..., None])
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

    k : int, default = at_most(20)
        Number of neighbours to consider for neighbourhood consensus.

    owa_weights: (int -> np.array) = LinearWeights()
        OWA weights to use for calculation of soft maximum and/or minimum.

    dissimilarity: str or float or (np.array -> float) or ((np.array, np.array) -> float) = 'boscovich'
        The dissimilarity measure to use.
        The similarity between two instances is calculated as 1 minus their dissimilarity.

        A vector size measure `np.array -> float` induces a dissimilarity measure through application to `y - x`.
        A float is interpreted as Minkowski size with the corresponding value for `p`.
        For convenience, a number of popular measures can be referred to by name.

        When a float or string is passed, the corresponding dissimilarity measure is automatically scaled
        to ensure that the dissimilarity of `[1, 1, ..., 1]` with `[0, 0, ..., 0]` is 1.

        For instance, the default Boscovich norm (also known as cityblock, Manhattan or taxicab norm)
        normally assigns a dissimilarity that is the sum of the per-attribute differences.
        In this case, the scaling step divides by the number of dimensions,
        and we obtain a dissimilarity that is the mean of the per-attribute differences.

        This can be prevented by explicitly passing a dissimilarity measure without scaling.

    nn_search: NeighbourSearchMethod = KDTree()
        Nearest neighbour search algorithm to use.

    preprocessors : iterable = (RangeNormaliser(), )
        Preprocessors to apply. The default range normaliser ensures that all features have range 1.

    References
    ----------

    .. [1] `Vluymans S, Cornelis C, Herrera F, Saeys Y (2018).
       Multi-label classification using a fuzzy rough neighborhood consensus.
       Information Sciences, vol 433, pp 96–114.
       doi: 10.1016/j.ins.2017.12.034
       <https://www.sciencedirect.com/science/article/pii/S002002551731157X>`_
    """

    def __init__(
            self, Q_type: int = 2, R_d_type: int = 1,
            k: int = at_most(20), owa_weights: Callable[[int], np.array] | None = LinearWeights(),
            dissimilarity: str or float or Callable[[np.array], float] or Callable[[np.array, np.array], float] = 'boscovich',
            nn_search: NeighbourSearchMethod = KDTree(),
            preprocessors=(RangeNormaliser(), )
    ):
        super().__init__(preprocessors=preprocessors)
        self.Q_type = Q_type
        self.R_d_type = R_d_type
        self.k = k
        self.owa_weights = owa_weights
        self.dissimilarity = resolve_dissimilarity(dissimilarity, scale_by_dimensionality=True)
        self.nn_search = nn_search

    def _construct(self, X, Y) -> Model:
        model: FRONEC.Model = super()._construct(X, Y)
        model.Q_type = self.Q_type
        model.R_d = model._R_d_2(Y) if self.R_d_type == 2 else model._R_d_1(Y)
        model.k = resolve_k(self.k, len(X))
        model.owa_weights = self.owa_weights
        model.nn_model = self.nn_search(X, dissimilarity=self.dissimilarity)
        model.Y = Y
        return model

    class Model(MultiLabelClassifier.Model):

        Q_type: int
        R_d: np.array
        k: int
        owa_weights: Callable[[int], np.array] | None
        nn_model: NeighbourSearchMethod.Model
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

        def _query(self, X):
            neighbours, distances = self.nn_model(X, self.k)
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
            return soft_min(vals, self.owa_weights, k=None, axis=1)

        def _Q_2(self, neighbours, R):
            vals = np.maximum(R[..., None] + self.R_d[neighbours, :] - 1, 0)
            return soft_max(vals, self.owa_weights, k=None, axis=1)
