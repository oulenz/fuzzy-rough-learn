"""Nearest neighbour preprocessors"""
from __future__ import annotations

import numpy as np

from frlearn.base import Preprocessor
from frlearn.neighbours.neighbour_search import KDTree, NNSearch
from frlearn.utils.np_utils import fractional_k, remove_diagonal
from frlearn.utils.owa_operators import OWAOperator, invadd, deltaquadsigmoid
from frlearn.utils.t_norms import lukasiewicz


class FRFS(Preprocessor):
    """
    Implementation of the Fuzzy Rough Feature Selection (FRFS) preprocessor.

    Greedily selects features that induce the greatest increase in the size of the positive region,
    until it matches the size of the positive region with all features,
    or until the required number of features is selected.

    The positive region is defined as the union of the lower approximations of the decision classes in `X`.
    Its size is the sum of its membership values.

    Parameters
    ----------
    n_features : int or None, default=None
        Number of features to select. If None, will continue to add features until positive region size becomes maximal.
    owa_weights: OWAOperator, default=deltaquadsigmoid(0.2, 1)
        OWA weights to use for calculation of the soft minimum in the positive regions.
    t_norm : (ndarray, int, ) -> ndarray, default=lukasiewicz
        Function that takes an ndarray and a keyword argument `axis`,
        and returns an ndarray with the corresponding axis removed.
        Used to define the similarity relation `R` from the per-attribute similarities.
        Should be a t-norm, or else the size of the positive region may decrease as features are added.


    References
    ----------

    .. [1] `Cornelis C, Verbiest N, Jensen R (2011).
       Ordered Weighted Average Based Fuzzy Rough Sets
       In: Yu J, Greco S, Lingras P, Wang G, Skowron A (eds). Rough Set and Knowledge Technology. RSKT 2010.
       Lecture Notes in Computer Science, vol 6401. Springer, Berlin, Heidelberg.
       doi: 10.1007/978-3-642-16248-0_16
       <https://link.springer.com/chapter/10.1007/978-3-642-16248-0_16>`_
    """

    def __init__(self, n_features=None, owa_weights: OWAOperator = deltaquadsigmoid(0.2, 1), t_norm=lukasiewicz):
        self.n_features = n_features
        self.owa_weights = owa_weights
        self.t_norm = t_norm

    def process(self, X, y):
        X_scaled = X / np.std(X, axis=0)
        R_a = np.minimum(np.maximum(1 - np.abs(X_scaled[:, None, :] - X_scaled), 0), y[:, None, None] != y[:, None])
        POS_A_size = self._POS_size(R_a)
        selected_attributes = np.full(X.shape[-1], False)
        remaining_attributes = set(range(X.shape[-1]))
        best_size = 0
        condition = (lambda: np.sum(selected_attributes) < self.n_features) if self.n_features else (lambda: best_size < POS_A_size)
        while condition():
            best_size = 0
            for i in remaining_attributes:
                candidate = selected_attributes.copy()
                candidate[i] = True
                candidate_size = self._POS_size(R_a[..., candidate])
                if candidate_size > best_size:
                    best_size = candidate_size
                    new_attribute = i
            selected_attributes[new_attribute] = True
            remaining_attributes.remove(new_attribute)
        return X[:, selected_attributes], y

    def _POS_size(self, R_a):
        R = self.t_norm(R_a, axis=-1)
        return np.sum(self.owa_weights.soft_min(1 - R, k=fractional_k(1), axis=-1))


class FRPS(Preprocessor):
    """
    Implementation of the Fuzzy Rough Prototype Selection (FRPS) preprocessor.

    Calculates quality measure for each training instance,
    the values of which serve as potential thresholds for selecting instances.
    Each potential threshold and corresponding candidate instance set is evaluated
    by comparing the decision class of each instance with the decision class
    of its nearest neighbour within the candidate instance set (excluding itself).
    The candidate instance set with the highest number of matches is selected.

    Parameters
    ----------
    quality_measure : str {'upper', 'lower', 'both', }, default='lower'
        Quality measure to use for calculating thresholds.
        Either the upper approximation of the decision class of each attribute,
        the lower approximation, or the mean value of both.
        [2] recommends the lower approximation.
    aggr_R : (ndarray, int, ) -> ndarray, default=np.mean
        Function that takes an ndarray and a keyword argument `axis`,
        and returns an ndarray with the corresponding axis removed.
        Used to define the similarity relation `R` from the per-attribute similarities.
        [1] uses the Łukasiewicz t-norm,
        while [2] offers a choice between the Łukasiewicz t-norm, the mean and the minimum,
        and recommends the mean.
    owa_weights: OWAOperator, default=invadd()
        OWA weights to use for calculation of soft maximum and/or minimum in quality measure.
        [1] uses additive weights, while [2] uses inverse additive weights.
    nn_search : NNSearch, default=KDTree()
        Nearest neighbour search algorithm to use.

    Notes
    -----
    There are a number of implementation differences between [1] and [2],
    in each case the present implementation follows [2]:

    * [1] calculates upper and lower approximations using all instances,
      while [2] only calculates upper approximation membership over instances of the decision class,
      and lower approximation membership over instances of other decision classes.
      This affects over what length the weight vector is 'stretched'.
    * In addition, [2] excludes each instance from the calculation of its own upper approximation membership.
    * [1] uses additive weights, while [2] uses inverse additive weights.
    * [1] uses the Łukasiewicz t-norm to aggregate per-attribute similarities,
      while [2] recommends using the mean.

    In addition, there are implementation issues not addressed in [1] or [2]:

    * It is unclear what metric the nearest neighbour search should use.
      It seems reasonable that it should either correspond with the similarity relation `R`
      (and therefore incorporate the same aggregation strategy from per-attribute similarities),
      or that it should match whatever metric is used by nearest neighbour classifition subsequent to FRPS.
      By default, the present implementation uses manhattan distance on the scaled attribute values.
    * When the largest quality measure value corresponds to a singleton candidate instance set,
      it cannot be evaluated (because the single instance in that set has no nearest neighbour).
      Since this is an edge case that would not score highly anyway, it is simply excluded from consideration.


    References
    ----------

    .. [1] `Verbiest N, Cornelis C, Herrera F (2013).
       OWA-FRPS: A Prototype Selection Method Based on Ordered Weighted Average Fuzzy Rough Set Theory
       In: Ciucci D, Inuiguchi M, Yao Y, Ślęzak D, Wang G (eds).
       Rough Sets, Fuzzy Sets, Data Mining, and Granular Computing. RSFDGrC 2013.
       Lecture Notes in Computer Science, vol 8170. Springer, Berlin, Heidelberg.
       doi: 10.1007/978-3-642-41218-9_19
       <https://link.springer.com/chapter/10.1007/978-3-642-41218-9_19>`_
    .. [2] `Verbiest N (2014).
       Fuzzy Rough and Evolutionary Approaches to Instance Selection.
       Doctoral dissertation, Universiteit Gent.
       <https://biblio.ugent.be/publication/5671992>`_
    """
    def __init__(
            self,
            owa_weights: OWAOperator = invadd(),
            quality_measure: str = 'lower',
            aggr_R = np.mean,
            nn_search: NNSearch = KDTree(),
    ):
        self.nn_search = nn_search
        self.owa = owa_weights
        self.aggr_R = aggr_R
        self.quality_measure = quality_measure

    def process(self, X, y):
        classes = np.unique(y)
        Cs = [X[np.where(y == c)] for c in classes]
        X_unscaled = np.concatenate(Cs, axis=0)
        scale = np.amax(X_unscaled, axis=0) - np.amin(X_unscaled, axis=0)
        X = X_unscaled/scale
        Cs = [C/scale for C in Cs]

        y = np.concatenate([np.full(C.shape[0], c) for c, C in zip(classes, Cs)])

        if self.quality_measure == 'upper':
            Q = self._upper(Cs)
        elif self.quality_measure == 'lower':
            co_Cs = [X[np.where(y != c)] for c in classes]
            Q = self._lower(Cs, co_Cs)
        else:
            co_Cs = [X[np.where(y != c)] for c in classes]
            Q = (self._upper(Cs) + self._lower(Cs, co_Cs))/2

        best_acc = 0
        best_tau = 0
        for tau in np.unique(Q):
            if np.sum(Q >= tau) <= 1:
                continue
            nn_model = self.nn_search.construct(X[Q >= tau])
            neighbours = nn_model.query(X[Q >= tau], k=2)[0][:, 1]
            deselected = X[Q < tau]
            if len(deselected) >= 1:
                neighbours = np.concatenate([neighbours, nn_model.query(deselected, k=1)[0][:, 0]])
            S_y = y[Q >= tau]
            y_neighbours = S_y[neighbours]
            acc = np.sum(y_neighbours == np.concatenate([y[Q >= tau], y[Q < tau]]))
            if acc > best_acc or (acc == best_acc and tau < best_tau):
                best_acc = acc
                best_tau = tau
        return X_unscaled[Q >= best_tau], y[Q >= best_tau]

    def _upper(self, Cs):
        return np.concatenate([self.owa.soft_max(
            remove_diagonal(self.aggr_R(1 - np.abs(C[:, None, :] - C), axis=-1)),
            k=fractional_k(1), axis=-1
        ) for C in Cs], axis=0)

    def _lower(self, Cs, co_Cs):
        return np.concatenate([self.owa.soft_min(
            self.aggr_R(np.abs(C[:, None, :] - co_C), axis=-1),
            k=fractional_k(1), axis=-1
        ) for C, co_C in zip(Cs, co_Cs)], axis=0)
