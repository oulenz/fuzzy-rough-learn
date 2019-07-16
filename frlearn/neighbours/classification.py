"""Fuzzy Rough Nearest Neighbour Classification"""

import numpy as np

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.neighbors.base import SupervisedIntegerMixin
from sklearn.utils import check_array

from .neighbour_search import KDTree, NNSearch
from .owa_operators import OWAOperator, additive
from ..utils import argmax_and_max


class FRNNClassifierBase(BaseEstimator, SupervisedIntegerMixin,
                         ClassifierMixin):

    """
    Base class for fuzzy rough nearest neighbour classifiers.
    """

    def __init__(self, *, nn_search: NNSearch):
        self.nn_search = nn_search

    def _fit(self, X):
        self.n_classes_ = len(self.classes_)
        self.n_attributes_ = X.shape[-1]

        self.scale_ = np.max(X, axis=0) - np.min(X, axis=0)
        X = (X / self.scale_) / self.n_attributes_

        # group by class
        Cs = [X[np.where(self._y == i)] for i in range(self.n_classes_)]
        self.class_counts_ = [len(C) for C in Cs]
        self.nn_indices_ = [self.nn_search.construct(C) for C in Cs]

        return self

    def _get_approximation_memberships(self, X):
        X = check_array(X, accept_sparse='csr')

        X = (X / self.scale_) / self.n_attributes_

        distances = [
            np.pad(index.query(X, min(k, max_k)),
                   ((0, 0), (0, max(0, k - max_k)))
                   , mode='constant', constant_values=1)
            for index, k, max_k
            in zip(self.nn_indices_, self.ks_, self.class_counts_)]

        vals = []
        if self.upper_weights_:
            similarities = [1 - d for d in distances]
            upper_approximations = np.array([w.soft_max(s) if w else np.full(len(s), np.nan) for s, w in zip(similarities, self.upper_weights_)])
            vals.append(upper_approximations)

        if self.lower_weights_:
            co_distances = [np.concatenate([dist for j, dist in enumerate(distances) if j != i]) for i in range(self.n_classes_)]
            lower_approximations = np.array([w.soft_min(d) if w else np.full(len(d), np.nan) for d, w in zip(co_distances, self.lower_weights_)])
            vals.append(lower_approximations)

        # take the average if using both upper and lower approximation
        vals = np.stack(vals, axis=0)
        vals = np.nansum(vals, axis=0)/np.sum(~np.isnan(vals), axis=0)

        return vals

    def predict(self, X):
        """
        Predict the class labels for the instances in X.

        Parameters
        ----------
        X : array shape=(n_instances, n_features, )
            Query instances.

        Returns
        -------
        y : array shape=(n_instances, )
            Class label for each query instance.
        """
        vals = self._get_approximation_memberships(X)
        return np.array(self.classes_)[np.argmax(vals, axis=0)]

    def predict_proba(self, X):
        """
        Calculate probability estimates for the instances in X.

        Parameters
        ----------
        X : array shape=(n_instances, n_features, )
            Query instances.

        Returns
        -------
        p : array shape=(n_instances, n_classes, )
            The class probabilities of the query instances. Classes are ordered
            by lexicographic order.
        """
        # normalise membership degrees into confidence scores
        vals = self._get_approximation_memberships(X)
        return vals / np.sum(vals, axis=0)

    def predict_with_confidence(self, X):
        """
        Predict the class labels for the instances in X and include the
        corresponding probability estimates.

        Parameters
        ----------
        X : array shape=(n_instances, n_features, )
            Query instances.

        Returns
        -------
        y : array shape=(n_instances, )
            Class labels for each query instance.

        p : array shape=(n_instances, ).
            The probability of the predicted class for each query instance.
        """
        # normalise membership degrees into confidence scores
        vals = self._get_approximation_memberships(X)
        indices, scores = argmax_and_max(vals, axis=0)
        classes = np.array(self.classes_)[indices]
        scores = scores / np.sum(vals, axis=0)
        return classes, scores

class FRNNClassifier(FRNNClassifierBase):
    """
    Classifier implementing fuzzy rough nearest neighbour classification
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

    .. [1] `Jensen R., Cornelis C. (2008) A New Approach to Fuzzy-Rough Nearest
       Neighbour Classification. In: Chan CC., Grzymala-Busse J.W., Ziarko W.P.
       (eds) Rough Sets and Current Trends in Computing. RSCTC 2008. Lecture
       Notes in Computer Science, vol 5306. Springer, Berlin, Heidelberg. doi:
       10.1007/978-3-540-88425-5_32
       <https://link.springer.com/chapter/10.1007/978-3-540-88425-5_32>`_
    .. [2] `Cornelis C., Verbiest N., Jensen R. (2010) Ordered Weighted Average
       Based Fuzzy Rough Sets. In: Yu J., Greco S., Lingras P., Wang G.,
       Skowron A. (eds) Rough Set and Knowledge Technology. RSKT 2010. Lecture
       Notes in Computer Science, vol 6401. Springer, Berlin, Heidelberg. doi:
       10.1007/978-3-642-16248-0_16
       <https://link.springer.com/chapter/10.1007/978-3-642-16248-0_16>`_
    .. [3] `E. Ramentol et al., "IFROWANN: Imbalanced Fuzzy-Rough Ordered
       Weighted Average Nearest Neighbor Classification," in IEEE Transactions
       on Fuzzy Systems, vol. 23, no. 5, pp. 1622-1637, Oct. 2015. doi:
       10.1109/TFUZZ.2014.2371472
       <https://ieeexplore.ieee.org/document/6960859>`_
    """

    def __init__(self, *,
                 upper_weights: OWAOperator = additive(20),
                 lower_weights: OWAOperator = None,
                 nn_search: NNSearch = KDTree()):
        super().__init__(nn_search=nn_search)
        self.upper_weights = upper_weights
        self.lower_weights = lower_weights

    def _fit(self, X):
        super()._fit(X)
        k = max(self.upper_weights.k if self.upper_weights else 0, self.lower_weights.k if self.lower_weights else 0)
        self.ks_ = [k for _ in self.classes_]
        self.upper_weights_ = [self.upper_weights for _ in self.classes_] if self.upper_weights else None
        self.lower_weights_ = [self.lower_weights for _ in self.classes_] if self.lower_weights else None
        return self

class ImbalancedFRNNClassifier(FRNNClassifierBase):
    """
    Classifier implementing imbalanced binary fuzzy rough nearest neighbour
    classification.

    Parameters
    ----------
    neg_upper_weights : OWAOperator, default=additive(20)
        OWA operator to use in calculation of the upper approximation of
        the negative class.

    neg_lower_weights : OWAOperator, default=None
        OWA operator to use in calculation of the lower approximation of
        the negative class.

    pos_upper_weights : OWAOperator, default=additive(20)
        OWA operator to use in calculation of the upper approximation of
        the positive class.

    pos_lower_weights : OWAOperator, default=None
        OWA operator to use in calculation of the lower approximation of
        the positive class.

    nn_search : NNSearch, default=KDTree()
        Nearest neighbour search algorithm to use.

    Notes
    -----
    This allows for the implementation of the proposal in [1]_.

    References
    ----------

    .. [1] `E. Ramentol et al., "IFROWANN: Imbalanced Fuzzy-Rough Ordered
       Weighted Average Nearest Neighbor Classification," in IEEE Transactions
       on Fuzzy Systems, vol. 23, no. 5, pp. 1622-1637, Oct. 2015. doi:
       10.1109/TFUZZ.2014.2371472
       <https://ieeexplore.ieee.org/document/6960859>`_
    """

    def __init__(self, *,
                 neg_upper_weights: OWAOperator = additive(20),
                 neg_lower_weights: OWAOperator = None,
                 pos_upper_weights: OWAOperator = additive(20),
                 pos_lower_weights: OWAOperator = None,
                 nn_search: NNSearch = KDTree()):
        super().__init__(nn_search=nn_search)
        self.neg_upper_weights = neg_upper_weights
        self.pos_upper_weights = pos_upper_weights
        self.neg_lower_weights = neg_lower_weights
        self.pos_lower_weights = pos_lower_weights

    def _fit(self, X):
        super()._fit(X)
        try:
            assert len(self.class_counts_) == 2
        except AssertionError:
            raise NotImplementedError(
                'Tried to fit dataset with {} classes. Multiclass'
                'classification with ImbalancedFRNClassifier is not yet '
                'implemented'.format(len(self.class_counts_)))
        if self.class_counts_[0] < self.class_counts_[1]:
            self.upper_weights_ = (self.pos_upper_weights, self.neg_upper_weights)
            self.lower_weights_ = (self.pos_lower_weights, self.neg_lower_weights)
        else:
            self.upper_weights_ = (self.neg_upper_weights, self.pos_upper_weights)
            self.lower_weights_ = (self.neg_lower_weights, self.pos_lower_weights)
        self.ks_ = [max(u.k if u else 0, l.k if l else 0)
                   for u, l in zip(self.upper_weights_, self.lower_weights_[::-1])]
        return self
