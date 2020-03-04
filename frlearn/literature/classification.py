"""Classifiers from the literature"""
from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

from ..ensembles.classifiers import FuzzyRoughClassifier
from ..neighbours.approximators import ComplementedDistance
from ..neighbours.neighbour_search import KDTree, NNSearch
from ..utils.owa_operators import OWAOperator, additive


class FRNNClassifier(BaseEstimator, ClassifierMixin, ):
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

    def __init__(self, *, upper_weights: OWAOperator = additive(), upper_k: int = 20,
                 lower_weights: OWAOperator = additive(), lower_k: int = 20,
                 nn_search: NNSearch = KDTree()):
        super().__init__()
        upper_approximator = ComplementedDistance(owa=upper_weights, k=upper_k) if upper_weights else None
        lower_approximator = ComplementedDistance(owa=lower_weights, k=lower_k) if lower_weights else None
        self.classifier = FuzzyRoughClassifier(upper_approximator, lower_approximator, nn_search)

    def fit(self, X, y):
        self.model_ = self.classifier.construct(X, y)
        return self

    def query(self, X):
        return self.model_.query(X)

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
        vals = self.model_.query(X)
        return np.array(self.model_.classes)[np.argmax(vals, axis=1)]

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
        vals = self.model_.query(X)
        return self._get_probabilities(vals)

    def _get_probabilities(self, vals):
        # Divide probabilities evenly accross classes with infinite values, if present,
        # but leave np.nan alone, which has to be considered separately
        rows_inf = np.any(np.isposinf(vals), axis=1)
        vals[rows_inf] = np.where(np.isposinf(vals[rows_inf]), 1, vals[rows_inf])
        vals[rows_inf] = np.where(np.isfinite(vals[rows_inf]), 0, vals[rows_inf])
        # Divide probabilities evenly if all values are 0
        rows_0 = ~np.all(vals, axis=1)
        vals[rows_0] = 1
        return vals/np.sum(vals, axis=1, keepdims=True)
