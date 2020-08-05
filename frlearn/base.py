"""Base classes"""

from __future__ import annotations

from abc import ABC, abstractmethod
from copy import copy
from typing import Union

import numpy as np

from sklearn.base import BaseEstimator, ClassifierMixin


class Descriptor(ABC):

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def construct(self, X):
        return self.Description.__new__(self.Description)

    class Description(ABC):

        @abstractmethod
        def query(self, X):
            pass


class Classifier(ABC):

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def construct(self, X, _):
        model = self.Model.__new__(self.Model)
        model.n_attributes = X.shape[-1]
        return model

    class Model(ABC):

        n_attributes: int

        @abstractmethod
        def query(self, X):
            pass


class MultiClassClassifier(Classifier):

    def construct(self, X, y):
        model = super().construct(X, y)
        model.classes = np.unique(y)
        model.n_classes = len(model.classes)
        return model

    class Model(Classifier.Model):

        classes: np.array
        n_classes: int


class MultiLabelClassifier(Classifier):

    def construct(self, X, Y):
        model = super().construct(X, Y)
        model.n_classes = Y.shape[1]
        return model

    class Model(Classifier.Model):

        n_classes: int


class Preprocessor(ABC):

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def process(self, X, y):
        pass


def select_class(scores, abstention_threshold: float = -1, labels=None):
    """
    Convert an array of class scores into class predictions, by selecting the class with the highest score.
    If none of the scores is greater than `abstention_threshold`, a generic `other` class will be predicted.
    
    Parameters
    ----------
    scores : array shape=(n_instances, n_classes, )
        Array of class scores. Scores should be values in `[0, 1]`
    abstention_threshold : float=-1
        Threshold to use for predicting one of the classes.
    labels : array shape={(n_classes, ), (n_classes + 1, )}, default=None
        Labels of the classes in `scores` to be used in the return array. The first label is used for abstention,
        it may be omitted if `abstention_threshold == 0`. If `None`, positions are used instead, with 0 used
        for abstention if `abstention_threshold >= 0`.

    Returns
    -------
    predictions : array shape=(n_instances, )
        Class label for each query instance.
    
    """
    if abstention_threshold >= 0:
        scores = np.concatenate([np.broadcast_to(abstention_threshold, (len(scores), 1)), scores], axis=-1)
    predictions = np.argmax(scores, axis=-1)
    if labels is not None:
        predictions = labels[predictions]
    return predictions


def discretise(scores, threshold: float = 0.5, ):
    """
    Discretise an array of label scores in `[0, 1]` into discrete predictions in `{0, 1}`,
    by selecting all labels that score higher than `threshold`.

    Parameters
    ----------
    scores : array shape=(n_instances, n_classes, )
        Array of class scores. Scores should be values in `[0, 1]`
    threshold : float=0.5
        Threshold to use for selecting labels.

    Returns
    -------
    predictions : array shape=(n_instances, )
        Class label for each query instance.

    """
    return scores >= threshold


def probabilities_from_scores(scores):
    """
    Rescale an array of class scores into probabilities that sum to 1, by dividing each score by the total sum.
    If all scores are zero, probabilities are assigned equally (`1/n_classes`).

    Parameters
    ----------
    scores : array shape=(n_instances, n_classes, )
        Array of class scores.

    Returns
    -------
    probabilities : array shape=(n_instances, n_classes, )
        Array of class probabilities.

    """
    rows_0 = ~np.all(scores, axis=1)
    scores[rows_0] = 1
    return scores/np.sum(scores, axis=1, keepdims=True)


class FitPredictClassifier(BaseEstimator, ClassifierMixin, ):
    """
    Convenience class for using any classifier as a scikit-learn-style classifier with fit and predict methods.

    Parameters
    ----------
    classifier_or_class : {type, Classifier}
        Either an initialised Classifier, or a Classifier subclass. If a Classifier subclass, will be initialised with
        all remaining positional and keyword arguments.
    """

    def __init__(self, classifier_or_class: Union[type(Classifier), Classifier], *args, **kwargs):
        super().__init__()
        if isinstance(classifier_or_class, Classifier):
            self.classifier = classifier_or_class
        self.classifier = classifier_or_class(*args, **kwargs)

    def fit(self, X, y):
        """
        Fit the model using X as training data and y as target values

        Parameters
        ----------
        X : array shape=(n_instances, n_features, )
            Training data. If array or matrix, shape [n_samples, n_features],
            or [n_samples, n_samples] if metric='precomputed'.

        y : {array-like, sparse matrix}
            Target values of shape = [n_samples] or [n_samples, n_outputs]

        """
        self.model_ = self.classifier.construct(X, y)
        return self

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
        scores = self.model_.query(X)
        return select_class(scores, labels=self.model_.classes)

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
        scores = self.model_.query(X)
        return probabilities_from_scores(scores)
