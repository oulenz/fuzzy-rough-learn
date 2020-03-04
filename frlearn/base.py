"""Base classes"""

from __future__ import annotations

from abc import ABC, abstractmethod
from copy import copy

import numpy as np


class Approximator(ABC):

    @abstractmethod
    def __init__(self):
        pass

    def construct(self, *args, **kwargs):
        return self.Approximation(self, *args, **kwargs)

    class Approximation(ABC):

        @abstractmethod
        def __init__(self, approximator, *args, **kwargs):
            pass

        @abstractmethod
        def query(self, X):
            pass

        def copy(self, **attribute_values):
            other = copy(self)
            for a, v in attribute_values.items():
                setattr(other, a, v)
            return other


class Classifier(ABC):

    @abstractmethod
    def __init__(self):
        pass

    def construct(self, X, y, *args, **kwargs):
        return self.Model(self, X, y, *args, **kwargs)

    class Model(ABC):

        @abstractmethod
        def __init__(self, classifier, X, y, *args, **kwargs):
            self.classes = np.unique(y)
            self.n_classes = len(self.classes)
            self.n_attributes = X.shape[-1]

        @abstractmethod
        def query(self, X):
            pass

        def copy(self, **attribute_values):
            other = copy(self)
            for a, v in attribute_values.items():
                setattr(other, a, v)
            return other
