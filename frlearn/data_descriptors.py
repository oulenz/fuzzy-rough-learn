"""
Data descriptors in fuzzy-rough-learn.
"""

import importlib

from .neighbours.data_descriptors import ALP, LNND, LOF, NND
from .statistics.data_descriptors import CD, MD
from .support_vectors.data_descriptors import SVM
from .trees.data_descriptors import EIF, IF


__all__ = ['ALP', 'CD', 'EIF', 'IF', 'LNND', 'LOF', 'MD', 'NND', 'SVM']

dependencies = {
    'EIF': 'eif',
}


def __getattr__(name):
    if name in __all__:
        dependency = dependencies.get(name)
        if dependency:
            try:
                importlib.import_module(dependency)
            except ImportError:
                raise ImportError(f'{name} requires the optional dependency {dependency}') from None
        return globals()[name]
    raise AttributeError(f"module {__name__} has no attribute {name}")
