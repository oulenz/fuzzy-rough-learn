"""
Feature preprocessors in fuzzy-rough-learn.
"""

import importlib

from .neighbours.feature_preprocessors import FRFS
try:
    from .networks.feature_preprocessors import SAE
except ImportError:
    pass
from .statistics.feature_preprocessors import LinearNormaliser, IQRNormaliser, MaxAbsNormaliser, RangeNormaliser, Standardiser
from .utilities.feature_preprocessors import NormNormaliser

__all__ = [
    'FRFS',
    'IQRNormaliser',
    'LinearNormaliser',
    'MaxAbsNormaliser',
    'NormNormaliser',
    'RangeNormaliser',
    'SAE',
    'Standardiser',
]

dependencies = {
    'SAE': 'tensorflow',
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
