"""
Preprocessors in fuzzy-rough-learn.
"""

import importlib

from .neighbours.preprocessors import FRFS, FRPS
try:
    from .networks.preprocessors import SAE
except ImportError:
    pass

__all__ = ['FRFS', 'FRPS', 'SAE']

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
