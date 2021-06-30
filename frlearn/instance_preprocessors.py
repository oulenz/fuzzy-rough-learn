"""
Instance preprocessors in fuzzy-rough-learn.
"""

import importlib

from .neighbours.instance_preprocessors import FRPS

__all__ = [
    'FRPS',
]

dependencies = {
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
