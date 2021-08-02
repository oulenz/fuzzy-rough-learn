"""
Feature preprocessors in fuzzy-rough-learn.
"""

import importlib

_to_import = [
    ('neighbours', 'FRFS', [],),
    ('networks', 'SAE', ['tensorflow'],),
    ('statistics', 'IQRNormaliser', [],),
    ('statistics', 'LinearNormaliser', [],),
    ('statistics', 'MaxAbsNormaliser', [],),
    ('statistics', 'RangeNormaliser', [],),
    ('statistics', 'Standardiser', [],),
    ('uncategorised', 'VectorSizeNormaliser', [],),
]

_content = {}
_missing_dependencies = {}
__all__ = []

for package, name, dependencies in _to_import:
    for dependency in dependencies:
        try:
            importlib.import_module(dependency)
        except ImportError:
            _missing_dependencies[name] = dependency
            break
    else:
        module = importlib.import_module(f'frlearn.{package}.feature_preprocessors')
        _content[name] = getattr(module, name)
        __all__.append(name)
        continue


def __getattr__(name):
    if name in _content:
        return _content[name]
    if name in _missing_dependencies:
        raise ImportError(f'{name} requires the optional dependency {_missing_dependencies[name]}') from None
    raise AttributeError(f"module {__name__} has no attribute {name}")
