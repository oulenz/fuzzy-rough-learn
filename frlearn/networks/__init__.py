"""
The :mod:`frlearn.networks` subpackage implements neural networks algorithms.
"""


try:
    import tensorflow as tf
except ImportError:
    raise ImportError('Neural network algorithms require the tensorflow library.') from None

from .preprocessors import SAE
