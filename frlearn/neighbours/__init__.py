"""
The :mod:`frlearn.neighbours` subpackage implements fuzzy rough neighbour
algorithms.
"""

from .classification import FRNNClassifier, ImbalancedFRNNClassifier
from .neighbour_search import BallTree, KDTree, NNSearch
