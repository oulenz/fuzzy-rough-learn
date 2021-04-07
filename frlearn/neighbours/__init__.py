"""
The :mod:`frlearn.neighbours` subpackage implements nearest neighbour algorithms.
"""

from .descriptors import NND, LNND, LOF
from .classifiers import FuzzyRoughEnsemble, FRNN, FRONEC, FROVOCO
from .neighbour_search import BallTree, KDTree, NNSearch
from .preprocessors import FRFS, FRPS
