"""
Generating functions for weight vectors parametrised by length.
These weight vectors should sum to one, and they typically descend,
as they are meant to be used with the
`soft_head`, `soft_max`, `soft_min` and `soft_tail` functions in `utilities.numpy`,
where the first weight corresponds respectively to the first, largest, smallest and last values.
However, since `k` may in general be less than the length of the relevant axis,
weight vectors that are not descending can also be useful.

Specifically, using these weights with `soft_max` gives an Ordered Weighted Averaging (OWA) operator,
while using the same weights with `soft_min` is equivalent to an OWA operator with dual weights.
"""

from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass
from typing import Callable

import numpy as np

__all__ = [
    'Weights', 'ConstantWeights', 'ExponentialWeights', 'LinearWeights',
    'QuantifierWeights', 'ReciprocallyLinearWeights',
]


class Weights:
    """
    Abstract base class for parametrisable weights functions. Classes that inherit from `Weights` should
    overwrite `__call__` with the weight function, while `__init__` can be used to set any parameters.

    Returns
    -------
    f: int -> np.array
        Function that takes a positive integer `k` and returns a weight vector of length `k`.

    References
    ----------
    .. [1] `Yager RR (1988).
       On Ordered Weighted Averaging Aggregation Operators in Multicriteria Decisionmaking.
       IEEE Transactions on Systems, Man, and Cybernetics, vol 18, no 1, pp 183–190.
       doi: 10.1109/21.87068
       <https://ieeexplore.ieee.org/document/87068>`_
    """

    @abstractmethod
    def __call__(self, k: int):
        pass


@dataclass
class ConstantWeights(Weights):
    """
    `(1/4, 1/4, 1/4, 1/4)`
    Also known as *mean* weights, as they compute the unweighted mean.

    Returns
    -------
    f: int -> np.array
        Function that takes a positive integer `k` and returns a weight vector of length `k`
        with constant weights.
    """

    def __call__(self, k: int):
        return np.full(k, 1 / k)


@dataclass
class ExponentialWeights(Weights):
    """
    `(8/15, 4/15, 2/15, 1/15)`
    Exponentially decreasing weights with parametrisable base.

    Parameters
    ----------
    base: float
        Exponential base. Should be larger than 1.

    Returns
    -------
    f: int -> np.array
        Function that takes a positive integer `k` and returns a weight vector of length `k`
        with exponentially decreasing weights with base `b`.

    Notes
    -----
    With base 2, weights rapidly approach 0, meaning:

    - the resulting weight vector is not very useful, and quickly becomes insensitive to increasing `k`,

    - using large values for `k` will produce weights that are so small as to cause computational wonkiness.

    These issues are exacerbated for larger bases, so bases only slightly larger than 1 may be most useful.
    """

    base: float = 2

    def __call__(self, k: int):
        w = np.flip(self.base ** np.arange(k))
        return w / np.sum(w)


@dataclass
class LinearWeights(Weights):
    """
    `(4/10, 3/10, 2/10, 1/10)`
    Also known as *additive* weights.

    Returns
    -------
    f: int -> np.array
        Function that takes a positive integer `k` and returns a weight vector of length `k`
        with linearly decreasing weights.
    """

    def __call__(self, k: int):
        return np.flip(2 * np.arange(1, k + 1) / (k * (k + 1)))


@dataclass
class QuantifierWeights(Weights):
    """
    Weights that encode a regular non-decreasing quantifier [1]_.

    Parameters
    ----------
    q: float -> float
        Regular non-decreasing quantifier (a surjective, non-decreasing function `[0, 1] -> [0, 1]`).

    Returns
    -------
    f: int -> np.array
        Function that takes a positive integer `k` and returns the weight vector of length `k` corresponding to `q`.

    References
    ----------
    .. [1] `Yager RR (1988).
       On Ordered Weighted Averaging Aggregation Operators in Multicriteria Decisionmaking.
       IEEE Transactions on Systems, Man, and Cybernetics, vol 18, no 1, pp 183–190.
       doi: 10.1109/21.87068
       <https://ieeexplore.ieee.org/document/87068>`_
    """

    q: Callable[[float], float]

    def __call__(self, k: int):
        return self.q(np.arange(1, k+1) / k) - self.q(np.arange(k) / k)


@dataclass
class ReciprocallyLinearWeights(Weights):
    """
    `(12/25, 12/50, 12/75, 12/100)`
    Also known as *inverse additive* weights.

    Returns
    -------
    f: int -> np.array
        Function that takes a positive integer `k` and returns a weight vector of length `k`
        with reciprocally linearly decreasing weights.
    """

    def __call__(self, k: int):
        return 1 / (np.arange(1, k + 1) * np.sum(1 / np.arange(1, k + 1)))
