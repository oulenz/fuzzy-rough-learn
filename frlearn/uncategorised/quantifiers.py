"""Functions from [0, 1] to [0, 1]."""

from dataclasses import dataclass

import numpy as np


@dataclass
class QuadraticSigmoid():
    """
    Sigmoid function formed from two quadratic curves [1]_.

    Parameters
    ----------
    α: float
        Start of the sigmoid curve. Should be a value in `[0, 1]`, smaller than `β`.
    β: float
        End of the sigmoid curve. Should be a value in `[0, 1]`, larger than `α`.

    References
    ----------
    .. [1] `Cornelis C, Verbiest N, Jensen R (2011).
       Ordered Weighted Average Based Fuzzy Rough Sets
       In: Yu J, Greco S, Lingras P, Wang G, Skowron A (eds). Rough Set and Knowledge Technology. RSKT 2010.
       Lecture Notes in Computer Science, vol 6401. Springer, Berlin, Heidelberg.
       doi: 10.1007/978-3-642-16248-0_16
       <https://link.springer.com/chapter/10.1007/978-3-642-16248-0_16>`_
    """

    α: float
    β: float

    def __call__(self, a: np.array):
        α, β = self.α, self.β
        return np.where(a <= α, 0,
                        np.where(a >= β, 1,
                                 np.where(a <= (α + β) / 2, 2 * (a - α) ** 2 / (β - α) ** 2,
                                          1 - 2 * (a - β) ** 2 / (β - α) ** 2)))


@dataclass
class MoreThan():
    """"
    A RIM quantifier that represents the quantifier “more than 100 ∗ k%”[1]

    Parameters
    ----------
    k: float
       Should be a value in `[0, 1]`

    References
    ---------
    [1] THEERENS, Adnan et CORNELIS, Chris.
    Fuzzy rough sets based on fuzzy quantification.
    Fuzzy Sets and Systems, 2023, vol. 473, p. 108704.
"""

    k: float

    def __call__(self, p: float):
        k = self.k
        if p > k:
            return 1
        else:
            return 0


@dataclass
class AtLeast():
    """"
        A RIM quantifier that represent the quantifier “at least 100 ∗ k%”[1]

        Parameters
        ----------
        k: float
           Should be a value in `[0, 1]`

        References
        ---------
        [1] THEERENS, Adnan et CORNELIS, Chris.
        Fuzzy rough sets based on fuzzy quantification.
        Fuzzy Sets and Systems, 2023, vol. 473, p. 108704.
    """
    k: float

    def __call__(self, p: float):
        k = self.k
        if p >= k:
            return 1
        else:
            return 0
