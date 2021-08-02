from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MinkowskiSize:
    """
    Family of vector size measures of the form
    `(x1**p + x2**p + ... + xm**p)**(1/p)` (if `unrooted = False`), or
    `(x1**p + x2**p + ... + xm**p)` (if `unrooted = True`),
    for `0 < p < ∞`, and their limits in 0 and ∞.

    For `p = 0`, the rooted variant evaluates to ∞ if there is more than one non-zero coefficient,
    to 0 if all coefficients are zero, and to the only non-zero coefficient otherwise.
    The unrooted variant is equal to the number of non-zero coefficients.

    For `p = ∞`, the rooted variant is the maximum of all coefficients.
    The unrooted variant evaluates to ∞ if there is at least one coefficient larger than 1,
    and to the number of coefficients equal to 1 otherwise.

    Parameters
    ----------
    p: float = 1
        Exponent to use. Must be in `[0, ∞]`.

    unrooted: bool = False
        Whether to omit the root `**(1/p)` from the formula.
        For `p = 0`, this gives Hamming size.
        For `p = 2`, this gives squared Euclidean size.

    scale_by_dimensionality: bool = False
        If `True`, values are scaled linearly such that the vector `[1, 1, ..., 1]` has size 1.
        This can be used to ensure that the range of dissimilarity values in the unit hypercube is `[0, 1]`,
        which can be useful when working with features scaled to `[0, 1]`.

    Notes
    -----
    The most used parameter combinations have their own name.

    * Hamming size is unrooted `p = 0`.
    * The Boscovich norm is `p = 1`. Also known as cityblock, Manhattan or Taxicab norm.
    * The Euclidean norm is rooted `p = 2`. Also known as Pythagorean norm.
    * Squared Euclidean size is unrooted `p = 2`.
    * The Chebishev norm is rooted `p = ∞`. Also known as chessboard or maximum norm.
    """

    p: float
    unrooted: bool = False
    scale_by_dimensionality: bool = False

    def __post_init__(self):
        if self.p < 0:
            raise ValueError('`p` must be in `[0, ∞]`')

    def __call__(self, u, axis=-1):
        if self.p == 0:
            if self.unrooted:
                result = np.count_nonzero(u, axis=axis)
            else:
                result = np.where(np.count_nonzero(u, axis=axis) <= 1, np.sum(np.abs(u), axis=axis), np.inf)
        elif self.p == 1:
            result = np.sum(np.abs(u), axis=axis)
        elif self.p == np.inf:
            if self.unrooted:
                result = np.sum(np.where(np.abs(u) < 1, 0, np.where(np.abs(u) > 1, np.inf, 1)), axis=axis)
            else:
                result = np.max(u, axis=axis)
        else:
            result = np.sum(np.abs(u) ** self.p, axis=axis)
            if not self.unrooted:
                result = result**(1/self.p)
        if self.scale_by_dimensionality and self.p < np.inf:
            if self.unrooted:
                result = result / u.shape[axis]
            else:
                result = result / (u.shape[axis]**(1/self.p))
        return result
