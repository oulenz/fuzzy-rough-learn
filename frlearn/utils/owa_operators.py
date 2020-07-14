"""Owa operators"""
from __future__ import annotations

import numpy as np

from .np_utils import first, greatest, last, least


class OWAOperator():

    """
    Ordered Weighted Averaging (OWA) operator, which can be applied to an
    array to obtain its ordered weighted average. Intended specifically for
    dual pairs of OWA operators that approximate maxima and minima, which are
    encoded together in one object.

    Parameters
    ----------
    f : int -> array shape=(k, )
        Generating function which takes an integer k and returns a valid
        weight vector of length k. (The values should be in [0, 1] and sum to 1.)

    scale : boolean, default=False
        If True, weights will be scaled to sum to 1 before being applied.

    name : str, default=None
        Name of the weights to be displayed as its string representation.
        If None, `f` will be used to generate a weight array of length 4.
    """

    def __init__(self, f, *, scale: bool = False, name: str = None):
        self.f = f
        self.scale = scale
        self.name = name

    def __eq__(self, other):
        if isinstance(other, OWAOperator):
            return np.array_equal(self.f(16), other.f(16))
        return NotImplemented

    def __str__(self):
        return self.name or str(self.f(4))

    def _apply(self, a, axis, flavour: str):
        w = self.f(a.shape[axis])
        w = np.reshape(w, [-1] + ((len(a.shape) - axis - 1) % len(a.shape)) * [1])
        if flavour == 'arithmetic':
            return np.sum(w * a, axis=axis)
        if flavour == 'geometric':
            return np.exp(np.sum(w * np.log(a), axis=axis))
        if flavour == 'harmonic':
            return 1 / np.sum(w / a, axis=axis)

    def soft_max(self, a, k=None, axis=-1, flavour: str = 'arithmetic'):
        """
        Calculates the soft maximum of an array.

        Parameters
        ----------
        a : ndarray
            Input array of values.

        k : int or float or None, default=None
            Number of greatest values from which the soft maximum is calculated.
            If a float in (0, 1), taken as a fraction of the total number of values.
            If None, all values are used.

        axis : int, default=-1
            The axis along which the soft maximum is calculated.

        flavour : str {'arithmetic', 'geometric', 'harmonic', }, default='arithmetic'
            Determines the type of weighted average.

        Returns
        -------
        soft_max_along_axis : ndarray
            An array with the same shape as `a`, with the specified
            axis removed. If `a` is a 0-d array, a scalar is returned.
        """
        a = greatest(a, k, axis=axis)
        return self._apply(a, axis=axis, flavour=flavour)

    def soft_min(self, a, k=None, axis=-1, flavour: str = 'arithmetic'):
        """
        Calculates the soft minimum of an array.

        Parameters
        ----------
        a : ndarray
            Input array of values.

        k : int or float or None, default=None
            Number of least values from which the soft minimum is calculated.
            If a float in (0, 1), taken as a fraction of the total number of values.
            If None, all values are used.

        axis : int, default=-1
            The axis along which the soft minimum is calculated.

        flavour : str {'arithmetic', 'geometric', 'harmonic', }, default='arithmetic'
            Determines the type of weighted average.

        Returns
        -------
        soft_min_along_axis : ndarray
            An array with the same shape as `a`, with the specified
            axis removed. If `a` is a 0-d array, a scalar is returned.
        """
        a = least(a, k, axis=axis)
        return self._apply(a, axis=axis, flavour=flavour)

    def soft_head(self, a, k=None, axis=-1, flavour: str = 'arithmetic'):
        """
        Calculates the soft head of an array.

        Parameters
        ----------
        a : ndarray
            Input array of values.

        k : int or float or None, default=None
            Number of initial values from which the soft head is calculated.
            If a float in (0, 1), taken as a fraction of the total number of values.
            If None, all values are used.

        axis : int, default=-1
            The axis along which the soft head is calculated.

        flavour : str {'arithmetic', 'geometric', 'harmonic', }, default='arithmetic'
            Determines the type of weighted average.

        Returns
        -------
        soft_head_along_axis : ndarray
            An array with the same shape as `a`, with the specified
            axis removed. If `a` is a 0-d array, a scalar is returned.
        """
        a = first(a, k, axis=axis)
        return self._apply(a, axis=axis, flavour=flavour)

    def soft_tail(self, a, k=None, axis=-1, flavour: str = 'arithmetic'):
        """
        Calculates the soft tail of an array.

        Parameters
        ----------
        a : ndarray
            Input array of values.

        k : int or float or None, default=None
            Number of terminal values from which the soft tail is calculated.
            If a float in (0, 1), taken as a fraction of the total number of values.
            If None, all values are used.

        axis : int, default=-1
            The axis along which the soft tail is calculated.

        flavour : str {'arithmetic', 'geometric', 'harmonic', }, default='arithmetic'
            Determines the type of weighted average.

        Returns
        -------
        soft_tail_along_axis : ndarray
            An array with the same shape as `a`, with the specified
            axis removed. If `a` is a 0-d array, a scalar is returned.
        """
        a = last(a, k, axis=axis)
        return self._apply(a, axis=axis, flavour=flavour)


class strict(OWAOperator):
    def __init__(self):
        f = lambda k: np.append(np.ones(1), np.zeros(k - 1))
        super().__init__(f=f, name='strict')

    def _apply(self, a, axis, flavour: str):
        return a.take(indices=0, axis=axis)


class additive(OWAOperator):
    def __init__(self):
        f = lambda k: np.flip(2 * np.arange(1, k + 1) / (k * (k + 1)))
        super().__init__(f=f, name='additive')


class exponential(OWAOperator):
    def __init__(self):
        f = lambda k: np.flip(2 ** np.arange(k) / (2 ** k - 1)) if k < 32 else np.cumprod(np.full(k, 0.5))
        super().__init__(f=f, name='exponential')


class invadd(OWAOperator):
    def __init__(self):
        f = lambda k: 1 / (np.arange(1, k + 1) * np.sum(1 / np.arange(1, k + 1)))
        super().__init__(f=f, name='invadd')


class mean(OWAOperator):
    def __init__(self):
        f = lambda k: np.full(k, 1 / k)
        super().__init__(f=f, name='mean')


class trimmed(OWAOperator):
    def __init__(self):
        f = lambda k: np.append(np.zeros(k - 1), np.ones(1))
        super().__init__(f=f, name='trimmed')

    def _apply(self, a, axis, flavour: str):
        return a.take(indices=-1, axis=axis)

class deltaquadsigmoid(OWAOperator):
    def __init__(self, alpha, beta):
        def Q(a):
            return np.where(a <= alpha, 0,
                            np.where(a >= beta, 1,
                                     np.where(a <= (alpha + beta) / 2, 2 * (a - alpha) ** 2 / (beta - alpha) ** 2,
                                              1 - 2 * (a - beta) ** 2 / (beta - alpha) ** 2)))
            if a <= alpha:
                return 0
            if a >= beta:
                return 1
            if a <= (alpha + beta) / 2:
                return 2 * (a - alpha) ** 2 / (beta - alpha) ** 2
            return 1 - 2 * (a - beta) ** 2 / (beta - alpha) ** 2
        f = lambda k: Q(np.arange(1, k+1) / k) - Q(np.arange(k) / k)
        super().__init__(f=f, name='vague_quantifier_{}_{}'.format(alpha, beta))
