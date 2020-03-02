from functools import partial

import numpy as np

from ..utils import greatest, least


class OWAOperator():

    """
    Ordered Weighted Averaging (OWA) operator, which can be applied to an
    array to obtain its ordered weighted average. Intended specifically for
    dual pairs of OWA operators that approximate maxima and minima, which are
    encoded together in one object.

    Parameters
    ----------
    w : array shape=(n_weights, )
        Weights which define the OWA operator. The values should be in [0, 1]
        and sum to 1.

    name : str, default=None
        Name of the OWA operator to be displayed as its string representation.
        If None, the weight array will be displayed instead.
    """

    def __init__(self, w, *, name: str = None):
        self.w = w
        self.k = len(w)
        self.name = name

    @classmethod
    def from_function(cls, f, k: int, *, scale: bool = False, name: str = None):
        """
        Constructor to create an OWA operator from a function f and its length
        k.

        Parameters
        ----------
        f : int -> array shape=(k, )
            Generating function which takes an integer k and returns a valid
            weight vector of length k.

        k : int
            The length of the intended weight vector.

        name : str, default=None
            Name of the OWA operator to be displayed as its string
            representation. If None, the weight array will be displayed
            instead.

        Returns
        -------
        operator : OWAOperator
            OWA operator initialised with the weight vector obtained by
            applying f to k.
        """
        w = f(k)
        if scale:
            w /= np.sum(w)
        name = '{name}({k})'.format(name=name, k=k)
        return cls(w, name=name)

    @classmethod
    def family(cls, f, *, scale: bool = False, name: str = None):
        """
        Convenience method for defining OWA operators up to the length k of the
        weight vector.

        Parameters
        ----------
        f : int -> array shape=(k, )
            Generating function which takes an integer k and returns a valid
            weight vector of length k.

        name : str, default=None
            Name of the OWA operator to be displayed as its string
            representation. If None, the weight array will be displayed
            instead.

        Returns
        -------
        constructor : k -> OWAOperator
            Constructor that takes an integer k and creates the OWA operator
            initialised with the weight vector obtained by applying f to k.
        """
        return partial(cls.from_function, f, scale=scale, name=name)

    def __eq__(self, other):
        if isinstance(other, OWAOperator):
            return np.array_equal(self.w, other.w)
        return NotImplemented

    def __len__(self):
        return self.k

    def __str__(self):
        if self.name:
            return self.name.format(self.k)
        return str(self.w)

    def _apply(self, a, axis, flavour: str):
        w = np.reshape(self.w, [-1] + ((len(a.shape) - axis - 1) % len(a.shape)) * [1])
        if flavour == 'arithmetic':
            return np.sum(w * a, axis=axis)
        if flavour == 'geometric':
            return np.exp(np.sum(w * np.log(a), axis=axis))
        if flavour == 'harmonic':
            return 1 / np.sum(w / a, axis=axis)

    def soft_max(self, a, axis=-1, flavour: str = 'arithmetic'):
        """
        Calculates the soft maximum of an array.

        Parameters
        ----------
        a : ndarray
            Input array of values.

        axis : int, default=-1
            The axis along which the soft maximum is calculated

        flavour : str {'arithmetic', 'geometric', 'harmonic', }, default='arithmetic'
            Determines the type of weighted average.

        Returns
        -------
        soft_max_along_axis : ndarray
            An array with the same shape as `a`, with the specified
            axis removed. If `a` is a 0-d array, a scalar is returned.
        """
        a = greatest(a, self.k, axis=axis)
        return self._apply(a, axis=axis, flavour=flavour)

    def soft_min(self, a, axis=-1, flavour: str = 'arithmetic'):
        """
        Calculates the soft minimum of an array.

        Parameters
        ----------
        a : ndarray
            Input array of values.

        axis : int, default=-1
            The axis along which the soft minimum is calculated

        flavour : str {'arithmetic', 'geometric', 'harmonic', }, default='arithmetic'
            Determines the type of weighted average.

        Returns
        -------
        soft_min_along_axis : ndarray
            An array with the same shape as `a`, with the specified
            axis removed. If `a` is a 0-d array, a scalar is returned.
        """
        a = least(a, self.k, axis=axis)
        return self._apply(a, axis=axis, flavour=flavour)

    def soft_head(self, a, axis=-1, flavour: str = 'arithmetic'):
        """
        Calculates the soft head of an array.

        Parameters
        ----------
        a : ndarray
            Input array of values.

        axis : int, default=-1
            The axis along which the soft head is calculated

        flavour : str {'arithmetic', 'geometric', 'harmonic', }, default='arithmetic'
            Determines the type of weighted average.

        Returns
        -------
        soft_head_along_axis : ndarray
            An array with the same shape as `a`, with the specified
            axis removed. If `a` is a 0-d array, a scalar is returned.
        """
        slc = [slice(None)] * len(a.shape)
        slc[axis] = slice(0, self.k)
        a = a[tuple(slc)]
        return self._apply(a, axis=axis, flavour=flavour)

    def soft_tail(self, a, axis=-1, flavour: str = 'arithmetic'):
        """
        Calculates the soft tail of an array.

        Parameters
        ----------
        a : ndarray
            Input array of values.

        axis : int, default=-1
            The axis along which the soft tail is calculated

        flavour : str {'arithmetic', 'geometric', 'harmonic', }, default='arithmetic'
            Determines the type of weighted average.

        Returns
        -------
        soft_tail_along_axis : ndarray
            An array with the same shape as `a`, with the specified
            axis removed. If `a` is a 0-d array, a scalar is returned.
        """
        slc = [slice(None)] * len(a.shape)
        slc[axis] = slice(-1, -self.k -1, -1)
        a = a[tuple(slc)]
        return self._apply(a, axis=axis, flavour=flavour)


strict = OWAOperator(np.ones(1), name='strict')

additive = OWAOperator.family(
    lambda k: np.flip(2 * np.arange(1, k + 1) / (k * (k + 1))),
    name='additive')

exponential = OWAOperator.family(
    lambda k: np.flip(2 ** np.arange(k) / (2 ** k - 1)) if k < 32 else np.cumprod(np.full(k, 0.5)),
    name='exponential')

invadd = OWAOperator.family(
    lambda k: 1 / (np.arange(1, k + 1) * np.sum(1 / np.arange(1, k + 1))),
    name='invadd')

mean = OWAOperator.family(
    lambda k: np.full(k, 1 / k),
    name='mean')

trimmed = OWAOperator.family(
    lambda k: np.append(np.zeros(k - 1), np.ones(1)),
    name='trimmed')
