"""Functions for transforming ranges of values."""

import numpy as np

__all__ = [
    'contract', 'shifted_reciprocal', 'truncated_complement',
]


def contract(x, c: float = 1):
    """
    Strictly order-preserving function from `[-∞, ∞]` to `[0, 1]`
    that sends `-∞, -c, 0, c, ∞` to `0, 0.25, 0.5, 0.75, 1`, respectively.

    Parameters
    ----------
    x : float
        Input value. Should be in `[-∞, ∞]`.

    c : float = 1
        The secondary 'central' value that is sent to 0.75 (-c is sent to 0.25). Should be in `(0, ∞)`.

    Returns
    -------
    y : float
        Output value in [0, 1].
    """
    y = x/(2*(abs(x) + c)) + 0.5
    y = np.where(np.isneginf(x), 0, y)
    y = np.where(np.isposinf(x), 1, y)
    return y


def shifted_reciprocal(x, c: float = 1):
    """
    Order-reversing function from [0, ∞) to [0, 1] that sends `x` to `1/(1 + x/c)`.
    Strictly order-reversing, but does not preserve absolute differences.

    Parameters
    ----------
    x : float
        Input value. Should be in `[0, ∞)`.

    c : float = 1
        The 'central' value that is sent to 0.5. Should be in `(0, ∞)`.

    Returns
    -------
    y : float
        Output value in [0, 1].
    """
    return 1/(1 + x/c)


def truncated_complement(x):
    """
    Order-reversing function from [0, ∞) to [0, 1] that sends `x` to `max(0, 1 - x)`.
    Preserves absolute differences for values under 1, but discards all differences for larger values.

    Parameters
    ----------
    x : float
        Input value. Should be in `[0, ∞)`.

    Returns
    -------
    y : float
        Output value in [0, 1].
    """
    return np.maximum(0, 1 - x)
