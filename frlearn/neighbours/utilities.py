from typing import Callable


def resolve_k(k: float or Callable[[int], float] or None, n: int, k_max: int = None):
    """
    Helper method to obtain a valid number of neighbours
    from a parameter `k` given `n` target records,
    where `k` may be defined in terms of `n`.

    Parameters
    ----------
    k: float or (int -> float) or None
        Parameter value to resolve. Can be a float,
        a callable that takes `n` and returns a float,
        or None.

    n: int
        The input for `k` if `k` is callable.

    k_max: int = None
        The maximum allowed value for `k`.
        If None, this is equal to `n`.

    Returns
    -------
    k: int
       If `k` is a float in [1, k_max]: `k`;
       If `k` is None: `k_max`;
       If `k` is callable, the output of `k` applied to `n`,
       rounded to the nearest integer in `[1, k_max]`.

    Raises
    ------
    ValueError
        If `k` is a float not in [1, k_max].

    """
    if k_max is None:
        k_max = n
    if callable(k):
        k = k(n)
    elif k is None:
        k = k_max
    elif not 1 <= k <= k_max:
        raise ValueError(f'{k} is too many nearest neighbours, number has to be between 1 and {k_max}.')
    return min(max(1, round(k)), k_max)
