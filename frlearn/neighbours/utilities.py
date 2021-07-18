from typing import Callable


def resolve_k(k: float or Callable[[int], float] or None, n: int):
    """
    Helper function to obtain a valid number of neighbours
    from a parameter `k` given a maximum `n`,
    where `k` may be defined in terms of `n`.
    Parameters
    ----------
    k: float or (int -> float) or None
        Parameter value to resolve. Can be a float,
        a callable that takes `n` and returns a float,
        or None.
    n: int
        The maximum allowed number of neighbours,
        and the input for `k` if `k` is callable.

    Returns
    -------
    k: int
       If `k` is a float, `k`, if `k` is None, `n`,
       If `k` is callable, the output of `k` applied to `n`,
       rounded to the nearest integer in `[1, n]`.

    """
    if callable(k):
        k = k(n)
    if k is None:
        k = n
    return min(max(1, round(k)), n)
