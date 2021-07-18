from frlearn.neighbours.utilities import resolve_k
from frlearn.parametrisations import log_multiple, multiple


def test_resolve_k():
    assert resolve_k(k=2, n=8) == 2
    assert resolve_k(k=multiple(1), n=8) == 8
    assert resolve_k(k=multiple(.7), n=8) == 6
    assert resolve_k(k=multiple(.01), n=8) == 1
    assert resolve_k(k=log_multiple(2), n=8) == 4
    assert resolve_k(k=log_multiple(20), n=8) == 8
    assert resolve_k(k=None, n=8) == 8
