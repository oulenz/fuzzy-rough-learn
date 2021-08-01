import pytest

import numpy as np

from frlearn.array_functions import first, greatest, last, least


@pytest.fixture
def a():
    return np.array([[1, 3, 2], [6, 5, 4], [9, 7, 8]])


def test_first(a):
    assert np.array_equal(first(a, k=3, axis=-1), np.array([[1, 3, 2], [6, 5, 4], [9, 7, 8]]))
    assert np.array_equal(first(a, k=2, axis=-1), np.array([[1, 3], [6, 5], [9, 7]]))
    assert np.array_equal(first(a, k=2, axis=0), np.array([[1, 3, 2], [6, 5, 4]]))
    assert np.array_equal(first(a, k=1, axis=1), np.array([[1], [6], [9]]))


def test_last(a):
    assert np.array_equal(last(a, k=3, axis=-1), np.array([[2, 3, 1], [4, 5, 6], [8, 7, 9]]))
    assert np.array_equal(last(a, k=2, axis=-1), np.array([[2, 3], [4, 5], [8, 7]]))
    assert np.array_equal(last(a, k=2, axis=0), np.array([[9, 7, 8], [6, 5, 4]]))
    assert np.array_equal(last(a, k=1, axis=1), np.array([[2], [4], [8]]))


def test_least(a):
    assert np.array_equal(least(a, k=3, axis=-1), np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))
    assert np.array_equal(least(a, k=2, axis=-1), np.array([[1, 2], [4, 5], [7, 8]]))
    assert np.array_equal(least(a, k=2, axis=0), np.array([[1, 3, 2], [6, 5, 4]]))
    assert np.array_equal(least(a, k=1, axis=1), np.array([[1], [4], [7]]))


def test_greatest(a):
    assert np.array_equal(greatest(a, k=3, axis=-1), np.array([[3, 2, 1], [6, 5, 4], [9, 8, 7]]))
    assert np.array_equal(greatest(a, k=2, axis=-1), np.array([[3, 2], [6, 5], [9, 8]]))
    assert np.array_equal(greatest(a, k=2, axis=0), np.array([[9, 7, 8], [6, 5, 4]]))
    assert np.array_equal(greatest(a, k=1, axis=1), np.array([[3], [6], [9]]))
