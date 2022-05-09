import pytest

from sklearn.datasets import load_iris

from frlearn.neighbours.data_descriptors import ALP, LNND, LOF, NND
from frlearn.neighbours.utilities import resolve_k
from frlearn.parametrisations import log_multiple, multiple

@pytest.fixture
def multiclass_data():
    return load_iris(return_X_y=True)

def test_resolve_k():
    assert resolve_k(k=2, n=8) == 2
    assert resolve_k(k=multiple(1), n=8) == 8
    assert resolve_k(k=multiple(.7), n=8) == 6
    assert resolve_k(k=multiple(.01), n=8) == 1
    assert resolve_k(k=log_multiple(2), n=8) == 4
    assert resolve_k(k=log_multiple(20), n=8) == 8
    assert resolve_k(k=None, n=8) == 8

@pytest.mark.parametrize(
    'cls',
    [ALP, LNND, LOF, ],
)
def test_localised_k(multiclass_data, cls):
    X, y = multiclass_data
    descriptor = cls(k=lambda x: x)
    model = descriptor(X[y == 0])
    assert model.k == 49
    scores = model(X)

    with pytest.raises(ValueError):
        descriptor = cls(k=50)
        model = descriptor(X[y == 0])


@pytest.mark.parametrize(
    'cls',
    [NND, ],
)
def test_unlocalised_k(multiclass_data, cls):
    X, y = multiclass_data
    descriptor = cls(k=lambda x: x)
    model = descriptor(X[y == 0])
    assert model.k == 50
    scores = model(X)

    descriptor = cls(k=50)
    model = descriptor(X[y == 0])
    assert model.k == 50
    scores = model(X)


@pytest.mark.parametrize(
    'cls',
    [ALP, ],
)
def test_unlocalised_l(multiclass_data, cls):
    X, y = multiclass_data
    descriptor = cls(l=lambda x: x)
    model = descriptor(X[y == 0])
    assert model.l == 50
    scores = model(X)

    descriptor = cls(l=50)
    model = descriptor(X[y == 0])
    assert model.l == 50
    scores = model(X)
