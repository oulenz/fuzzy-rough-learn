import pytest

import numpy as np
from sklearn.datasets import load_iris

from frlearn.neighbours import FRNN, FRONEC, FROVOCO


@pytest.fixture
def data():
    return load_iris(return_X_y=True)


def test_frnn(data):
    X, y = data
    clf = FRNN()
    model = clf.construct(X, y)

    y_pred = model.query(X)
    assert y_pred.shape == (X.shape[0], 3)


def test_frovoco(data):
    X, y = data
    X = np.concatenate([X[:5], X[50:67], X[100:]], axis=0)
    y = np.concatenate([y[:5], y[50:67], y[100:]], axis=0)

    clf = FROVOCO()
    model = clf.construct(X, y)

    y_pred = model.query(X)
    assert y_pred.shape == (X.shape[0], 3)


def test_fronec(data):
    X, y = data
    Y = (y[:, None] == np.arange(3)).astype(int)
    Y[[109, 117, 131], 2] = 0
    Y[((X[:, 0] >= 6) & (y == 1)), 2] = 1
    Y[((X[:, 0] <= 6) & (y == 2)), 1] = 1

    clf = FRONEC()
    model = clf.construct(X, Y)

    y_pred = model.query(X)
    assert y_pred.shape == (X.shape[0], 3)
