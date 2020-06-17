import pytest

from sklearn.datasets import load_iris

from frlearn.neighbours import FRPS


@pytest.fixture
def data():
    return load_iris(return_X_y=True)


def test_frps(data):
    X_orig, y_orig = data
    preprocessor = FRPS()
    X, y = preprocessor.process(X_orig, y_orig)
    assert y.shape[0] == X.shape[0]
    assert X.shape[1] == X_orig.shape[1]
    assert X.shape[0] < X_orig.shape[0]
