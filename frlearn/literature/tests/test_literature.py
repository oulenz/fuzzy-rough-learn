import pytest

from sklearn.datasets import load_iris

from frlearn.literature import FRNNClassifier


@pytest.fixture
def data():
    return load_iris(return_X_y=True)


def test_frnn_classifier(data):
    X, y = data
    clf = FRNNClassifier()
    clf.fit(X, y)

    y_pred = clf.predict(X)
    assert y_pred.shape == (X.shape[0],)

