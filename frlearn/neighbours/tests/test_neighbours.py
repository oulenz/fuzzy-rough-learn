import pytest

from sklearn.datasets import load_iris

from frlearn.neighbours import FRNNClassifier, ImbalancedFRNNClassifier


@pytest.fixture
def data():
    return load_iris(return_X_y=True)


def test_frnn_classifier(data):
    X, y = data
    clf = FRNNClassifier()
    clf.fit(X, y)
    assert hasattr(clf, 'classes_')
    assert hasattr(clf, 'n_classes_')
    assert hasattr(clf, 'n_attributes_')
    assert hasattr(clf, 'scale_')
    assert hasattr(clf, 'class_counts_')
    assert hasattr(clf, 'nn_indices_')

    y_pred = clf.predict(X)
    assert y_pred.shape == (X.shape[0],)


def test_imbalanced_frnn_classifier(data):
    X, y = data
    # reduce to binary classification
    y[y != 1] = 0
    clf = ImbalancedFRNNClassifier()
    clf.fit(X, y)
    assert hasattr(clf, 'classes_')
    assert hasattr(clf, 'n_classes_')
    assert hasattr(clf, 'n_attributes_')
    assert hasattr(clf, 'scale_')
    assert hasattr(clf, 'class_counts_')
    assert hasattr(clf, 'nn_indices_')

    y_pred = clf.predict(X)
    assert y_pred.shape == (X.shape[0],)
