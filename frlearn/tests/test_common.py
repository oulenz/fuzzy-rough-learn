import importlib
import pytest

import numpy as np
from sklearn.datasets import load_diabetes, load_iris

from frlearn.base import ClassSupervised, FeatureSelector, FeaturePreprocessor, MultiClassClassifier, MultiLabelClassifier, Unsupervised


algorithm_types = [
    'classifiers',
    'data_descriptors',
    'feature_preprocessors',
    'instance_preprocessors',
    'regressors'
]
algorithms = {}
for algorithm_type in algorithm_types:
    module = importlib.import_module('frlearn.' + algorithm_type)
    algorithms[algorithm_type] = [getattr(module, a) for a in module.__all__]


@pytest.fixture
def multiclass_data():
    return load_iris(return_X_y=True)


@pytest.fixture
def regression_data():
    return load_diabetes(return_X_y=True)


@pytest.mark.parametrize(
    'cls',
    [a for a in algorithms['classifiers'] if issubclass(a, MultiClassClassifier)],
)
def test_multiclass_classifier(multiclass_data, cls):
    X, y = multiclass_data
    clf = cls()
    model = clf(X, y)

    scores = model(X)
    assert scores.shape == (X.shape[0], 3)


@pytest.mark.parametrize(
    'cls',
    [a for a in algorithms['classifiers'] if issubclass(a, MultiLabelClassifier)],
)
def test_multilabel_classifier(multiclass_data, cls):
    X, y = multiclass_data
    Y = (y[:, None] == np.arange(3)).astype(int)
    Y[[109, 117, 131], 2] = 0
    Y[((X[:, 0] >= 6) & (y == 1)), 2] = 1
    Y[((X[:, 0] <= 6) & (y == 2)), 1] = 1

    clf = cls()
    model = clf(X, Y)

    scores = model(X)
    assert scores.shape == (X.shape[0], 3)


@pytest.mark.parametrize(
    'cls',
    algorithms['data_descriptors'],
)
def test_data_descriptor(multiclass_data, cls):
    X, y = multiclass_data
    descriptor = cls()
    model = descriptor(X[y == 0])

    scores = model(X)
    assert scores.shape == (X.shape[0], )


@pytest.mark.parametrize(
    'cls',
    algorithms['regressors'],
)
def test_regressor(regression_data, cls):
    X, y = regression_data
    regressor = cls()
    model = regressor(X, y)

    predictions = model(X)
    assert predictions.shape == (X.shape[0], )


@pytest.mark.parametrize(
    'cls',
    [a for a in algorithms['feature_preprocessors'] if issubclass(a, ClassSupervised) and issubclass(a, FeatureSelector)],
)
def test_supervised_feature_selector(multiclass_data, cls):
    X_orig, y_orig = multiclass_data

    preprocessor = cls()
    model = preprocessor(X_orig, y_orig)
    X = model(X_orig)
    assert X.shape[0] == X_orig.shape[0]
    assert X.shape[1] <= X_orig.shape[1]


@pytest.mark.parametrize(
    'cls',
    [a for a in algorithms['feature_preprocessors'] if issubclass(a, Unsupervised) and issubclass(a, FeaturePreprocessor)],
)
def test_unsupervised_feature_preprocessor(multiclass_data, cls):
    X_orig, y_orig = multiclass_data

    preprocessor = cls()
    model = preprocessor(X_orig)
    X = model(X_orig)
    assert X.shape[0] == X_orig.shape[0]


# TODO: add test for SAE (but SAE takes a long time to run)


@pytest.mark.parametrize(
    'cls',
    algorithms['instance_preprocessors'],
)
def test_supervised_instance_selector(multiclass_data, cls):
    X_orig, y_orig = multiclass_data
    preprocessor = cls()

    X, y = preprocessor(X_orig, y_orig)
    assert y.shape[0] == X.shape[0]
    assert X.shape[1] == X_orig.shape[1]
    assert X.shape[0] <= X_orig.shape[0]
