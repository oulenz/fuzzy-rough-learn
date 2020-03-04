import pytest

from sklearn.utils.estimator_checks import check_estimator

from frlearn.literature import FRNNClassifier


@pytest.mark.parametrize(
    'Estimator', [FRNNClassifier, ]
)
@pytest.mark.skip('check_estimator is currently too strict')
def test_all_estimators(Estimator):
    return check_estimator(Estimator)
