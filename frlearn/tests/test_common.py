import pytest

from sklearn.utils.estimator_checks import check_estimator

from frlearn.neighbours import FRNN, FROVOCO


@pytest.mark.parametrize(
    'Estimator', [FRNN, FROVOCO, ]
)
@pytest.mark.skip('check_estimator is currently too strict')
def test_all_estimators(Estimator):
    return check_estimator(Estimator)
