import numpy as np
import pytest

from mqboost.base import ModelName, ObjectiveName, ValidationException
from mqboost.objective import (
    MQObjective,
    eval_check_loss,
    validate_epsilon,
)


@pytest.fixture
def dummy_data():
    class DummyDtrain:
        def __init__(self, label):
            self.label = label

        def get_label(self):
            return self.label

    return DummyDtrain


alphas = [0.1, 0.5, 0.9]
y_pred = np.array([1.0, 1.5, 2.0, 2.5, 3.0] * 3)
y_true = np.array([1.01, 1.51, 1.98, 2.53, 3.05] * 3)


# Test MQObjective Initialization
def test_mqobjective_check_loss_initialization():
    """Test MQObjective initialization with check loss."""

    mq_objective = MQObjective(
        alphas=alphas,
        objective=ObjectiveName.check,
        weight=None,
        model=ModelName.xgboost,
        epsilon=0.0,
    )
    assert mq_objective.fobj is not None
    assert mq_objective.feval is not None
    assert callable(mq_objective.fobj)
    assert callable(mq_objective.feval)


def test_mqobjective_huber_loss_initialization():
    """Test MQObjective initialization with huber loss."""
    epsilon = 0.05
    mq_objective = MQObjective(
        alphas=alphas,
        objective=ObjectiveName.huber,
        weight=None,
        model=ModelName.lightgbm,
        epsilon=epsilon,
    )
    assert mq_objective.fobj is not None
    assert callable(mq_objective.fobj)


def test_mqobjective_approx_loss_initialization():
    """Test MQObjective initialization with approx loss."""
    epsilon = 0.1
    mq_objective = MQObjective(
        alphas=alphas,
        objective=ObjectiveName.approx,
        weight=None,
        model=ModelName.xgboost,
        epsilon=epsilon,
    )
    assert mq_objective.fobj is not None
    assert callable(mq_objective.fobj)


# Test loss functions: check, huber, approx
def test_check_loss_grad_hess(dummy_data):
    """Test check loss gradient and Hessian calculation."""
    dtrain = dummy_data(y_true)
    obj = MQObjective(alphas, ObjectiveName.check, ModelName.lightgbm, 1e-5)
    grads, hess = obj.fobj(y_pred, dtrain)
    # fmt: off
    expected_grads = [-0.02, -0.02, 0.18, -0.02, -0.02, -0.1, -0.1, 0.1, -0.1, -0.1, -0.18, -0.18, 0.02, -0.18, -0.18]
    # fmt: on
    np.testing.assert_almost_equal(grads, np.array(expected_grads))
    assert grads.shape == hess.shape
    assert len(grads) == len(y_pred)


# fmt: off
@pytest.mark.parametrize(
    "epsilon, expected_grads",
    [
        (0.01, [-0.02, -0.02, 0.18, -0.02, -0.02, -0.1, -0.1, 0.1, -0.1, -0.1, -0.18, -0.18, 0.02, -0.18, -0.18]),
        (0.02, [-0.01, -0.01, 0.18, -0.02, -0.02, -0.05, -0.05, 0.1, -0.1, -0.1, -0.09, -0.09, 0.02, -0.18, -0.18]),
        (0.05, [-0.004, -0.004, 0.072, -0.012, -0.02, -0.02, -0.02, 0.04, -0.06, -0.1, -0.036, -0.036, 0.008, -0.108, -0.18]),
    ],
)
# fmt: on
def test_huber_loss_grad_hess(dummy_data, epsilon, expected_grads):
    """Test huber loss gradient and Hessian calculation with multiple datasets and epsilon values."""
    dtrain = dummy_data(y_true)
    obj = MQObjective(alphas, ObjectiveName.huber, ModelName.lightgbm, epsilon)
    grads, hess = obj.fobj(y_pred, dtrain)
    np.testing.assert_almost_equal(grads, np.array(expected_grads))
    assert grads.shape == hess.shape
    assert len(grads) == len(y_pred)


# fmt: off
@pytest.mark.parametrize(
    "epsilon, expected_grads, expected_hess",
    [
        (
            0.01,
            [0.03, 0.03, 0.1467, 0.005, -0.0033, -0.05, -0.05, 0.0667, -0.075, -0.0833, -0.13, -0.13, -0.0133, -0.155, -0.1633],
            [5.0, 5.0, 3.3333, 2.5, 1.6667, 5.0, 5.0, 3.3333, 2.5, 1.66666667, 5.0, 5.0, 3.3333, 2.5, 1.66666667],
        ),
        (
            0.005,
            [0.0133, 0.0133, 0.16, -0.00571, -0.0109, -0.0667, -0.0667, 0.08, -0.0857, -0.0909, -0.1467, -0.14667, 0.0, -0.1657, -0.1709],
            [6.6667, 6.6667, 4.0, 2.8571, 1.8182, 6.6667, 6.6667, 4.0, 2.8571, 1.8182, 6.6667, 6.6667, 4.0, 2.8571, 1.8182],
        ),
        (
            0.001,
            [-0.0109, -0.0109, 0.1752, -0.01676, -0.01804, -0.0909, -0.0909, 0.0952, -0.09678, -0.09804, -0.1709, -0.1709, 0.01524, -0.17678, -0.17804],
            [9.0909, 9.0909, 4.7619, 3.2258, 1.9608, 9.0909, 9.0909, 4.7619, 3.2258, 1.9608, 9.0909, 9.0909, 4.7619, 3.2258, 1.9608],
        ),
    ],
)
# fmt: on
def test_approx_loss_grad_hess(dummy_data, epsilon, expected_grads, expected_hess):
    """Test approx loss gradient and Hessian calculation."""
    dtrain = dummy_data(y_true)
    obj = MQObjective(alphas, ObjectiveName.approx, ModelName.lightgbm, epsilon)
    grads, hess = obj.fobj(y_pred, dtrain)
    np.testing.assert_almost_equal(grads, np.array(expected_grads), decimal=4)
    np.testing.assert_almost_equal(hess, np.array(expected_hess), decimal=4)
    assert grads.shape == hess.shape
    assert len(grads) == len(y_pred)


# Test evaluation functions
def test_eval_check_loss(dummy_data):
    """Test evaluation of the check loss."""
    dtrain = dummy_data(y_true)
    loss = eval_check_loss(y_pred=y_pred, dtrain=dtrain, alphas=alphas)
    np.testing.assert_almost_equal(loss, 0.036)
    assert isinstance(loss, float)
    assert loss > 0


def test_xgb_eval_loss(dummy_data):
    """Test XGBoost evaluation function."""
    dtrain = dummy_data(y_true)
    obj = MQObjective(alphas, ObjectiveName.check, ModelName.xgboost, 1e-5, None)
    metric_name, loss = obj.xgb_feval(y_pred, dtrain)
    assert metric_name == "check_loss"
    assert isinstance(loss, float)


def test_lgb_eval_loss(dummy_data):
    """Test LightGBM evaluation function."""
    dtrain = dummy_data(y_true)
    obj = MQObjective(alphas, ObjectiveName.check, ModelName.lightgbm, 1e-5, None)
    metric_name, loss, higher_better = obj.lgb_feval(y_pred, dtrain)
    assert metric_name == "check_loss"
    assert isinstance(loss, float)
    assert higher_better is False


# Test error handling for invalid parameters
def test_invalid_epsilon_for_huber():
    """Test that invalid epsilon for Huber loss raises an exception."""
    alphas = [0.1, 0.5, 0.9]
    with pytest.raises(ValidationException):
        MQObjective(
            alphas=alphas,
            objective=ObjectiveName.huber,
            weight=None,
            model=ModelName.xgboost,
            epsilon=-0.1,  # Invalid epsilon (negative)
        )


def test_invalid_epsilon_for_approx():
    """Test that invalid epsilon for approx loss raises an exception."""
    alphas = [0.1, 0.5, 0.9]
    with pytest.raises(ValidationException):
        MQObjective(
            alphas=alphas,
            objective=ObjectiveName.approx,
            weight=None,
            model=ModelName.xgboost,
            epsilon=-0.01,  # Invalid epsilon (negative)
        )

# Test for validate_epsilon
def test_validate_epsilon_valid_epsilon():
    epsilon = 0.04
    assert validate_epsilon(epsilon) is None

    epsilon = 0.01
    assert validate_epsilon(epsilon) is None

def test_validate_epsilon_invalid_type():
    with pytest.raises(ValidationException, match="Epsilon is not float type"):
        validate_epsilon(1)


def test_validate_epsilon_negative_epsilon():
    with pytest.raises(ValidationException, match="Epsilon must be positive"):
        validate_epsilon(-0.01)
