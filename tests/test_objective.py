import numpy as np
import pytest

from mqboost.base import ModelName, ObjectiveName, ValidationException
from mqboost.objective import (
    MQObjective,
    _eval_check_loss,
    _lgb_eval_loss,
    _xgb_eval_loss,
    approx_loss_grad_hess,
    check_loss_grad_hess,
    huber_loss_grad_hess,
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
        model=ModelName.xgboost,
        delta=0.0,
        epsilon=0.0,
    )
    assert mq_objective.fobj is not None
    assert mq_objective.feval is not None
    assert callable(mq_objective.fobj)
    assert callable(mq_objective.feval)


def test_mqobjective_huber_loss_initialization():
    """Test MQObjective initialization with huber loss."""
    delta = 0.05
    mq_objective = MQObjective(
        alphas=alphas,
        objective=ObjectiveName.huber,
        model=ModelName.lightgbm,
        delta=delta,
        epsilon=0.0,
    )
    assert mq_objective.fobj is not None
    assert callable(mq_objective.fobj)


def test_mqobjective_approx_loss_initialization():
    """Test MQObjective initialization with approx loss."""
    epsilon = 0.1
    mq_objective = MQObjective(
        alphas=alphas,
        objective=ObjectiveName.approx,
        model=ModelName.xgboost,
        delta=0.0,
        epsilon=epsilon,
    )
    assert mq_objective.fobj is not None
    assert callable(mq_objective.fobj)


# Test loss functions: check, huber, approx
def test_check_loss_grad_hess(dummy_data):
    """Test check loss gradient and Hessian calculation."""
    dtrain = dummy_data(y_true)
    grads, hess = check_loss_grad_hess(y_pred=y_pred, dtrain=dtrain, alphas=alphas)
    # fmt: off
    expected_grads = [-0.1, -0.1, 0.9, -0.1, -0.1, -0.5, -0.5, 0.5, -0.5, -0.5, -0.9, -0.9, 0.1, -0.9, -0.9]
    # fmt: on
    np.testing.assert_almost_equal(grads, np.array(expected_grads))
    assert grads.shape == hess.shape
    assert len(grads) == len(y_pred)


# fmt: off
@pytest.mark.parametrize(
    "delta, expected_grads",
    [
        (0.01, [-0.1, -0.1, 0.9, -0.1, -0.1, -0.5, -0.5, 0.5, -0.5, -0.5, -0.9, -0.9, 0.1, -0.9, -0.9]),
        (0.02, [0.001, 0.001, 0.9, -0.1, -0.1, 0.005, 0.005, 0.5, -0.5, -0.5, 0.009, 0.009, 0.1, -0.9, -0.9]),
        (0.05, [0.001, 0.001, 0.018, 0.003, 0.005, 0.005, 0.005, 0.01, 0.015, 0.025, 0.009, 0.009, 0.002, 0.027, 0.045]),
    ],
)
# fmt: on
def test_huber_loss_grad_hess(dummy_data, delta, expected_grads):
    """Test huber loss gradient and Hessian calculation with multiple datasets and deltas."""
    dtrain = dummy_data(y_true)
    grads, hess = huber_loss_grad_hess(
        y_pred=y_pred, dtrain=dtrain, alphas=alphas, delta=delta
    )

    np.testing.assert_almost_equal(grads, np.array(expected_grads))
    assert grads.shape == hess.shape
    assert len(grads) == len(y_pred)


# fmt: off
@pytest.mark.parametrize(
    "epsilon, expected_grads, expected_hess",
    [
        (
            0.01,
            [0.15, 0.15, 0.7333, 0.025, -0.0167, -0.25, -0.25, 0.3333, -0.375, -0.4167, -0.65, -0.65, -0.0667, -0.775, -0.8167],
            [25.0, 25.0, 16.6666, 12.5, 8.3333, 25.0, 25.0, 16.6666, 12.5, 8.33333333, 25.0, 25.0, 16.6666, 12.5, 8.33333333],
        ),
        (
            0.005,
            [0.0667, 0.0667, 0.8, -0.02857, -0.0545, -0.3333, -0.3333, 0.4, -0.4286, -0.4545, -0.7333, -0.73333, 0.0, -0.8286, -0.8545],
            [33.3333, 33.3333, 20.0, 14.2857, 9.0909, 33.3333, 33.3333, 20.0, 14.2857, 9.0909, 33.3333, 33.3333, 20.0, 14.2857, 9.0909],
        ),
        (
            0.001,
            [-0.0545, -0.0545, 0.8762, -0.0838, -0.0902, -0.4545, -0.4545, 0.4762, -0.4839, -0.4902, -0.8545, -0.8545, 0.0762, -0.8839, -0.8902],
            [45.4545, 45.4545, 23.8095, 16.1290, 9.8039, 45.4545, 45.4545, 23.8095, 16.1290, 9.8039, 45.4545, 45.4545, 23.8095, 16.1290, 9.8039],
        ),
    ],
)
# fmt: on
def test_approx_loss_grad_hess(dummy_data, epsilon, expected_grads, expected_hess):
    """Test approx loss gradient and Hessian calculation."""
    dtrain = dummy_data(y_true)
    grads, hess = approx_loss_grad_hess(
        y_pred=y_pred, dtrain=dtrain, alphas=alphas, epsilon=epsilon
    )
    np.testing.assert_almost_equal(grads, np.array(expected_grads), decimal=4)
    np.testing.assert_almost_equal(hess, np.array(expected_hess), decimal=4)
    assert grads.shape == hess.shape
    assert len(grads) == len(y_pred)


# Test evaluation functions
def test_eval_check_loss(dummy_data):
    """Test evaluation of the check loss."""
    dtrain = dummy_data(y_true)
    loss = _eval_check_loss(y_pred=y_pred, dtrain=dtrain, alphas=alphas)
    np.testing.assert_almost_equal(loss, 0.012)
    assert isinstance(loss, float)
    assert loss > 0


def test_xgb_eval_loss(dummy_data):
    """Test XGBoost evaluation function."""
    dtrain = dummy_data(y_true)
    metric_name, loss = _xgb_eval_loss(y_pred=y_pred, dtrain=dtrain, alphas=alphas)
    assert metric_name == "check_loss"
    assert isinstance(loss, float)


def test_lgb_eval_loss(dummy_data):
    """Test LightGBM evaluation function."""
    dtrain = dummy_data(y_true)
    metric_name, loss, higher_better = _lgb_eval_loss(
        y_pred=y_pred, dtrain=dtrain, alphas=alphas
    )
    assert metric_name == "check_loss"
    assert isinstance(loss, float)
    assert higher_better is False


# Test error handling for invalid parameters
def test_invalid_delta_for_huber():
    """Test that invalid delta for Huber loss raises an exception."""
    alphas = [0.1, 0.5, 0.9]
    with pytest.raises(ValidationException):
        MQObjective(
            alphas=alphas,
            objective=ObjectiveName.huber,
            model=ModelName.xgboost,
            delta=-0.1,  # Invalid delta (negative)
            epsilon=0.0,
        )


def test_invalid_epsilon_for_approx():
    """Test that invalid epsilon for approx loss raises an exception."""
    alphas = [0.1, 0.5, 0.9]
    with pytest.raises(ValidationException):
        MQObjective(
            alphas=alphas,
            objective=ObjectiveName.approx,
            model=ModelName.xgboost,
            delta=0.0,
            epsilon=-0.01,  # Invalid epsilon (negative)
        )
