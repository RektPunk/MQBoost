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


# Test data and helper functions
@pytest.fixture
def dummy_data():
    """Fixture to generate dummy data for training and predictions."""
    y_pred = np.array([1.0, 1.5, 2.0, 2.5, 3.0] * 3)
    y_true = np.array([1.1, 1.6, 1.9, 2.6, 3.1] * 3)
    dtrain = DummyDtrain(y_true)
    alphas = [0.1, 0.5, 0.9]
    return y_pred, dtrain, alphas


class DummyDtrain:
    """Dummy class to simulate DtrainLike behavior."""

    def __init__(self, label):
        self.label = label

    def get_label(self):
        return self.label


# Test MQObjective Initialization
def test_mqobjective_check_loss_initialization():
    """Test MQObjective initialization with check loss."""
    alphas = [0.1, 0.5, 0.9]
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
    alphas = [0.1, 0.5, 0.9]
    delta = 0.1
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
    alphas = [0.1, 0.5, 0.9]
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
    y_pred, dtrain, alphas = dummy_data
    grads, hess = check_loss_grad_hess(y_pred=y_pred, dtrain=dtrain, alphas=alphas)
    np.testing.assert_almost_equal(
        grads,
        np.array(
            [
                -0.1,
                -0.1,
                0.9,
                -0.1,
                -0.1,
                -0.5,
                -0.5,
                0.5,
                -0.5,
                -0.5,
                -0.9,
                -0.9,
                0.1,
                -0.9,
                -0.9,
            ]
        ),
    )
    assert grads.shape == hess.shape
    assert len(grads) == len(y_pred)


def test_huber_loss_grad_hess(dummy_data):
    """Test huber loss gradient and Hessian calculation."""
    y_pred, dtrain, alphas = dummy_data
    delta = 0.1
    grads, hess = huber_loss_grad_hess(
        y_pred=y_pred, dtrain=dtrain, alphas=alphas, delta=delta
    )
    np.testing.assert_almost_equal(
        grads,
        np.array(
            [
                -0.1,
                -0.1,
                0.9,
                -0.1,
                -0.1,
                -0.5,
                -0.5,
                0.5,
                -0.5,
                -0.5,
                -0.9,
                -0.9,
                0.1,
                -0.9,
                -0.9,
            ]
        ),
    )

    delta = 10
    grads, hess = huber_loss_grad_hess(
        y_pred=y_pred, dtrain=dtrain, alphas=alphas, delta=delta
    )
    np.testing.assert_almost_equal(
        grads,
        np.array(
            [
                0.01,
                0.01,
                0.09,
                0.01,
                0.01,
                0.05,
                0.05,
                0.05,
                0.05,
                0.05,
                0.09,
                0.09,
                0.01,
                0.09,
                0.09,
            ]
        ),
    )
    assert grads.shape == hess.shape
    assert len(grads) == len(y_pred)


def test_approx_loss_grad_hess(dummy_data):
    """Test approx loss gradient and Hessian calculation."""
    y_pred, dtrain, alphas = dummy_data
    epsilon = 0.01
    grads, hess = approx_loss_grad_hess(
        y_pred=y_pred, dtrain=dtrain, alphas=alphas, epsilon=epsilon
    )
    np.testing.assert_almost_equal(
        grads,
        np.array(
            [
                -0.05454545,
                -0.05454545,
                0.85454545,
                -0.05454545,
                -0.05454545,
                -0.45454545,
                -0.45454545,
                0.45454545,
                -0.45454545,
                -0.45454545,
                -0.85454545,
                -0.85454545,
                0.05454545,
                -0.85454545,
                -0.85454545,
            ]
        ),
    )
    np.testing.assert_almost_equal(
        hess,
        np.array([4.54545455] * 15),
    )
    assert grads.shape == hess.shape
    assert len(grads) == len(y_pred)


# Test evaluation functions
def test_eval_check_loss(dummy_data):
    """Test evaluation of the check loss."""
    y_pred, dtrain, alphas = dummy_data
    loss = _eval_check_loss(y_pred=y_pred, dtrain=dtrain, alphas=alphas)
    assert isinstance(loss, float)
    assert loss > 0


def test_xgb_eval_loss(dummy_data):
    """Test XGBoost evaluation function."""
    y_pred, dtrain, alphas = dummy_data
    metric_name, loss = _xgb_eval_loss(y_pred=y_pred, dtrain=dtrain, alphas=alphas)
    np.testing.assert_almost_equal(loss, 0.05)
    assert metric_name == "check_loss"
    assert isinstance(loss, float)


def test_lgb_eval_loss(dummy_data):
    """Test LightGBM evaluation function."""
    y_pred, dtrain, alphas = dummy_data
    metric_name, loss, higher_better = _lgb_eval_loss(
        y_pred=y_pred, dtrain=dtrain, alphas=alphas
    )
    np.testing.assert_almost_equal(loss, 0.05)
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
