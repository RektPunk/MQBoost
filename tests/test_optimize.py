from unittest.mock import MagicMock, patch

import numpy as np
import optuna
import pandas as pd
import pytest

from mqboost.base import FittingException, ModelName, ObjectiveName, ValidationException
from mqboost.dataset import MQDataset
from mqboost.objective import MQObjective
from mqboost.optimize import MQOptimizer

# Mocking lightgbm and xgboost to avoid actual model training during tests
# with patch.dict(
#     "sys.modules",
#     {
#         "lightgbm": MagicMock(),
#         "xgboost": MagicMock(),
#         "lightgbm.train": MagicMock(),
#         "xgboost.train": MagicMock(),
#     },
# ):
#     import lightgbm as lgb
#     import xgboost as xgb


# Fixtures for test data
@pytest.fixture
def sample_data():
    """Generates sample data for testing."""
    X = pd.DataFrame(np.random.rand(100, 5), columns=[f"feature_{i}" for i in range(5)])
    y = np.random.rand(100)
    alphas = [0.1, 0.5, 0.9]
    dataset = MQDataset(alphas=alphas, data=X, label=y, model=ModelName.lightgbm.value)
    return dataset


@pytest.fixture
def sample_valid_data():
    """Generates sample validation data."""
    X_valid = pd.DataFrame(
        np.random.rand(20, 5), columns=[f"feature_{i}" for i in range(5)]
    )
    y_valid = np.random.rand(20)
    alphas = [0.1, 0.5, 0.9]
    valid_set = MQDataset(
        alphas=alphas, data=X_valid, label=y_valid, model=ModelName.lightgbm.value
    )
    return valid_set


# Test MQOptimizer Initialization
def test_mqoptimizer_initialization():
    """Test initialization with valid parameters."""
    optimizer = MQOptimizer(
        model=ModelName.lightgbm.value,
        objective=ObjectiveName.check.value,
        delta=0.01,
        epsilon=1e-5,
    )
    assert optimizer._model == ModelName.lightgbm
    assert optimizer._objective == ObjectiveName.check
    assert optimizer._delta == 0.01
    assert optimizer._epsilon == 1e-5


def test_mqoptimizer_invalid_model():
    """Test initialization with an invalid model."""
    with pytest.raises(ValueError):
        MQOptimizer(model="invalid_model")


def test_mqoptimizer_invalid_objective():
    """Test initialization with an invalid objective."""
    with pytest.raises(ValueError):
        MQOptimizer(objective="invalid_objective")


def test_mqoptimizer_invalid_delta():
    """Test initialization with invalid delta value."""
    with pytest.raises(ValidationException):
        MQOptimizer(delta=-0.01)


def test_mqoptimizer_invalid_epsilon():
    """Test initialization with invalid epsilon value."""
    with pytest.raises(ValidationException):
        MQOptimizer(epsilon=-1e-5)


# Test optimize_params method
def test_optimize_params_with_default_get_params(sample_data):
    """Test optimize_params with default get_params function."""
    optimizer = MQOptimizer()
    with patch.object(optimizer, "_MQObj", create=True):
        optimizer._MQObj = MagicMock(spec=MQObjective)
        optimizer._MQObj.feval.return_value = ("metric", 0.1, False)

        # Mock the training functions to avoid actual training
        with patch("lightgbm.train", return_value=MagicMock()):
            best_params = optimizer.optimize_params(dataset=sample_data, n_trials=1)
            assert isinstance(best_params, dict)
            assert optimizer._is_optimized


def test_optimize_params_with_custom_get_params(sample_data):
    """Test optimize_params with a custom get_params function."""
    optimizer = MQOptimizer()

    def custom_get_params(trial):
        return {
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1),
            "num_leaves": trial.suggest_int("num_leaves", 10, 50),
        }

    with patch.object(optimizer, "_MQObj", create=True):
        optimizer._MQObj = MagicMock(spec=MQObjective)
        optimizer._MQObj.feval.return_value = ("metric", 0.05, False)

        with patch("lightgbm.train", return_value=MagicMock()):
            best_params = optimizer.optimize_params(
                dataset=sample_data, n_trials=1, get_params_func=custom_get_params
            )
            assert "learning_rate" in best_params
            assert "num_leaves" in best_params


def test_optimize_params_with_valid_set(sample_data, sample_valid_data):
    """Test optimize_params with a provided validation set."""
    optimizer = MQOptimizer()

    with patch.object(optimizer, "_MQObj", create=True):
        optimizer._MQObj = MagicMock(spec=MQObjective)
        optimizer._MQObj.feval.return_value = ("metric", 0.05, False)

        with patch("lightgbm.train", return_value=MagicMock()):
            best_params = optimizer.optimize_params(
                dataset=sample_data, n_trials=1, valid_set=sample_valid_data
            )
            assert isinstance(best_params, dict)
            assert optimizer._is_optimized


def test_optimize_params_without_optimization(sample_data):
    """Test accessing best_params before optimization is completed."""
    optimizer = MQOptimizer()
    with pytest.raises(FittingException, match="Optimization is not completed."):
        _ = optimizer.best_params


def test_study_property_before_optimization(sample_data):
    """Test accessing study property before optimization."""
    optimizer = MQOptimizer()
    assert optimizer.study is None


def test_study_property_after_optimization(sample_data):
    """Test accessing study property after optimization."""
    optimizer = MQOptimizer()

    with patch.object(optimizer, "_MQObj", create=True):
        optimizer._MQObj = MagicMock(spec=MQObjective)
        optimizer._MQObj.feval.return_value = ("metric", 0.05, False)

        with patch("lightgbm.train", return_value=MagicMock()):
            optimizer.optimize_params(dataset=sample_data, n_trials=1)
            assert isinstance(optimizer.study, optuna.Study)


def test_mqobjective_property(sample_data):
    """Test MQObj property after initialization."""
    optimizer = MQOptimizer()
    optimizer._MQObj = MQObjective(
        alphas=sample_data.alphas,
        objective=optimizer._objective,
        model=optimizer._model,
        delta=optimizer._delta,
        epsilon=optimizer._epsilon,
    )
    assert isinstance(optimizer.MQObj, MQObjective)


def test_optimize_params_invalid_dataset():
    """Test optimize_params with an invalid dataset."""
    optimizer = MQOptimizer()
    with pytest.raises(AttributeError):
        optimizer.optimize_params(dataset=None, n_trials=1)


def test_optimize_params_invalid_n_trials(sample_data):
    """Test optimize_params with invalid n_trials."""
    optimizer = MQOptimizer()
    with pytest.raises(ValueError):
        optimizer.optimize_params(dataset=sample_data, n_trials=0)


def test_optimize_params_with_exception_in_objective(sample_data):
    """Test that exceptions in the objective function are handled."""
    optimizer = MQOptimizer()

    with patch("lightgbm.train", side_effect=Exception("Training failed")):
        with pytest.raises(Exception, match="Training failed"):
            optimizer.optimize_params(dataset=sample_data, n_trials=1)


def test_lgb_get_params():
    """Test the default LightGBM get_params function."""
    trial = optuna.trial.FixedTrial(
        {
            "learning_rate": 0.05,
            "max_depth": 5,
            "lambda_l1": 0.1,
            "lambda_l2": 0.1,
            "num_leaves": 31,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
        }
    )
    params = _lgb_get_params(trial)
    expected_keys = {
        "verbose",
        "learning_rate",
        "max_depth",
        "lambda_l1",
        "lambda_l2",
        "num_leaves",
        "feature_fraction",
        "bagging_fraction",
        "bagging_freq",
    }
    assert set(params.keys()) == expected_keys


def test_xgb_get_params():
    """Test the default XGBoost get_params function."""
    trial = optuna.trial.FixedTrial(
        {
            "learning_rate": 0.05,
            "max_depth": 5,
            "reg_lambda": 0.1,
            "reg_alpha": 0.1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
        }
    )
    params = _xgb_get_params(trial)
    expected_keys = {
        "learning_rate",
        "max_depth",
        "reg_lambda",
        "reg_alpha",
        "subsample",
        "colsample_bytree",
    }
    assert set(params.keys()) == expected_keys
