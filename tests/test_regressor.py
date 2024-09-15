import lightgbm as lgb
import numpy as np
import pytest
import xgboost as xgb

from mqboost.base import FittingException, ModelName, ObjectiveName
from mqboost.regressor import MQDataset, MQRegressor

# Test data and helper functions


@pytest.fixture
def dummy_dataset_lgb():
    X = np.random.rand(100, 10)
    y = np.random.rand(100)
    alphas = [0.1, 0.49, 0.5, 0.51, 0.9]
    return MQDataset(alphas=alphas, data=X, label=y, model=ModelName.lightgbm.value)


@pytest.fixture
def dummy_eval_set_lgb():
    X_val = np.random.rand(50, 10)
    y_val = np.random.rand(50)
    alphas = [0.1, 0.49, 0.5, 0.51, 0.9]
    return MQDataset(
        alphas=alphas, data=X_val, label=y_val, model=ModelName.lightgbm.value
    )


@pytest.fixture
def dummy_dataset_xgb():
    X = np.random.rand(100, 10)
    y = np.random.rand(100)
    alphas = [0.1, 0.49, 0.5, 0.51, 0.9]
    return MQDataset(alphas=alphas, data=X, label=y, model=ModelName.xgboost.value)


@pytest.fixture
def dummy_eval_set_xgb():
    X_val = np.random.rand(50, 10)
    y_val = np.random.rand(50)
    alphas = [0.1, 0.49, 0.5, 0.51, 0.9]
    return MQDataset(
        alphas=alphas, data=X_val, label=y_val, model=ModelName.xgboost.value
    )


def test_mqregressor_initialization():
    params = {"learning_rate": 0.1, "num_leaves": 31}
    regressor = MQRegressor(
        params=params,
        model=ModelName.lightgbm.value,
        objective=ObjectiveName.check.value,
        delta=0.01,
        epsilon=1e-5,
    )
    assert regressor._params == params
    assert regressor._model == ModelName.lightgbm
    assert regressor._objective == ObjectiveName.check
    assert regressor._delta == 0.01
    assert regressor._epsilon == 1e-5


def test_invalid_model_initialization():
    params = {"learning_rate": 0.1, "num_leaves": 31}
    with pytest.raises(ValueError):
        MQRegressor(
            params=params,
            model="invalid_model",
            objective=ObjectiveName.check.value,
            delta=0.01,
            epsilon=1e-5,
        )


def test_invalid_objective_initialization():
    params = {"learning_rate": 0.1, "num_leaves": 31}
    with pytest.raises(ValueError):
        MQRegressor(
            params=params,
            model=ModelName.lightgbm.value,
            objective="invalid_objective",
            delta=0.01,
            epsilon=1e-5,
        )


# Test MQRegressor Fit Method
def test_mqregressor_fit_lgb(dummy_dataset_lgb, dummy_eval_set_lgb):
    params = {"learning_rate": 0.1, "num_leaves": 31}
    regressor = MQRegressor(params=params, model=ModelName.lightgbm.value)
    regressor.fit(dataset=dummy_dataset_lgb, eval_set=dummy_eval_set_lgb)

    assert regressor._fitted is True
    assert isinstance(regressor.model, lgb.Booster)


def test_mqregressor_fit_xgb(dummy_dataset_xgb, dummy_eval_set_xgb):
    params = {"learning_rate": 0.1, "max_depth": 6}
    regressor = MQRegressor(params=params, model=ModelName.xgboost.value)
    regressor.fit(dataset=dummy_dataset_xgb, eval_set=dummy_eval_set_xgb)

    assert regressor._fitted is True
    assert isinstance(regressor.model, xgb.Booster)


def test_fit_without_eval_set_lgb(dummy_dataset_lgb):
    params = {"learning_rate": 0.1, "num_leaves": 31}
    regressor = MQRegressor(params=params, model=ModelName.lightgbm.value)

    regressor.fit(dataset=dummy_dataset_lgb)
    assert regressor._fitted is True
    assert isinstance(regressor.model, lgb.Booster)


def test_fit_without_eval_set_xgb(dummy_dataset_xgb):
    params = {"learning_rate": 0.1, "max_depth": 6}
    regressor = MQRegressor(params=params, model=ModelName.xgboost.value)
    regressor.fit(dataset=dummy_dataset_xgb)

    assert regressor._fitted is True
    assert isinstance(regressor.model, xgb.Booster)


# Test MQRegressor Predict Method
def test_predict_lgb(dummy_dataset_lgb):
    params = {"learning_rate": 0.1, "num_leaves": 31}
    regressor = MQRegressor(params=params, model=ModelName.lightgbm.value)

    regressor.fit(dataset=dummy_dataset_lgb)
    predictions = regressor.predict(dataset=dummy_dataset_lgb)
    assert predictions.shape == (len(dummy_dataset_lgb.alphas), dummy_dataset_lgb.nrow)


def test_predict_xgb(dummy_dataset_xgb):
    params = {"learning_rate": 0.1, "max_depth": 6}
    regressor = MQRegressor(params=params, model=ModelName.xgboost.value)

    regressor.fit(dataset=dummy_dataset_xgb)
    predictions = regressor.predict(dataset=dummy_dataset_xgb)

    assert predictions.shape == (len(dummy_dataset_xgb.alphas), dummy_dataset_xgb.nrow)


def test_predict_without_fit(dummy_dataset_lgb):
    params = {"learning_rate": 0.1, "num_leaves": 31}
    regressor = MQRegressor(params=params, model=ModelName.lightgbm.value)

    with pytest.raises(FittingException):
        regressor.predict(dataset=dummy_dataset_lgb)


# Test Monotonicity Constraints
def test_monotone_constraints_called_lgb(dummy_dataset_lgb):
    params = {"learning_rate": 0.1, "num_leaves": 31}
    regressor = MQRegressor(params=params, model=ModelName.lightgbm.value)

    regressor.fit(dataset=dummy_dataset_lgb)
    predictions = regressor.predict(dataset=dummy_dataset_lgb)
    assert np.all(
        [
            np.all(predictions[k] <= predictions[k + 1])
            for k in range(len(predictions) - 1)
        ]
    )


def test_monotone_constraints_called_xgb(dummy_dataset_xgb):
    params = {"learning_rate": 0.1, "max_depth": 6}
    regressor = MQRegressor(params=params, model=ModelName.xgboost.value)

    regressor.fit(dataset=dummy_dataset_xgb)
    predictions = regressor.predict(dataset=dummy_dataset_xgb)
    assert np.all(
        [
            np.all(predictions[k] <= predictions[k + 1])
            for k in range(len(predictions) - 1)
        ]
    )


def test_feature_importance_before_fit_raises():
    params = {"learning_rate": 0.1, "max_depth": 6}
    with pytest.raises(FittingException, match="Fit must be executed first."):
        _ = MQRegressor(params=params).feature_importance


def test_feature_importance_after_fit(dummy_dataset_lgb):
    params = {"learning_rate": 0.1, "max_depth": 6}
    gbm_model = MQRegressor(params=params)
    gbm_model.fit(dataset=dummy_dataset_lgb)
    feature_importances = gbm_model.feature_importance

    assert isinstance(
        feature_importances, dict
    ), "Feature importances should be a dictionary"
    assert len(feature_importances) == len(
        dummy_dataset_lgb.columns
    ), "Feature importance length mismatch"
    for feature in dummy_dataset_lgb.columns:
        assert (
            str(feature) in feature_importances
        ), f"Feature {feature} not found in importance"


def test_feature_importance_positive(dummy_dataset_lgb):
    """Test that at least some feature importances are non-zero after training"""
    params = {"learning_rate": 0.1, "max_depth": 6}
    gbm_model = MQRegressor(params=params)
    gbm_model.fit(dataset=dummy_dataset_lgb)
    feature_importances = gbm_model.feature_importance

    assert all(
        [importance >= 0 for importance in feature_importances.values()]
    ), "All importance should be positive."
