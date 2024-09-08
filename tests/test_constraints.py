import pandas as pd

from mqboost.base import ModelName, MQStr
from mqboost.constraints import set_monotone_constraints


# Test function for setting monotone constraints
def test_set_monotone_constraints_with_existing_constraints_lgb():
    params = {
        MQStr.mono.value: [1, -1],
    }
    columns = pd.Index(["feature_1", "feature_2", "_tau"])
    model_name = ModelName.lightgbm
    updated_params = set_monotone_constraints(params, columns, model_name)
    expected_constraints = [1, -1, 1]
    assert updated_params[MQStr.mono.value] == expected_constraints


def test_set_monotone_constraints_without_existing_constraints_lgb():
    params = {}
    columns = pd.Index(["feature_1", "feature_2", "_tau"])
    model_name = ModelName.lightgbm
    updated_params = set_monotone_constraints(params, columns, model_name)
    expected_constraints = [0, 0, 1]
    assert updated_params[MQStr.mono.value] == expected_constraints


def test_set_monotone_constraints_with_existing_constraints_xgb():
    params = {
        MQStr.mono.value: [1, -1],
    }
    columns = pd.Index(["feature_1", "feature_2", "_tau"])
    model_name = ModelName.xgboost
    updated_params = set_monotone_constraints(params, columns, model_name)
    expected_constraints = (1, -1, 1)
    assert updated_params[MQStr.mono.value] == expected_constraints


def test_set_monotone_constraints_without_existing_constraints_xgb():
    params = {}
    columns = pd.Index(["feature_1", "feature_2", "_tau"])
    model_name = ModelName.xgboost
    updated_params = set_monotone_constraints(params, columns, model_name)
    expected_constraints = (0, 0, 1)
    assert updated_params[MQStr.mono.value] == expected_constraints


# Edge case where no constraints or empty columns
def test_set_monotone_constraints_with_empty_columns():
    params = {}
    columns = pd.Index([])
    model_name = ModelName.lightgbm
    updated_params = set_monotone_constraints(params, columns, model_name)
    expected_constraints = []
    assert updated_params[MQStr.mono.value] == expected_constraints


def test_set_monotone_constraints_with_empty_params_and_columns():
    params = {}
    columns = pd.Index([])
    model_name = ModelName.xgboost
    updated_params = set_monotone_constraints(params, columns, model_name)
    expected_constraints = ()
    assert updated_params[MQStr.mono.value] == expected_constraints
