import pandas as pd

from mqboost.base import FUNC_TYPE, ModelName, MQStr, TypeName
from mqboost.constraints import set_monotone_constraints


# Test function for setting monotone constraints
def test_set_monotone_constraints_with_existing_constraints():
    # Test data
    params = {
        MQStr.mono.value: [1, -1],
    }
    columns = pd.Index(["feature_1", "feature_2", "_tau"])
    model_name = ModelName.lightgbm

    # Call function
    updated_params = set_monotone_constraints(params, columns, model_name)

    # Expected constraints list will append 1 and be converted by constraint function
    expected_constraints = FUNC_TYPE[ModelName.lightgbm][TypeName.constraints_type](
        [1, -1, 1]
    )

    # Assert the result
    assert updated_params[MQStr.mono.value] == expected_constraints


def test_set_monotone_constraints_without_existing_constraints():
    # Test data
    params = {}
    columns = pd.Index(["feature_1", "feature_2", "_tau"])
    model_name = ModelName.xgboost

    # Call function
    updated_params = set_monotone_constraints(params, columns, model_name)

    # Expected constraints: _tau column will get 1, rest 0
    expected_constraints = FUNC_TYPE[ModelName.xgboost][TypeName.constraints_type](
        [0, 0, 1]
    )

    # Assert the result
    assert updated_params[MQStr.mono.value] == expected_constraints


def test_set_monotone_constraints_with_empty_columns():
    # Test data with empty columns
    params = {}
    columns = pd.Index([])
    model_name = ModelName.lightgbm

    # Call function
    updated_params = set_monotone_constraints(params, columns, model_name)

    # Expected constraints: empty since no columns
    expected_constraints = FUNC_TYPE[ModelName.lightgbm][TypeName.constraints_type]([])

    # Assert the result
    assert updated_params[MQStr.mono.value] == expected_constraints


# Edge case where no constraints and empty columns
def test_set_monotone_constraints_with_empty_params_and_columns():
    # Test data with empty params and empty columns
    params = {}
    columns = pd.Index([])
    model_name = ModelName.xgboost

    # Call function
    updated_params = set_monotone_constraints(params, columns, model_name)

    # Expected constraints: empty list or tuple, depending on model
    expected_constraints = FUNC_TYPE[ModelName.xgboost][TypeName.constraints_type]([])

    # Assert the result
    assert updated_params[MQStr.mono.value] == expected_constraints
