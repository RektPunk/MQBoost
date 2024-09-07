import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest
import xgboost as xgb

from mqboost.base import (
    FUNC_TYPE,
    FittingException,
    ModelName,
    ObjectiveName,
    TypeName,
    ValidationException,
    _lgb_predict_dtype,
)


# Test Enum behavior
def test_model_name_enum():
    assert ModelName.get("lightgbm") == ModelName.lightgbm
    assert ModelName.get("xgboost") == ModelName.xgboost

    with pytest.raises(ValueError):
        ModelName.get("invalid_model")


def test_objective_name_enum():
    assert ObjectiveName.get("check") == ObjectiveName.check
    assert ObjectiveName.get("huber") == ObjectiveName.huber
    assert ObjectiveName.get("approx") == ObjectiveName.approx

    with pytest.raises(ValueError):
        ObjectiveName.get("invalid_objective")


# Test FUNC_TYPE
def test_func_type_for_lightgbm():
    assert FUNC_TYPE[ModelName.lightgbm][TypeName.train_dtype] == lgb.Dataset
    assert isinstance(
        FUNC_TYPE[ModelName.lightgbm][TypeName.predict_dtype](pd.DataFrame([1, 2, 3])),
        pd.DataFrame,
    )
    assert isinstance(
        FUNC_TYPE[ModelName.lightgbm][TypeName.predict_dtype](pd.Series([1, 2, 3])),
        pd.Series,
    )
    assert isinstance(
        FUNC_TYPE[ModelName.lightgbm][TypeName.predict_dtype](np.array([1, 2, 3])),
        np.ndarray,
    )
    assert FUNC_TYPE[ModelName.lightgbm][TypeName.constraints_type] == list


def test_func_type_for_xgboost():
    assert FUNC_TYPE[ModelName.xgboost][TypeName.train_dtype] == xgb.DMatrix
    assert FUNC_TYPE[ModelName.xgboost][TypeName.predict_dtype] == xgb.DMatrix
    assert FUNC_TYPE[ModelName.xgboost][TypeName.constraints_type] == tuple


# Test _lgb_predict_dtype
def test_lgb_predict_dtype():
    data = pd.DataFrame([1, 2, 3])
    assert _lgb_predict_dtype(data) is data

    array_data = np.array([1, 2, 3])
    assert _lgb_predict_dtype(array_data) is array_data


# Test custom exceptions
def test_custom_exceptions():
    with pytest.raises(FittingException):
        raise FittingException("Fitting failed")

    with pytest.raises(ValidationException):
        raise ValidationException("Validation failed")
