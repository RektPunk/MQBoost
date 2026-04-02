from enum import StrEnum
from typing import Any

import lightgbm as lgb
import xgboost as xgb


# Name
class ModelName(StrEnum):
    lightgbm = "lightgbm"
    xgboost = "xgboost"


class ObjectiveName(StrEnum):
    check = "check"
    huber = "huber"
    approx = "approx"


class TypeName(StrEnum):
    train_dtype = "train_dtype"
    predict_dtype = "predict_dtype"
    constraints_type = "constraints_type"


# Functions
def _lgb_predict_dtype(data: Any):
    return data


FUNC_TYPE: dict[ModelName, dict[TypeName, Any]] = {
    ModelName.lightgbm: {
        TypeName.train_dtype: lgb.Dataset,
        TypeName.predict_dtype: _lgb_predict_dtype,
        TypeName.constraints_type: list,
    },
    ModelName.xgboost: {
        TypeName.train_dtype: xgb.DMatrix,
        TypeName.predict_dtype: xgb.DMatrix,
        TypeName.constraints_type: tuple,
    },
}


# Exception
class FittingException(Exception):
    pass


class ValidationException(Exception):
    pass
