from enum import Enum
from typing import Any

import lightgbm as lgb
import xgboost as xgb


class ModelName(str, Enum):
    lightgbm = "lightgbm"
    xgboost = "xgboost"


class ObjectiveName(str, Enum):
    check = "check"
    huber = "huber"
    approx = "approx"


class TypeName(str, Enum):
    train_dtype = "train_dtype"
    predict_dtype = "predict_dtype"
    constraints_type = "constraints_type"


FUNC_TYPE: dict[ModelName, dict[TypeName, Any]] = {
    ModelName.lightgbm: {
        TypeName.train_dtype: lgb.Dataset,
        TypeName.predict_dtype: lambda data: data,
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
