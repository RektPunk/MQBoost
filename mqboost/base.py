from enum import StrEnum
from typing import Callable

import lightgbm as lgb
import numpy.typing as npt
import pandas as pd
import xgboost as xgb

# Type
XdataLike = pd.DataFrame | pd.Series | npt.NDArray
YdataLike = pd.Series | npt.NDArray
AlphaLike = list[float] | float
ModelLike = lgb.basic.Booster | xgb.Booster
DtrainLike = lgb.basic.Dataset | xgb.DMatrix
ParamsLike = dict[str, float | int | str | bool | list[int]]
WeightLike = list[float] | list[int] | npt.NDArray | pd.Series


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
def _lgb_predict_dtype(data: XdataLike):
    return data


FUNC_TYPE: dict[ModelName, dict[TypeName, Callable]] = {
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
