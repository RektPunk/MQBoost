from enum import Enum
from typing import Callable

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb


class BaseEnum(Enum):
    @classmethod
    def get(cls, text: str) -> "BaseEnum":
        cls._isin(text)
        return cls[text]

    @classmethod
    def _isin(cls, text: str) -> None:
        if text not in cls._member_names_:
            valid_members = ", ".join(cls._member_names_)
            raise ValueError(
                f"Invalid value: '{text}'. Expected one of: {valid_members}."
            )


# Type
XdataLike = pd.DataFrame | pd.Series | np.ndarray
YdataLike = pd.Series | np.ndarray
AlphaLike = list[float] | float
ModelLike = lgb.basic.Booster | xgb.Booster
DtrainLike = lgb.basic.Dataset | xgb.DMatrix
ParamsLike = dict[str, float | int | str | bool]
WeightLike = list[float] | list[int] | np.ndarray | pd.Series


# Name
class ModelName(BaseEnum):
    lightgbm: str = "lightgbm"
    xgboost: str = "xgboost"


class ObjectiveName(BaseEnum):
    check: str = "check"
    huber: str = "huber"
    approx: str = "approx"


class TypeName(BaseEnum):
    train_dtype: str = "train_dtype"
    predict_dtype: str = "predict_dtype"
    constraints_type: str = "constraints_type"


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
