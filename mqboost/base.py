from collections.abc import Callable
from dataclasses import dataclass
from typing import Dict, List, Union

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb

from mqboost.objective import (
    check_loss_grad_hess,
    huber_loss_grad_hess,
    lgb_eval,
    xgb_eval,
)


# Name
class BaseName:
    def get(self, name):
        if name in self.__annotations__:
            return getattr(self, name)
        else:
            available_attributes = list(self.__annotations__.keys())
            raise AttributeError(
                f"'{self.__class__.__name__}' object has no attribute '{name}'. "
                f"Available attributes are: {', '.join(available_attributes)}"
            )


@dataclass
class ModelName(BaseName):
    lightgbm: str = "lightgbm"
    xgboost: str = "xgboost"


@dataclass
class ObjectiveName(BaseName):
    check: str = "check"
    huber: str = "huber"


@dataclass
class TypeName(BaseName):
    train_dtype: str = "train_dtype"
    predict_dtype: str = "predict_dtype"
    constraints_type: str = "constraints_type"
    eval: str = "eval"


# Functions
FUNC_TYPE: Dict[str, Dict[str, Callable]] = {
    ModelName.lightgbm: {
        TypeName.train_dtype: lgb.Dataset,
        TypeName.predict_dtype: lambda x: x,
        TypeName.constraints_type: list,
        TypeName.eval: lgb_eval,
    },
    ModelName.xgboost: {
        TypeName.train_dtype: xgb.DMatrix,
        TypeName.predict_dtype: xgb.DMatrix,
        TypeName.constraints_type: tuple,
        TypeName.eval: xgb_eval,
    },
}

OBJECTIVE_FUNC: Dict[str, Callable] = {
    "check": check_loss_grad_hess,
    "huber": huber_loss_grad_hess,
}

# Type
XdataLike = Union[pd.DataFrame, pd.Series, np.ndarray]
YdataLike = Union[pd.Series, np.ndarray]
AlphaLike = Union[List[float], float]
ModelLike = Union[lgb.basic.Booster, xgb.Booster]


# Exception
class FittingException(Exception):
    pass


class ValidationException(Exception):
    pass
