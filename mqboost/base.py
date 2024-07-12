from dataclasses import dataclass
from typing import Callable, Dict, List, Union

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb

from mqboost.objective import check_loss_grad_hess, huber_loss_grad_hess


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


# Functions
TRAIN_DATASET_FUNC: Dict[str, Callable] = {
    "lightgbm": lgb.Dataset,
    "xgboost": xgb.DMatrix,
}

MONOTONE_CONSTRAINTS_TYPE: Dict[str, Callable] = {
    "lightgbm": list,
    "xgboost": tuple,
}

PREDICT_DATASET_FUNC: Dict[str, Callable] = {
    "lightgbm": lambda x: x,
    "xgboost": xgb.DMatrix,
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
