from typing import Union, Dict, Callable, List
from dataclasses import dataclass

import numpy as np
import pandas as pd

import lightgbm as lgb
import xgboost as xgb

from .objective import check_loss_grad_hess, huber_loss_grad_hess


@dataclass
class ModelName:
    lightgbm: str = "lightgbm"
    xgboost: str = "xgboost"


@dataclass
class ObjectiveName:
    check: str = "check"
    huber: str = "huber"

    def get(self, name):
        if name in self.__annotations__:
            return getattr(self, name)
        else:
            available_attributes = list(self.__annotations__.keys())
            raise AttributeError(
                f"'{self.__class__.__name__}' object has no attribute '{name}'. "
                f"Available attributes are: {', '.join(available_attributes)}"
            )


TRAIN_DATASET_FUNC: Dict[str, Union[lgb.Dataset, xgb.DMatrix]] = {
    "lightgbm": lgb.Dataset,
    "xgboost": xgb.DMatrix,
}

MONOTONE_CONSTRAINTS_TYPE: Dict[str, Union[list, tuple]] = {
    "lightgbm": list,
    "xgboost": tuple,
}

PREDICT_DATASET_FUNC: Dict[str, Union[Callable, xgb.DMatrix]] = {
    "lightgbm": lambda x: x,
    "xgboost": xgb.DMatrix,
}

OBJECTIVE_FUNC: Dict[str, Callable] = {
    "check": check_loss_grad_hess,
    "huber": huber_loss_grad_hess,
}

XdataLike = Union[pd.DataFrame, pd.Series, np.ndarray]
YdataLike = Union[pd.Series, np.ndarray]
AlphaLike = Union[List[float], float]
