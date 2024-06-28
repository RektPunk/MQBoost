from typing import List, Union, Any, Dict
from functools import partial
from enum import Enum

import numpy as np
import pandas as pd

import lightgbm as lgb
import xgboost as xgb

from module.utils import alpha_validate, prepare_train, prepare_x
from module.objective import check_loss_grad_hess, check_loss_eval


class ModelName(str, Enum):
    lightgbm: str = "lightgbm"
    xgboost: str = "xgboost"


_train_dataset_funcs = {
    "lightgbm": lgb.Dataset,
    "xgboost": xgb.DMatrix,
}

_monotone_constraints_type = {
    "lightgbm": list,
    "xgboost": tuple,
}

_predict_dataset_funcs = {
    "lightgbm": lambda x: x,
    "xgboost": xgb.DMatrix,
}


class MonotoneQuantileRegressor:
    def __init__(
        self,
        x: Union[pd.DataFrame, pd.Series, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        alphas: Union[List[float], float],
        _model_name: ModelName,
    ):
        alphas = alpha_validate(alphas)
        self._model_name = _model_name
        self.x_train, self.y_train = prepare_train(x, y, alphas)
        self.fobj = partial(check_loss_grad_hess, alphas=alphas)
        self.feval = partial(check_loss_eval, alphas=alphas)
        self.dataset = _train_dataset_funcs.get(self._model_name)(
            data=self.x_train, label=self.y_train
        )

    def train(self, params: Dict[str, Any]):
        self._params = params.copy()
        if "monotone_constraints" in self._params:
            _monotone_constraints = list(self._params["monotone_constraints"])
            _monotone_constraints.append(1)
            self._params["monotone_constraints"] = _monotone_constraints_type.get(
                self._model_name
            )(_monotone_constraints)
        else:
            self._params.update(
                {
                    "monotone_constraints": _monotone_constraints_type.get(
                        self._model_name
                    )([1 if "_tau" == col else 0 for col in self.x_train.columns])
                }
            )

    def predict(
        self,
        x: Union[pd.DataFrame, pd.Series, np.ndarray],
        alphas: Union[List[float], float],
    ) -> np.ndarray:
        alphas = alpha_validate(alphas)
        _x = prepare_x(x, alphas)
        _x = _predict_dataset_funcs.get(self._model_name)(_x)
        _pred = self.model.predict(_x)
        _pred = _pred.reshape(len(alphas), len(x))
        return _pred
