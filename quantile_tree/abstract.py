from typing import List, Union, Any, Dict, Callable
from functools import partial
from enum import Enum

import numpy as np
import pandas as pd

import lightgbm as lgb
import xgboost as xgb

from .utils import alpha_validate, prepare_train, prepare_x
from .objective import check_loss_grad_hess, check_loss_eval


__all__ = ["MonotoneQuantileRegressor"]


class _ModelName(str, Enum):
    lightgbm: str = "lightgbm"
    xgboost: str = "xgboost"


TRAIN_DATASET_FUNCS: Dict[str, Union[lgb.Dataset, xgb.DMatrix]] = {
    "lightgbm": lgb.Dataset,
    "xgboost": xgb.DMatrix,
}

MONOTONE_CONSTRAINTS_TYPE: Dict[str, Union[list, tuple]] = {
    "lightgbm": list,
    "xgboost": tuple,
}

PREDICT_DATASET_FUNCS: Dict[str, Union[Callable, xgb.DMatrix]] = {
    "lightgbm": lambda x: x,
    "xgboost": xgb.DMatrix,
}


class MonotoneQuantileRegressor:
    """
    Monotone quantile regressor which preserving monotonicity among quantiles
    Attributes
    ----------
    x: Union[pd.DataFrame, pd.Series, np.ndarray]
    y: Union[pd.Series, np.ndarray]
    alphas: Union[List[float], float]
    _model_name: _ModelName

    Methods
    -------
    train
    predict
    """

    def __init__(
        self,
        x: Union[pd.DataFrame, pd.Series, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        alphas: Union[List[float], float],
        _model_name: _ModelName,
    ):
        """
        Set objevtive, dataset

        Args:
            x (Union[pd.DataFrame, pd.Series, np.ndarray])
            y (Union[pd.Series, np.ndarray])
            alphas (Union[List[float], float])
            _model_name (_ModelName)
        """
        alphas = alpha_validate(alphas)
        self._model_name = _model_name
        self.x_train, self.y_train = prepare_train(x, y, alphas)
        self.fobj = partial(check_loss_grad_hess, alphas=alphas)
        self.feval = partial(check_loss_eval, alphas=alphas)
        self.dataset = TRAIN_DATASET_FUNCS.get(self._model_name)(
            data=self.x_train, label=self.y_train
        )

    def train(self, params: Dict[str, Any]):
        """
        Set monotone constraints in params

        Args:
            params (Dict[str, Any])
        """
        self._params = params.copy()
        monotone_constraints_str: str = "monotone_constraints"
        if monotone_constraints_str in self._params:
            _monotone_constraints = list(self._params[monotone_constraints_str])
            _monotone_constraints.append(1)
            self._params[monotone_constraints_str] = MONOTONE_CONSTRAINTS_TYPE.get(
                self._model_name
            )(_monotone_constraints)
        else:
            self._params.update(
                {
                    monotone_constraints_str: MONOTONE_CONSTRAINTS_TYPE.get(
                        self._model_name
                    )([1 if "_tau" == col else 0 for col in self.x_train.columns])
                }
            )

    def predict(
        self,
        x: Union[pd.DataFrame, pd.Series, np.ndarray],
        alphas: Union[List[float], float],
    ) -> np.ndarray:
        """
        Return predicted quantiles
        Args:
            x (Union[pd.DataFrame, pd.Series, np.ndarray])
            alphas (Union[List[float], float])

        Returns:
            np.ndarray
        """
        alphas = alpha_validate(alphas)
        _x = prepare_x(x, alphas)
        _x = PREDICT_DATASET_FUNCS.get(self._model_name)(_x)
        _pred = self.model.predict(_x)
        _pred = _pred.reshape(len(alphas), len(x))
        return _pred
