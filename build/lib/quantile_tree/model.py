from typing import List, Union, Dict, Any

import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb

from .abstract import MonotoneQuantileRegressor

__all__ = [
    "QuantileRegressorLgb",
    "QuantileRegressorXgb",
]


class QuantileRegressorLgb(MonotoneQuantileRegressor):
    """
    Monotone quantile regressor which preserving monotonicity among quantiles
    Attributes
    ----------
    x: Union[pd.DataFrame, pd.Series, np.ndarray]
    y: Union[pd.Series, np.ndarray]
    alphas: Union[List[float], float]

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
    ):
        super().__init__(
            x=x,
            y=y,
            alphas=alphas,
            _model_name="lightgbm",
        )

    def train(self, params: Dict[str, Any]) -> lgb.basic.Booster:
        """
        Train regressor and return model
        Args:
            params (Dict[str, Any]): params of lgb

        Returns:
            lgb.basic.Booster
        """
        super().train(params=params)
        self._params.update({"objective": self.fobj})
        self.model = lgb.train(
            train_set=self.dataset,
            params=self._params,
            feval=self.feval,
        )
        return self.model


class QuantileRegressorXgb(MonotoneQuantileRegressor):
    """
    Monotone quantile regressor which preserving monotonicity among quantiles
    Attributes
    ----------
    x: Union[pd.DataFrame, pd.Series, np.ndarray]
    y: Union[pd.Series, np.ndarray]
    alphas: Union[List[float], float]

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
    ):
        super().__init__(
            x=x,
            y=y,
            alphas=alphas,
            _model_name="xgboost",
        )

    def train(self, params: Dict[str, Any]) -> xgb.Booster:
        """
        Train regressor and return model
        Args:
            params (Dict[str, Any]): params of xgb

        Returns:
            xgb.Booster
        """
        super().train(params=params)
        self.model = xgb.train(
            dtrain=self.dataset,
            verbose_eval=False,
            params=self._params,
            obj=self.fobj,
        )
        return self.model
