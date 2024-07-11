from typing import Dict, Any

import lightgbm as lgb
import xgboost as xgb

from .base import ModelName, ObjectiveName, XdataLike, YdataLike, AlphaLike
from .engine import MonotoneQuantileRegressor

__all__ = [
    "QuantileRegressorLgb",
    "QuantileRegressorXgb",
]


class QuantileRegressorLgb(MonotoneQuantileRegressor):
    """
    Monotone quantile regressor which preserving monotonicity among quantiles with LightGBM
    Attributes
    ----------
    x: XdataLike
    y: YdataLike
    alphas: AlphaLike
    objective: ObjectiveName
        Determine objective function. options: "check" (default), "huber"
        If objective is "huber", you can set "delta" (default = 0.05)
    **kwargs: Any

    Methods
    -------
    train
    predict
    """

    def __init__(
        self,
        x: XdataLike,
        y: YdataLike,
        alphas: AlphaLike,
        objective: ObjectiveName = ObjectiveName.check,
        **kwargs: Any,
    ):
        super().__init__(
            x=x,
            y=y,
            alphas=alphas,
            objective=objective,
            _model_name=ModelName.lightgbm,
            **kwargs,
        )

    def train(self, params: Dict[str, Any]) -> lgb.basic.Booster:
        """
        Train regressor and return model
        Args:
            params (Dict[str, Any]): params of lgb

        Returns:
            lgb.basic.Booster
        """
        super().set_params(params=params)
        self._params.update({"objective": self.fobj})
        self.model = lgb.train(
            train_set=self.dataset,
            params=self._params,
            # feval=self.feval, #TODO
        )

        return self.model


class QuantileRegressorXgb(MonotoneQuantileRegressor):
    """
    Monotone quantile regressor which preserving monotonicity among quantiles with XGBoost
    Attributes
    ----------
    x: XdataLike
    y: YdataLike
    alphas: AlphaLike
    objective: ObjectiveName
        Determine objective function. options: "check" (default), "huber"
        If objective is "huber", you can set "delta" (default = 0.05)
    **kwargs: Any

    Methods
    -------
    train
    predict
    """

    def __init__(
        self,
        x: XdataLike,
        y: YdataLike,
        alphas: AlphaLike,
        objective: ObjectiveName = ObjectiveName.check,
        **kwargs: Any,
    ):
        super().__init__(
            x=x,
            y=y,
            alphas=alphas,
            objective=objective,
            _model_name=ModelName.xgboost,
            **kwargs,
        )

    def train(self, params: Dict[str, Any]) -> xgb.Booster:
        """
        Train regressor and return model
        Args:
            params (Dict[str, Any]): params of xgb

        Returns:
            xgb.Booster
        """
        super().set_params(params=params)
        self.model = xgb.train(
            dtrain=self.dataset,
            verbose_eval=False,
            params=self._params,
            obj=self.fobj,
        )

        return self.model
