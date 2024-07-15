from __future__ import annotations
from typing import Any
from functools import partial

import lightgbm as lgb
import numpy as np
import xgboost as xgb

from mqboost.base import (
    EVAL_FUNC,
    MONOTONE_CONSTRAINTS_TYPE,
    OBJECTIVE_FUNC,
    PREDICT_DATASET_FUNC,
    TRAIN_DATASET_FUNC,
    AlphaLike,
    ModelLike,
    XdataLike,
    YdataLike,
    FittingException,
    ModelName,
    ObjectiveName,
)
from mqboost.objective import CHECK_LOSS
from mqboost.utils import alpha_validate, delta_validate, prepare_train, prepare_x

__all__ = ["MQRegressor"]


class MQRegressor:
    """
    Monotone quantile regressor which preserving monotonicity among quantiles with LightGBM
    Attributes
    ----------
    x: XdataLike
    y: YdataLike
    alphas: AlphaLike
        It must be in ascending order and not contain duplicates
    objective: str
        Determine objective function. options: "check" (default), "huber"
        If objective is "huber", you can set "delta" (default = 0.05)
        "delta" must be smaller than 0.1
    model: str
        Determine base model. options: "lightgbm" (default), "xgboost"
    **kwargs: Any

    Methods
    ----------
    train
    predict

    Property
    ----------
    train_score
    """

    def __init__(
        self,
        x: XdataLike,
        y: YdataLike,
        alphas: AlphaLike,
        model: str = ModelName.lightgbm,
        objective: str = ObjectiveName.check,
        **kwargs: Any,
    ) -> None:
        """
        Set objective, dataset
        Args:
            x (XdataLike)
            y (YdataLike)
            alphas (AlphaLike)
            model (ModelName)
            objective (ObjectiveName)
            **kwargs (Any)
        """
        alphas = alpha_validate(alphas)
        self._model = ModelName().get(model)
        self._objective = ObjectiveName().get(objective)
        self.x_train, self.y_train = prepare_train(x, y, alphas)
        if self._objective == ObjectiveName.huber:
            delta = kwargs.get("delta", 0.05)
            delta_validate(delta)
            self.fobj = partial(
                OBJECTIVE_FUNC.get(objective), alphas=alphas, delta=delta
            )
        else:
            self.fobj = partial(OBJECTIVE_FUNC.get(objective), alphas=alphas)
        self.feval = partial(EVAL_FUNC.get(model), alphas=alphas)
        self.dataset = TRAIN_DATASET_FUNC.get(self._model)(
            data=self.x_train, label=self.y_train
        )

    def __set_params(self, params: dict[str, Any]) -> None:
        """
        Set monotone constraints in params
        Args:
            params (dict[str, Any])
        """
        if isinstance(params, dict) and "objective" in params:
            raise FittingException(
                "The parameter named 'objective' must not be included in params"
            )
        self._params = params.copy()
        monotone_constraints_str: str = "monotone_constraints"
        if monotone_constraints_str in self._params:
            _monotone_constraints = list(self._params[monotone_constraints_str])
            _monotone_constraints.append(1)
            self._params[monotone_constraints_str] = MONOTONE_CONSTRAINTS_TYPE.get(
                self._model
            )(_monotone_constraints)
        else:
            self._params.update(
                {
                    monotone_constraints_str: MONOTONE_CONSTRAINTS_TYPE.get(
                        self._model
                    )([1 if "_tau" == col else 0 for col in self.x_train.columns])
                }
            )

    def train(self, params: dict[str, Any]) -> ModelLike:
        """
        Train regressor and return model
        Args:
            params (dict[str, Any])

        Returns:
            ModelLike
        """
        self.__set_params(params=params)

        if self._model == ModelName.lightgbm:
            self._params.update({"objective": self.fobj})
            self.model = lgb.train(
                train_set=self.dataset,
                params=self._params,
                feval=self.feval,
                valid_sets=[self.dataset],
            )
            self._train_score = self.model.best_score["training"][CHECK_LOSS]
        else:
            _evals_result = {}
            self.model = xgb.train(
                dtrain=self.dataset,
                verbose_eval=False,
                params=self._params,
                obj=self.fobj,
                custom_metric=self.feval,
                evals=[
                    (self.dataset, "train"),
                ],
                evals_result=_evals_result,
            )
            self._train_score = min(_evals_result["train"][CHECK_LOSS])
        self._fitted = True
        return self.model

    def predict(
        self,
        x: XdataLike,
        alphas: AlphaLike,
    ) -> np.ndarray:
        """
        Return predicted quantiles
        Args:
            x (XdataLike)
            alphas (AlphaLike)

        Returns:
            np.ndarray
        """
        if not self.__is_fitted:
            raise FittingException("train must be executed before predict")
        alphas = alpha_validate(alphas)
        _x = prepare_x(x, alphas)
        _x = PREDICT_DATASET_FUNC.get(self._model)(_x)
        _pred = self.model.predict(_x)
        _pred = _pred.reshape(len(alphas), len(x))
        return _pred

    @property
    def __is_fitted(self) -> bool:
        return getattr(self, "_fitted", False)

    @property
    def train_score(self) -> float:
        if not self.__is_fitted:
            raise FittingException("train must be executed before predict")
        return float(self._train_score)
