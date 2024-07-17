from dataclasses import dataclass
from functools import partial
from typing import Any, Dict

import lightgbm as lgb
import numpy as np
import xgboost as xgb

from mqboost.base import (
    FUNC_TYPE,
    OBJECTIVE_FUNC,
    AlphaLike,
    FittingException,
    ModelLike,
    ModelName,
    ObjectiveName,
    TypeName,
    ValidationException,
    XdataLike,
    YdataLike,
)
from mqboost.objective import CHECK_LOSS
from mqboost.utils import alpha_validate, delta_validate, prepare_train, prepare_x

__all__ = ["MQRegressor"]


@dataclass
class _MQStr:
    _mono: str = "monotone_constraints"
    _obj: str = "objective"
    _tr: str = "train"
    _trg: str = "training"


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
        self._funcs = FUNC_TYPE.get(model)
        self.x_train, self.y_train = prepare_train(x, y, alphas)
        if self._objective == ObjectiveName.huber:
            delta = kwargs.get("delta", 0.05)
            delta_validate(delta)
            self.fobj = partial(
                OBJECTIVE_FUNC.get(objective), alphas=alphas, delta=delta
            )
        else:
            self.fobj = partial(OBJECTIVE_FUNC.get(objective), alphas=alphas)
        self.feval = partial(self._funcs.get(TypeName.eval), alphas=alphas)
        self.dataset = self._funcs.get(TypeName.train_dtype)(
            data=self.x_train, label=self.y_train
        )

    def train(self, params: Dict[str, Any]) -> ModelLike:
        """
        Train regressor and return model
        Args:
            params (Dict[str, Any])

        Returns:
            ModelLike
        """
        self.__set_params(params=params)

        if self._model == ModelName.lightgbm:
            self._params.update({_MQStr._obj: self.fobj})
            self.model = lgb.train(
                train_set=self.dataset,
                params=self._params,
                feval=self.feval,
                valid_sets=[self.dataset],
            )
            self._train_score = self.model.best_score[_MQStr._trg][CHECK_LOSS]
        else:
            _evals_result = {}
            self.model = xgb.train(
                dtrain=self.dataset,
                verbose_eval=False,
                params=self._params,
                obj=self.fobj,
                custom_metric=self.feval,
                evals=[
                    (self.dataset, _MQStr._tr),
                ],
                evals_result=_evals_result,
            )
            self._train_score = min(_evals_result[_MQStr._tr][CHECK_LOSS])
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
        self.__is_fitted
        alphas = alpha_validate(alphas)
        _x = prepare_x(x, alphas)
        _x = self._funcs.get(TypeName.predict_dtype)(_x)
        _pred = self.model.predict(_x)
        _pred = _pred.reshape(len(alphas), len(x))
        return _pred

    def __set_params(self, params: Dict[str, Any]) -> None:
        """
        Set monotone constraints in params
        Args:
            params (Dict[str, Any])
        """
        if isinstance(params, dict) and _MQStr._obj in params:
            raise ValidationException(
                "The parameter named 'objective' must not be included in params"
            )
        self._params = params.copy()

        if _MQStr._mono in self._params:
            _monotone_constraints = list(self._params[_MQStr._mono])
            _monotone_constraints.append(1)
            self._params[_MQStr._mono] = self._funcs.get(TypeName.constraints_type)(
                _monotone_constraints
            )
        else:
            self._params.update(
                {
                    _MQStr._mono: self._funcs.get(TypeName.constraints_type)(
                        [1 if "_tau" == col else 0 for col in self.x_train.columns]
                    )
                }
            )

    @property
    def __is_fitted(self) -> None:
        if not getattr(self, "_fitted", False):
            raise FittingException("train must be executed before predict")

    @property
    def train_score(self) -> float:
        self.__is_fitted
        return float(self._train_score)
