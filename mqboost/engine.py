from typing import Any, Dict, Optional

import lightgbm as lgb
import numpy as np
import xgboost as xgb

from mqboost.base import (
    FUNC_TYPE,
    AlphaLike,
    FittingException,
    ModelLike,
    ModelName,
    MQStr,
    ObjectiveName,
    TypeName,
    ValidationException,
    XdataLike,
    YdataLike,
)
from mqboost.objective import MQObjective
from mqboost.utils import alpha_validate, prepare_train, prepare_x

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
        delta: float = 0.05,
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
        self._MQObjective = MQObjective(alphas, self._objective, self._model, delta)
        self.x_train, self.y_train = prepare_train(x, y, alphas)
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
        self._params = self.__set_monotone_constraints(params=params)

        if self._model == ModelName.lightgbm:
            self._params.update({MQStr.obj: self._MQObjective.fobj})
            self.model = lgb.train(
                train_set=self.dataset,
                params=self._params,
                feval=self._MQObjective.feval,
                valid_sets=[self.dataset],
            )
            self._train_score = self.model.best_score[MQStr.trg][
                self._MQObjective.eval_name
            ]
        else:
            _evals_result = {}
            self.model = xgb.train(
                dtrain=self.dataset,
                verbose_eval=False,
                params=self._params,
                obj=self._MQObjective.fobj,
                custom_metric=self._MQObjective.feval,
                evals=[
                    (self.dataset, MQStr.tr),
                ],
                evals_result=_evals_result,
            )
            self._train_score = min(
                _evals_result[MQStr.tr][self._MQObjective.eval_name]
            )
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

    def __set_monotone_constraints(
        self, params: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Set monotone constraints in params
        Args:
            params (Dict[str, Any])
        """
        if MQStr.obj in params:
            raise ValidationException(
                "The parameter named 'objective' must not be included in params"
            )
        _params = params.copy() if params is not None else dict()
        if MQStr.mono in _params:
            _monotone_constraints = list(_params[MQStr.mono])
            _monotone_constraints.append(1)
            _params[MQStr.mono] = self._funcs.get(TypeName.constraints_type)(
                _monotone_constraints
            )
        else:
            _params.update(
                {
                    MQStr.mono: self._funcs.get(TypeName.constraints_type)(
                        [1 if "_tau" == col else 0 for col in self.x_train.columns]
                    )
                }
            )
        return _params

    @property
    def __is_fitted(self) -> None:
        if not getattr(self, "_fitted", False):
            raise FittingException("train must be executed before predict")

    @property
    def train_score(self) -> float:
        self.__is_fitted
        return float(self._train_score)
