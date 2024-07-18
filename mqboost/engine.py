from typing import Any, Callable, Dict, Optional

import lightgbm as lgb
import numpy as np
import optuna
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
    XdataLike,
    YdataLike,
)
from mqboost.constraints import set_monotone_constraints
from mqboost.hpo import get_params, train_valid_split
from mqboost.objective import MQObjective, eval_loss
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
    delta: float

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
            delta (float)
        """
        self._alphas = alpha_validate(alphas)
        self._model = ModelName().get(model)
        self._objective = ObjectiveName().get(objective)
        _funcs = FUNC_TYPE.get(model)
        self._train_dtype: Callable = _funcs.get(TypeName.train_dtype)
        self._predict_dtype: Callable = _funcs.get(TypeName.predict_dtype)
        self._constraints_type: Callable = _funcs.get(TypeName.constraints_type)
        self._MQObjective = MQObjective(
            self._alphas, self._objective, self._model, delta
        )
        self.x_train, self.y_train = prepare_train(x, y, self._alphas)
        self.dataset = self._train_dtype(data=self.x_train, label=self.y_train)

    def train(
        self,
        params: Optional[Dict[str, Any]] = None,
        n_trials: int = 20,
    ) -> ModelLike:
        """
        Train regressor and return model
        Args:
            params (Optional[Dict[str, Any]])
                train parameter, default = None
                if None, hyperparameter tuning executed
            n_trials (int):
                the number of hyperparameter tuning, defalut = 3
        Returns:
            ModelLike
        """
        if params is None:
            params = self.hpo(n_trials=n_trials)

        self._params = set_monotone_constraints(
            params=params,
            x_train=self.x_train,
            constraints_fucs=self._constraints_type,
        )

        if self.__is_lgb:
            self._params.update({MQStr.obj: self._MQObjective.fobj})
            self.model = lgb.train(
                train_set=self.dataset,
                params=self._params,
                feval=self._MQObjective.feval,
                valid_sets=[self.dataset],
            )
        elif self.__is_xgb:
            self.model = xgb.train(
                dtrain=self.dataset,
                verbose_eval=False,
                params=self._params,
                obj=self._MQObjective.fobj,
                custom_metric=self._MQObjective.feval,
                # evals=[
                #     (self.dataset, "train"),
                # ],
            )
        else:
            raise FittingException("model name is invalid")
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
        _x = self._predict_dtype(_x)
        _pred = self.model.predict(_x)
        _pred = _pred.reshape(len(alphas), len(x))
        return _pred

    def hpo(self, n_trials, get_params_func=get_params) -> Dict[str, Any]:
        x_train, x_valid, y_train, y_valid = train_valid_split(
            self.x_train, self.y_train
        )

        def study_func_func(trial):
            return self.__optuna_objective(
                trial,
                x_train,
                y_train,
                x_valid,
                y_valid,
                self._constraints_type,
                get_params_func,
            )

        study = optuna.create_study(
            study_name=f"MQBoost_{self._model}",
            direction="minimize",
            load_if_exists=True,
        )
        study.optimize(study_func_func, n_trials=n_trials)
        return study.best_params

    def __optuna_objective(
        self,
        trial: optuna.Trial,
        x_train: XdataLike,
        y_train: YdataLike,
        x_valid: XdataLike,
        y_valid: YdataLike,
        constraints_func: Callable,
        get_params_func: Callable,
    ) -> float:
        params = get_params_func(trial, self._model)
        params = set_monotone_constraints(
            params, x_train=x_train, constraints_fucs=constraints_func
        )
        dvalid = self._train_dtype(x_valid, label=y_valid)
        if self.__is_lgb:
            model_params = dict(
                params=params,
                train_set=self._train_dtype(x_train, label=y_train),
                valid_sets=self._train_dtype(x_valid, label=y_valid),
            )
            _gbm = lgb.train(**model_params)
            _preds = _gbm.predict(x_valid, num_iteration=_gbm.best_iteration)
            _, loss = eval_loss(_preds, dvalid, self._alphas)
        elif self.__is_xgb:
            model_params = dict(
                params=params,
                dtrain=self._train_dtype(x_train, label=y_train),
                early_stopping_rounds=19,
                evals=[
                    (self._train_dtype(x_valid, label=y_valid), MQStr.valid),
                ],
            )
            _gbm = xgb.train(**model_params)
            _preds = _gbm.predict(self._predict_dtype(x_valid))
            _, loss = eval_loss(_preds, dvalid, self._alphas)
        else:
            raise FittingException("model name is invalid")
        return loss

    @property
    def __is_lgb(self) -> bool:
        return self._model == ModelName.lightgbm

    @property
    def __is_xgb(self) -> bool:
        return self._model == ModelName.xgboost

    @property
    def __is_fitted(self) -> None:
        if not getattr(self, "_fitted", False):
            raise FittingException("train must be executed before predict")
