from typing import Any, Callable, Dict, List, Optional

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
import xgboost as xgb

from mqboost.base import (
    FUNC_TYPE,
    AlphaLike,
    FittingException,
    ModelName,
    MQStr,
    ObjectiveName,
    TypeName,
    ValidationException,
    XdataLike,
    YdataLike,
)
from mqboost.constraints import set_monotone_constraints
from mqboost.hpo import get_params, train_valid_split
from mqboost.objective import MQObjective
from mqboost.utils import alpha_validate, prepare_train, prepare_x

__all__ = ["MQRegressor"]


class MQRegressor:
    """
    Monotone quantile regressor which preserving monotonicity among quantiles
    Attributes
    ----------
    x: XdataLike
    y: YdataLike
    alphas: AlphaLike
        It must be in ascending order and not contain duplicates.
    objective: str
        Determine objective function. Defaults to "check", another option is "huber".
    model: str
        Determine base model. Defaults to "lightgbm", another option is "xgboost".
    delta (float, optional).
        Only used with "huber" objective.
        Defaults to 0.05 and must be smaller than 0.1.

    Methods
    ----------
    train
    predict
    optimize_params
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
        self._alphas = alpha_validate(alphas)
        self._model = ModelName().get(model)
        self._objective = ObjectiveName().get(objective)
        _funcs = FUNC_TYPE.get(model)
        self._train_dtype: Callable = _funcs.get(TypeName.train_dtype)
        self._predict_dtype: Callable = _funcs.get(TypeName.predict_dtype)
        self._constraints_type: Callable = _funcs.get(TypeName.constraints_type)
        self._MQObj = MQObjective(
            alphas=self._alphas,
            objective=self._objective,
            model=self._model,
            delta=delta,
        )
        self.x_train, self.y_train = prepare_train(x=x, y=y, alphas=self._alphas)
        self.dataset = self._train_dtype(data=self.x_train, label=self.y_train)

    def train(
        self,
        params: Optional[Dict[str, Any]] = None,
        n_trials: int = 20,
    ) -> None:
        """
        Train regressor and return model
        Args:
            params (Optional[Dict[str, Any]])
                Train parameter. Default to None.
                If None, hyperparameter optimization process is executed.
            n_trials (int):
                The number of hyperparameter tuning. Default to 20.
        """
        if params is None:
            params = self.optimize_params(n_trials=n_trials)

        self._params = set_monotone_constraints(
            params=params,
            x_train=self.x_train,
            constraints_fucs=self._constraints_type,
        )

        if self.__is_lgb:
            self._params.update({MQStr.obj: self._MQObj.fobj})
            self.model = lgb.train(
                train_set=self.dataset,
                params=self._params,
                feval=self._MQObj.feval,
                valid_sets=[self.dataset],
            )
        elif self.__is_xgb:
            self.model = xgb.train(
                dtrain=self.dataset,
                verbose_eval=False,
                params=self._params,
                obj=self._MQObj.fobj,
                custom_metric=self._MQObj.feval,
                evals=[(self.dataset, "train")],
            )
        else:
            raise FittingException("model name is invalid")
        self._fitted = True

    def predict(
        self,
        x: XdataLike,
        alphas: Optional[AlphaLike] = None,
    ) -> np.ndarray:
        """
        Return predicted quantiles
        Args:
            x (XdataLike)
            alphas (Optional[AlphaLike], optional). Defaults to None.

        Returns:
            np.ndarray
        """
        self.__is_fitted
        if alphas is None:
            alphas = self._alphas
        else:
            alphas = alpha_validate(alphas=alphas)
        _x = prepare_x(x=x, alphas=alphas)
        _x = self._predict_dtype(_x)
        _pred = self.model.predict(data=_x)
        _pred = _pred.reshape(len(alphas), len(x))
        return _pred

    def optimize_params(
        self,
        n_trials: int,
        get_params_func: Callable = get_params,
        valid_dict: Optional[Dict[str, XdataLike]] = None,
    ) -> Dict[str, Any]:
        """
        Optimize hyperparameter
        Args:
            n_trials (int): The number of trials for the hyperparameter optimization.
            get_params_func (Callable, optional):
                A function to get the parameters for the model.
                For example,
                    def get_params(trial: Trial, model: ModelName):
                        return {
                            "learning_rate": trial.suggest_float("learning_rate", 1e-2, 1.0, log=True),
                            "max_depth": trial.suggest_int("max_depth", 1, 10),
                            "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
                            "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
                            "num_leaves": trial.suggest_int("num_leaves", 2, 256),
                            "feature_fraction": trial.suggest_float("feature_fraction", 0.4, 1.0),
                            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.4, 1.0),
                            "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
                        }
            valid_dict (Dict):
                Manually selected validation set. Keys must contain "data" and "label".
                For example,
                    valid_dict = {
                        "data": ...,
                        "label": ...,
                    }
        Returns:
            Dict[str, Any]: best params
        """
        if valid_dict is None:
            x_train, x_valid, y_train, y_valid = train_valid_split(
                x_train=self.x_train, y_train=self.y_train
            )
        else:
            _valid_dict_keys: List[str] = ["data", "label"]
            if any([_ not in valid_dict.keys() for _ in _valid_dict_keys]):
                raise ValidationException(
                    "Key of valid_dict must contains 'data' and 'label'"
                )
            x_train = self.x_train
            y_train = self.y_train
            _x_valid = valid_dict.get("data")
            _y_valid = valid_dict.get("label")
            x_valid, y_valid = prepare_train(
                x=_x_valid, y=_y_valid, alphas=self._alphas
            )

        def _study_func(trial: optuna.Trial) -> float:
            return self.__optuna_objective(
                trial=trial,
                x_train=x_train,
                x_valid=x_valid,
                y_train=y_train,
                y_valid=y_valid,
                constraints_func=self._constraints_type,
                get_params_func=get_params_func,
            )

        self._study = optuna.create_study(
            study_name=f"MQBoost_{self._model}",
            direction="minimize",
            load_if_exists=True,
        )
        self._study.optimize(_study_func, n_trials=n_trials)
        return self._study.best_params

    def __optuna_objective(
        self,
        trial: optuna.Trial,
        x_train: pd.DataFrame,
        x_valid: pd.DataFrame,
        y_train: np.ndarray,
        y_valid: np.ndarray,
        constraints_func: Callable,
        get_params_func: Callable,
    ) -> float:
        """
        objective function for optuna
        Args:
            trial (optuna.Trial)
            x_train (pd.DataFrame)
            x_valid (pd.DataFrame)
            y_train (np.ndarray)
            y_valid (np.ndarray)
            constraints_func (Callable)
            get_params_func (Callable)
        Returns:
            float
        """
        params = get_params_func(trial, self._model)
        params = set_monotone_constraints(
            params, x_train=x_train, constraints_fucs=constraints_func
        )
        dtrain = self._train_dtype(data=x_train, label=y_train)
        dvalid = self._train_dtype(data=x_valid, label=y_valid)
        if self.__is_lgb:
            model_params = dict(
                params=params,
                train_set=dtrain,
                valid_sets=dvalid,
            )
            _gbm = lgb.train(**model_params)
            _preds = _gbm.predict(data=x_valid, num_iteration=_gbm.best_iteration)
            _, loss, _ = self._MQObj.feval(y_pred=_preds, dtrain=dvalid)
        elif self.__is_xgb:
            model_params = dict(
                params=params,
                dtrain=dtrain,
                evals=[(dvalid, MQStr.valid)],
            )
            _gbm = xgb.train(**model_params)
            _preds = _gbm.predict(data=self._predict_dtype(x_valid))
            _, loss = self._MQObj.feval(y_pred=_preds, dtrain=dvalid)
        else:
            raise FittingException("model name is invalid")
        return loss

    @property
    def MQObj(self) -> MQObjective:
        return self._MQObj

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

    @property
    def study(self) -> optuna.Study:
        return getattr(self, "_study", None)
