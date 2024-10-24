from typing import Callable

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from optuna import Trial
from sklearn.model_selection import train_test_split

from mqboost.base import (
    DtrainLike,
    FittingException,
    ModelName,
    ObjectiveName,
    ParamsLike,
)
from mqboost.constraints import set_monotone_constraints
from mqboost.dataset import MQDataset
from mqboost.objective import MQObjective
from mqboost.utils import delta_validate, epsilon_validate, params_validate

__all__ = ["MQOptimizer"]


def _lgb_get_params(trial: Trial):
    return {
        "verbose": -1,
        "learning_rate": trial.suggest_float("learning_rate", 1e-2, 1.0),
        "max_depth": trial.suggest_int("max_depth", 1, 10),
        "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0),
        "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0),
        "num_leaves": trial.suggest_int("num_leaves", 2, 256),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.4, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.4, 1.0),
        "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
    }


def _xgb_get_params(trial: Trial):
    return {
        "learning_rate": trial.suggest_float("learning_rate", 1e-2, 1.0),
        "max_depth": trial.suggest_int("max_depth", 1, 10),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 100.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 100.0),
        "subsample": trial.suggest_float("subsample", 0.1, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.1, 1.0),
    }


_GET_PARAMS_FUNC = {
    ModelName.lightgbm: _lgb_get_params,
    ModelName.xgboost: _xgb_get_params,
}


def _train_valid_split(
    x_train: pd.DataFrame, y_train: np.ndarray
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    return train_test_split(
        x_train, y_train, test_size=0.2, random_state=42, stratify=x_train["_tau"]
    )


class MQOptimizer:
    """
    MQOptimizer is designed to optimize hyperparameters for MQRegressor with Optuna.

    Attributes:
        model (str): The model type (either 'lightgbm' or 'xgboost'). Default is 'lightgbm'.
        objective (str):
            The objective function for the quantile regression ('check', 'huber', or 'phuber'). Default is 'check'.
        delta (float): Delta parameter for the 'huber' objective function. Default is 0.01.
        epsilon (float): Epsilon parameter for the 'apptox' objective function. Default is 1e-5.

    Methods:
        optimize_params(dataset, n_trials, get_params_func, valid_set):
            Optimizes the hyperparameters for the specified dataset using Optuna.

    Property
        MQObj: Returns the MQObjective instance.
        study: Returns the Optuna study instance.
        best_params: Returns the best hyperparameters found by the optimization process.
    """

    def __init__(
        self,
        model: str = ModelName.lightgbm.value,
        objective: str = ObjectiveName.check.value,
        delta: float = 0.01,
        epsilon: float = 1e-5,
    ) -> None:
        """Initialize the MQOptimizer."""
        delta_validate(delta=delta)
        epsilon_validate(epsilon=epsilon)

        self._model = ModelName.get(model)
        self._objective = ObjectiveName.get(objective)
        self._delta = delta
        self._epsilon = epsilon
        self._get_params = _GET_PARAMS_FUNC.get(self._model)

    def optimize_params(
        self,
        dataset: MQDataset,
        n_trials: int,
        get_params_func: Callable[[Trial], ParamsLike] | None = None,
        valid_set: MQDataset | None = None,
    ) -> ParamsLike:
        """
        Optimize hyperparameters.
        Args:
            dataset (MQDataset): The dataset to be used for optimization.
            n_trials (int): The number of trials for the hyperparameter optimization.
            get_params_func (Callable, optional): A custom function to get the parameters for the model.
                For example,
                    def get_params(trial: Trial) -> dict[str, Any]:
                        return {
                            "learning_rate": trial.suggest_float("learning_rate", 1e-2, 1.0),
                            "max_depth": trial.suggest_int("max_depth", 1, 10),
                            "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0),
                            "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0),
                        }
            valid_set (MQDataset, optional): The validation dataset. Defaults to None.
        Returns:
            dict[str, Any]: The best hyperparameters found by the optimization process.
        """
        self._dataset = dataset
        self._label_mean = dataset.label_mean
        self._MQObj = MQObjective(
            alphas=dataset.alphas,
            objective=self._objective,
            model=self._model,
            delta=self._delta,
            epsilon=self._epsilon,
        )
        if valid_set is None:
            x_train, x_valid, y_train, y_valid = _train_valid_split(
                x_train=self._dataset.data, y_train=self._dataset.label
            )
            dtrain = self._dataset.train_dtype(data=x_train, label=y_train)
            dvalid = self._dataset.train_dtype(data=x_valid, label=y_valid)
            deval = self._dataset.predict_dtype(data=x_valid)
        else:
            dtrain = self._dataset.dtrain
            dvalid = valid_set.dtrain
            deval = valid_set.dpredict

        if get_params_func is None:
            get_params_func = _GET_PARAMS_FUNC.get(self._model)

        def _study_func(trial: optuna.Trial) -> float:
            return self.__optuna_objective(
                trial=trial,
                dtrain=dtrain,
                dvalid=dvalid,
                deval=deval,
                get_params_func=get_params_func,
            )

        self._study = optuna.create_study(
            study_name=f"MQBoost_{self._model}",
            direction="minimize",
            load_if_exists=True,
        )
        self._study.optimize(_study_func, n_trials=n_trials)
        self._is_optimized = True
        return self._study.best_params

    def __optuna_objective(
        self,
        trial: optuna.Trial,
        dtrain: DtrainLike,
        dvalid: DtrainLike,
        deval: DtrainLike | pd.DataFrame,
        get_params_func: Callable[[Trial], ParamsLike],
    ) -> float:
        """Objective function for Optuna to minimize."""
        params = get_params_func(trial=trial)
        params_validate(params=params)
        params = set_monotone_constraints(
            params=params,
            columns=self._dataset.columns,
            model_name=self._model,
        )
        if self.__is_lgb:
            model_params = dict(
                params=params,
                train_set=dtrain,
                valid_sets=dvalid,
            )
            _gbm = lgb.train(**model_params)
            _preds = (
                _gbm.predict(data=deval, num_iteration=_gbm.best_iteration)
                + self._label_mean
            )
            _, loss, _ = self._MQObj.feval(y_pred=_preds, dtrain=dvalid)
        elif self.__is_xgb:
            model_params = dict(
                params=params,
                dtrain=dtrain,
                evals=[(dvalid, "valid")],
                num_boost_round=100,
            )
            _gbm = xgb.train(**model_params)
            _preds = _gbm.predict(data=deval) + self._label_mean
            _, loss = self._MQObj.feval(y_pred=_preds, dtrain=dvalid)
        else:
            raise FittingException("Model name is invalid")
        return loss

    @property
    def MQObj(self) -> MQObjective:
        """Get the MQObjective instance."""
        return self._MQObj

    @property
    def study(self) -> optuna.Study:
        """Get the Optuna study instance."""
        return getattr(self, "_study", None)

    @property
    def best_params(self) -> ParamsLike:
        """Get the best hyperparameters found by the optimization process."""
        self.__is_optimized()
        return {
            "params": self._study.best_params,
            "model": self._model.value,
            "objective": self._objective.value,
            "delta": self._delta,
        }

    @property
    def __is_lgb(self) -> bool:
        """Check if the model is LightGBM."""
        return self._model == ModelName.lightgbm

    @property
    def __is_xgb(self) -> bool:
        """Check if the model is XGBoost."""
        return self._model == ModelName.xgboost

    def __is_optimized(self) -> None:
        """Check if the optimization process has been completed."""
        if not getattr(self, "_is_optimized", False):
            raise FittingException("Optimization is not completed.")
