from typing import Tuple

import numpy as np
import pandas as pd
from optuna import Trial
from sklearn.model_selection import train_test_split

from mqboost.base import FittingException, ModelName


def get_params(trial: Trial, model: ModelName):
    if model == ModelName.lightgbm:
        params = {
            "verbose": -1,
            "learning_rate": trial.suggest_float("learning_rate", 1e-2, 1.0, log=True),
            "max_depth": trial.suggest_int("max_depth", 1, 10),
            "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
            "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 2, 256),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.4, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.4, 1.0),
            "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
        }
    elif model == ModelName.xgboost:
        params = {
            "learning_rate": trial.suggest_float("learning_rate", 1e-2, 1.0, log=True),
            "max_depth": trial.suggest_int("max_depth", 1, 10),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 100.0, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 100.0, log=True),
            "subsample": trial.suggest_float("subsample", 0.1, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.1, 1.0),
        }
    else:
        raise FittingException("model name is invalid")
    return params


def train_valid_split(
    x_train: pd.DataFrame, y_train: np.ndarray
) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    x_train, x_valid, y_train, y_valid = train_test_split(
        x_train, y_train, test_size=0.2, random_state=42, stratify=x_train["_tau"]
    )
    return x_train, x_valid, y_train, y_valid
