from typing import Any, Dict
from functools import partial

import numpy as np

from .base import (
    ModelName,
    ObjectiveName,
    TRAIN_DATASET_FUNC,
    MONOTONE_CONSTRAINTS_TYPE,
    PREDICT_DATASET_FUNC,
    OBJECTIVE_FUNC,
    XdataLike,
    YdataLike,
    AlphaLike,
)
from .utils import alpha_validate, prepare_train, prepare_x, delta_validate


__all__ = ["MonotoneQuantileRegressor"]


class MonotoneQuantileRegressor:
    """
    Monotone quantile regressor which preserving monotonicity among quantiles
    Attributes
    ----------
    x: XdataLike
    y: YdataLike
    alphas: AlphaLike
    objective: ObjectiveName
    _model_name: ModelName

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
        objective: ObjectiveName,
        _model_name: ModelName,
        **kwargs,
    ):
        """
        Set objevtive, dataset

        Args:
            x (XdataLike)
            y (YdataLike)
            alphas (AlphaLike)
            objective (ObjectiveName)
            _model_name (ModelName)
        """
        alphas = alpha_validate(alphas)
        self._model_name = _model_name
        self._objective = objective
        self.x_train, self.y_train = prepare_train(x, y, alphas)
        if ObjectiveName().get(objective) == ObjectiveName.huber:
            delta = kwargs.get("delta", 0.05)
            delta_validate(delta)
            self.fobj = partial(
                OBJECTIVE_FUNC.get(objective), alphas=alphas, delta=delta
            )
        else:
            self.fobj = partial(OBJECTIVE_FUNC.get(objective), alphas=alphas)
        # self.feval = partial(check_loss_eval, alphas=alphas)
        self.dataset = TRAIN_DATASET_FUNC.get(self._model_name)(
            data=self.x_train, label=self.y_train
        )

    def train(self, params: Dict[str, Any]) -> None:
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
        alphas = alpha_validate(alphas)
        _x = prepare_x(x, alphas)
        _x = PREDICT_DATASET_FUNC.get(self._model_name)(_x)
        _pred = self.model.predict(_x)
        _pred = _pred.reshape(len(alphas), len(x))
        return _pred
