from functools import partial
from typing import Any, Callable

import lightgbm as lgb
import numpy as np
import numpy.typing as npt
import xgboost as xgb

from mqboost.base import ModelName, ObjectiveName
from mqboost.utils import delta_validate, epsilon_validate


# check loss
def _grad_rho(error: npt.NDArray, alpha: float) -> npt.NDArray:
    return (error < 0).astype(int) - alpha


def _rho(error: npt.NDArray, alpha: float) -> npt.NDArray:
    return -error * _grad_rho(error=error, alpha=alpha)


def _hess_rho(error: npt.NDArray, **kwargs) -> npt.NDArray:
    return np.ones_like(error)


# Huber loss
def _grad_huber(error: npt.NDArray, alpha: float, delta: float) -> npt.NDArray:
    _abs_error = np.abs(error)
    _smaller_delta = (_abs_error <= delta).astype(int)
    _bigger_delta = (_abs_error > delta).astype(int)
    _r = _rho(error=error, alpha=alpha)
    _grad = _grad_rho(error=error, alpha=alpha)
    return _r * _smaller_delta + _grad * _bigger_delta


def _hess_huber(error: npt.NDArray, **kwargs) -> npt.NDArray:
    return np.ones_like(error)


# Approx loss (MM loss)
def _grad_approx(error: npt.NDArray, alpha: float, epsilon: float) -> npt.NDArray:
    _grad = 0.5 * (1 - 2 * alpha - error / (epsilon + np.abs(error)))
    return _grad


def _hess_approx(error: npt.NDArray, epsilon: float, **kwargs) -> npt.NDArray:
    _hess = 1 / (2 * (epsilon + np.abs(error)))
    return _hess


def _train_pred_reshape(
    y_pred: npt.NDArray,
    dtrain: lgb.Dataset | xgb.DMatrix,
    len_alpha: int,
) -> tuple[npt.NDArray, npt.NDArray]:
    """Reshape training predictions and labels to match the number of quantile levels."""
    _y_train = dtrain.get_label()
    if not isinstance(_y_train, np.ndarray):
        _y_train = np.array(_y_train)
    return _y_train.reshape(len_alpha, -1), y_pred.reshape(len_alpha, -1)


# Compute gradient hessian logic
def compute_grad_hess(grad_fn: Callable, hess_fn: Callable) -> Callable:
    """Return computing gradient hessian function."""

    def _compute_grads_hess(
        y_pred: npt.NDArray,
        dtrain: lgb.Dataset | xgb.DMatrix,
        alphas: list[float],
        weight: npt.NDArray | None,
        **kwargs: Any,
    ) -> tuple[npt.NDArray, npt.NDArray]:
        _len_alpha = len(alphas)
        _y_train, _y_pred = _train_pred_reshape(
            y_pred=y_pred, dtrain=dtrain, len_alpha=_len_alpha
        )
        grads: list[np.ndarray] = []
        hess: list[np.ndarray] = []
        _len_y = len(_y_train[0])
        for alpha_inx in range(len(alphas)):
            _err_for_alpha: np.ndarray = _y_train[alpha_inx] - _y_pred[alpha_inx]
            _grad = grad_fn(error=_err_for_alpha, alpha=alphas[alpha_inx], **kwargs)
            _hess = hess_fn(error=_err_for_alpha, alpha=alphas[alpha_inx], **kwargs)
            grads.append(_grad / _len_y)
            hess.append(_hess / _len_y)

        if isinstance(weight, np.ndarray):
            return np.concatenate(grads) * weight, np.concatenate(hess) * weight
        else:
            return np.concatenate(grads), np.concatenate(hess)

    return _compute_grads_hess


# Gradient and Hessian functions
check_loss_grad_hess = compute_grad_hess(grad_fn=_grad_rho, hess_fn=_hess_rho)
huber_loss_grad_hess = compute_grad_hess(grad_fn=_grad_huber, hess_fn=_hess_huber)
approx_loss_grad_hess = compute_grad_hess(grad_fn=_grad_approx, hess_fn=_hess_approx)


def _eval_check_loss(
    y_pred: np.ndarray,
    dtrain: lgb.Dataset | xgb.DMatrix,
    alphas: list[float],
) -> float:
    """Evaluate the check loss function."""
    _len_alpha = len(alphas)
    _y_train, _y_pred = _train_pred_reshape(
        y_pred=y_pred, dtrain=dtrain, len_alpha=_len_alpha
    )
    loss: float = 0.0
    for alpha_inx in range(_len_alpha):
        _err_for_alpha = _y_train[alpha_inx] - _y_pred[alpha_inx]
        _loss = _rho(error=_err_for_alpha, alpha=alphas[alpha_inx])
        loss += float(np.mean(_loss))
    return loss


def _xgb_eval_loss(
    y_pred: np.ndarray,
    dtrain: lgb.Dataset | xgb.DMatrix,
    alphas: list[float],
) -> tuple[str, float]:
    loss = _eval_check_loss(y_pred=y_pred, dtrain=dtrain, alphas=alphas)
    return "check_loss", loss


def _lgb_eval_loss(
    y_pred: np.ndarray,
    dtrain: lgb.Dataset | xgb.DMatrix,
    alphas: list[float],
) -> tuple[str, float, bool]:
    loss = _eval_check_loss(y_pred=y_pred, dtrain=dtrain, alphas=alphas)
    return "check_loss", loss, False


def validate_parameters(objective: ObjectiveName, delta: float, epsilon: float) -> None:
    if objective == ObjectiveName.huber:
        delta_validate(delta=delta)
    elif objective == ObjectiveName.approx:
        epsilon_validate(epsilon=epsilon)


def get_fobj_function(
    objective: ObjectiveName,
    weight: np.ndarray | None,
    alphas: list[float],
    delta: float,
    epsilon: float,
) -> Callable:
    objective_mapping: dict[ObjectiveName, Callable] = {
        ObjectiveName.check: partial(
            check_loss_grad_hess, weight=weight, alphas=alphas
        ),
        ObjectiveName.huber: partial(
            huber_loss_grad_hess, weight=weight, alphas=alphas, delta=delta
        ),
        ObjectiveName.approx: partial(
            approx_loss_grad_hess, weight=weight, alphas=alphas, epsilon=epsilon
        ),
    }
    return objective_mapping[objective]


def get_feval_function(model: ModelName, alphas: list[float]) -> Callable:
    model_mapping: dict[ModelName, Callable] = {
        ModelName.lightgbm: partial(_lgb_eval_loss, alphas=alphas),
        ModelName.xgboost: partial(_xgb_eval_loss, alphas=alphas),
    }
    return model_mapping[model]


class MQObjective:
    """MQObjective provides a monotone quantile objective and evaluation function for models.

    Attributes:
        alphas (list[float]): List of quantile levels for the model.
        objective (ObjectiveName): The objective function type (either 'huber' or 'check').
        model (ModelName): The model type (either 'lightgbm' or 'xgboost').
        delta (float): The delta parameter used for the 'huber' loss.
        epsilon (float): The epsilon parameter used for the 'approx' loss.
        weight (np.ndarray): The weight for each instance (if provided).
    """

    def __init__(
        self,
        alphas: list[float],
        objective: ObjectiveName,
        model: ModelName,
        delta: float,
        epsilon: float,
        weight: np.ndarray | None,
    ) -> None:
        """Initialize the MQObjective."""
        validate_parameters(objective=objective, delta=delta, epsilon=epsilon)
        self.fobj = get_fobj_function(
            objective=objective,
            weight=weight,
            alphas=alphas,
            delta=delta,
            epsilon=epsilon,
        )
        self.feval = get_feval_function(model=model, alphas=alphas)
