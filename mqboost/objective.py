from typing import Any, Callable

import lightgbm as lgb
import numpy as np
import numpy.typing as npt
import xgboost as xgb

from mqboost.base import ModelName, ObjectiveName
from mqboost.utils import delta_validate, epsilon_validate


def calc_rho(error: npt.NDArray, alpha: float) -> npt.NDArray:
    """Compute rho for the given error and alpha."""
    return (alpha - (error < 0).astype(int)) * error


def calc_check_grad_hess(
    error: npt.NDArray, alpha: float
) -> tuple[npt.NDArray, npt.NDArray]:
    """Compute gradient and Hessian for the check loss."""
    return (error < 0).astype(int) - alpha, np.ones_like(error)


def calc_huber_grad_hess(
    error: npt.NDArray, alpha: float, delta: float
) -> tuple[npt.NDArray, npt.NDArray]:
    """Compute gradient and Hessian for the Huber loss."""
    abs_error = np.abs(error)
    smaller_delta = (abs_error <= delta).astype(int)
    bigger_delta = (abs_error > delta).astype(int)
    rho_val = calc_rho(error=error, alpha=alpha)
    check_grad, check_hess = calc_check_grad_hess(error=error, alpha=alpha)
    return rho_val * smaller_delta + check_grad * bigger_delta, check_hess


def calc_approx_grad_hess(
    error: npt.NDArray, alpha: float, epsilon: float
) -> tuple[npt.NDArray, npt.NDArray]:
    """Compute gradient and Hessian for the approximate loss (MM loss)."""
    approx_grad = 0.5 * (1 - 2 * alpha - error / (epsilon + np.abs(error)))
    approx_hess = 1 / (2 * (epsilon + np.abs(error)))
    return approx_grad, approx_hess


def train_pred_reshape(
    dtrain: lgb.Dataset | xgb.DMatrix,
    y_pred: npt.NDArray,
    len_alpha: int,
) -> tuple[npt.NDArray, npt.NDArray]:
    """Reshape training predictions and labels to match the number of quantile levels."""
    y_train = dtrain.get_label()
    if not isinstance(y_train, np.ndarray):
        y_train = np.array(y_train)
    return y_train.reshape(len_alpha, -1), y_pred.reshape(len_alpha, -1)


def compute_grad_hess_single_alpha(
    y_true: npt.NDArray,
    y_pred: npt.NDArray,
    alpha: float,
    calc_grad_hess_fn: Callable,
    n: int,
    **kwargs,
) -> tuple[npt.NDArray, npt.NDArray]:
    """Compute gradient and Hessian using the given function for a single alpha value."""
    error = y_true - y_pred
    grad, hess = calc_grad_hess_fn(error=error, alpha=alpha, **kwargs)
    return grad / n, hess / n


def compute_grad_hess(
    calc_grad_hess_fn: Callable,
) -> Callable[...,]:
    """Return a function that computes gradient and Hessian for a given calc_grad_hess_fn."""

    def _compute_grads_hess(
        y_pred: npt.NDArray,
        dtrain: lgb.Dataset | xgb.DMatrix,
        alphas: list[float],
        weight: npt.NDArray | None,
        **kwargs: Any,
    ) -> tuple[npt.NDArray, npt.NDArray]:
        len_alpha = len(alphas)
        y_train_reshaped, y_pred_reshaped = train_pred_reshape(
            y_pred=y_pred, dtrain=dtrain, len_alpha=len_alpha
        )

        grads: list[np.ndarray] = []
        hess: list[np.ndarray] = []
        len_y = len(y_train_reshaped[0])
        for alpha_inx in range(len(alphas)):
            _grad, _hess = compute_grad_hess_single_alpha(
                y_train_reshaped[alpha_inx],
                y_pred_reshaped[alpha_inx],
                alphas[alpha_inx],
                calc_grad_hess_fn,
                len_y,
                **kwargs,
            )
            grads.append(_grad)
            hess.append(_hess)

        if isinstance(weight, np.ndarray):
            return np.concatenate(grads) * weight, np.concatenate(hess) * weight
        else:
            return np.concatenate(grads), np.concatenate(hess)

    return _compute_grads_hess


# Gradient and Hessian functions
check_loss_grad_hess = compute_grad_hess(calc_grad_hess_fn=calc_check_grad_hess)
huber_loss_grad_hess = compute_grad_hess(calc_grad_hess_fn=calc_huber_grad_hess)
approx_loss_grad_hess = compute_grad_hess(calc_grad_hess_fn=calc_approx_grad_hess)


def eval_check_loss(
    y_pred: np.ndarray,
    dtrain: lgb.Dataset | xgb.DMatrix,
    alphas: list[float],
) -> float:
    """Evaluate the check loss function."""
    len_alpha = len(alphas)
    y_train_reshaped, y_pred_reshaped = train_pred_reshape(
        y_pred=y_pred, dtrain=dtrain, len_alpha=len_alpha
    )
    loss: float = 0.0
    for alpha_inx in range(len_alpha):
        _err_for_alpha = y_train_reshaped[alpha_inx] - y_pred_reshaped[alpha_inx]
        _loss = calc_rho(error=_err_for_alpha, alpha=alphas[alpha_inx])
        loss += float(np.mean(_loss))
    return loss


def build_fobj(
    alphas: list[float],
    objective: ObjectiveName,
    delta: float,
    epsilon: float,
    weight: np.ndarray | None,
) -> Callable[..., tuple[npt.NDArray, npt.NDArray]]:
    """Return fobj function."""
    if objective == ObjectiveName.approx:
        epsilon_validate(epsilon)

    if objective == ObjectiveName.huber:
        delta_validate(delta)

    def fobj(
        y_pred: npt.NDArray, dtrain: lgb.Dataset | xgb.DMatrix
    ) -> tuple[npt.NDArray, npt.NDArray]:
        if objective == ObjectiveName.check:
            return check_loss_grad_hess(
                y_pred=y_pred,
                dtrain=dtrain,
                alphas=alphas,
                weight=weight,
            )

        elif objective == ObjectiveName.huber:
            return huber_loss_grad_hess(
                y_pred=y_pred,
                dtrain=dtrain,
                alphas=alphas,
                weight=weight,
                delta=delta,
            )

        elif objective == ObjectiveName.approx:
            return approx_loss_grad_hess(
                y_pred=y_pred,
                dtrain=dtrain,
                alphas=alphas,
                weight=weight,
                epsilon=epsilon,
            )

    return fobj


def build_feval(
    model: ModelName, alphas: list[float]
) -> Callable[[npt.NDArray, lgb.Dataset | xgb.DMatrix], tuple]:
    """Return feval function."""

    def feval(y_pred: npt.NDArray, dtrain: lgb.Dataset | xgb.DMatrix) -> tuple:
        loss = eval_check_loss(y_pred, dtrain, alphas)
        if model == ModelName.lightgbm:
            return "check_loss", loss, False
        elif model == ModelName.xgboost:
            return "check_loss", loss

    return feval


class MQObjective:
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
        self.fobj = build_fobj(alphas, objective, delta, epsilon, weight)
        self.feval = build_feval(model, alphas)
