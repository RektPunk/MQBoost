from __future__ import annotations
from collections.abc import Callable
from functools import partial
from typing import Any

import lightgbm as lgb
import numpy as np
import xgboost as xgb

_DtrainLike = lgb.basic.Dataset | xgb.DMatrix
CHECK_LOSS: str = "check_loss"


def _grad_rho(u: np.ndarray, alpha: float) -> np.ndarray:
    return (u < 0).astype(int) - alpha


def _rho(u: np.ndarray, alpha: float) -> np.ndarray:
    return -u * _grad_rho(u, alpha)


def _error_delta_compare(u: np.ndarray, delta: float):
    _abs_error = np.abs(u)
    return (_abs_error <= delta).astype(int), (_abs_error > delta).astype(int)


def _grad_huber(u: np.ndarray, alpha: float, delta: float) -> np.ndarray:
    _smaller_delta, _bigger_delta = _error_delta_compare(u, delta)
    _g = _grad_rho(u, alpha)
    _r = _rho(u, alpha)
    return _r * _smaller_delta + _g * _bigger_delta


def _train_pred_reshape(
    y_pred: np.ndarray,
    dtrain: _DtrainLike,
    len_alpha: int,
) -> tuple[np.ndarray, np.ndarray]:
    _y_train = dtrain.get_label()
    return _y_train.reshape(len_alpha, -1), y_pred.reshape(len_alpha, -1)


def _compute_grads_hess(
    y_pred: np.ndarray,
    dtrain: _DtrainLike,
    alphas: list[float],
    grad_fn: Callable[[np.ndarray, float, Any], np.ndarray],
    **kwargs: Any,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute gradients for given loss function
    Args:
        y_train (np.ndarray)
        y_pred (np.ndarray)
        alphas (list[float])
        grad_fn (Callable)
        **kwargs (Any): Additional arguments for grad_fn

    Returns:
        np.ndarray
    """
    _len_alpha = len(alphas)
    _y_train, _y_pred = _train_pred_reshape(y_pred, dtrain, _len_alpha)
    grads = []
    for alpha_inx in range(len(alphas)):
        _err_for_alpha = _y_train[alpha_inx] - _y_pred[alpha_inx]
        _grad = grad_fn(u=_err_for_alpha, alpha=alphas[alpha_inx], **kwargs)
        grads.append(_grad)
    return np.concatenate(grads), np.ones_like(y_pred)


check_loss_grad_hess: Callable = partial(_compute_grads_hess, grad_fn=_grad_rho)
huber_loss_grad_hess: Callable = partial(_compute_grads_hess, grad_fn=_grad_huber)


def xgb_eval(
    y_pred: np.ndarray,
    dtrain: _DtrainLike,
    alphas: list[float],
) -> float:
    """
    Compute gradients for given loss function
    Args:
        y_train (np.ndarray)
        y_pred (np.ndarray)
        alphas (list[float])
        grad_fn (Callable)
        **kwargs (Any): Additional arguments for grad_fn

    Returns:
        np.ndarray
    """
    _len_alpha = len(alphas)
    _y_train, _y_pred = _train_pred_reshape(y_pred, dtrain, _len_alpha)
    loss: float = 0.0
    for alpha_inx in range(len(alphas)):
        _err_for_alpha = _y_train[alpha_inx] - _y_pred[alpha_inx]
        _loss = _rho(u=_err_for_alpha, alpha=alphas[alpha_inx])
        loss += np.sum(_loss)
    return CHECK_LOSS, loss


def lgb_eval(
    y_pred: np.ndarray,
    dtrain: _DtrainLike,
    alphas: list[float],
) -> tuple[np.ndarray, np.ndarray]:
    loss_str, loss = xgb_eval(y_pred, dtrain, alphas)
    return loss_str, loss, False
