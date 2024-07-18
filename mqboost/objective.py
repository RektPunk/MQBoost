from collections.abc import Callable
from functools import partial
from typing import Any, List, Tuple

import numpy as np

from mqboost.base import DtrainLike, ModelName, ObjectiveName
from mqboost.utils import delta_validate

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
    dtrain: DtrainLike,
    len_alpha: int,
) -> Tuple[np.ndarray, np.ndarray]:
    _y_train: np.ndarray = dtrain.get_label()
    return _y_train.reshape(len_alpha, -1), y_pred.reshape(len_alpha, -1)


def _compute_grads_hess(
    y_pred: np.ndarray,
    dtrain: DtrainLike,
    alphas: List[float],
    grad_fn: Callable[[np.ndarray, float, Any], np.ndarray],
    **kwargs: Any,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute gradients for given loss function
    Args:
        y_train (np.ndarray)
        y_pred (np.ndarray)
        alphas (List[float])
        grad_fn (Callable[[np.ndarray, float, Any], np.ndarray])
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


def _eval(
    y_pred: np.ndarray,
    dtrain: DtrainLike,
    alphas: List[float],
) -> float:
    """
    eval funcs
    Args:
        y_pred (np.ndarray)
        d_train (DtrainLike)
        alphas (List[float])
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
    dtrain: DtrainLike,
    alphas: List[float],
) -> Tuple[np.ndarray, np.ndarray]:
    loss_str, loss = _eval(y_pred, dtrain, alphas)
    return loss_str, loss, False


class MQObjective:
    def __init__(
        self,
        alphas: List[float],
        objective: ObjectiveName,
        model: ModelName,
        delta: float,
    ) -> None:
        if objective == ObjectiveName.huber:
            delta_validate(delta)
            self._fobj = partial(huber_loss_grad_hess, alphas=alphas, delta=delta)
        else:
            self._fobj = partial(check_loss_grad_hess, alphas=alphas)

        self._eval_name = CHECK_LOSS
        if model == ModelName.lightgbm:
            self._feval = partial(lgb_eval, alphas=alphas)
        else:
            self._feval = partial(_eval, alphas=alphas)

    @property
    def fobj(self) -> Callable:
        return self._fobj

    @property
    def feval(self) -> Callable:
        return self._feval

    @property
    def eval_name(self) -> str:
        return self._eval_name
