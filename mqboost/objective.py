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
    return -u * _grad_rho(u=u, alpha=alpha)


def _error_delta_compare(u: np.ndarray, delta: float):
    _abs_error = np.abs(u)
    return (_abs_error <= delta).astype(int), (_abs_error > delta).astype(int)


def _grad_huber(u: np.ndarray, alpha: float, delta: float) -> np.ndarray:
    _smaller_delta, _bigger_delta = _error_delta_compare(u=u, delta=delta)
    _g = _grad_rho(u=u, alpha=alpha)
    _r = _rho(u=u, alpha=alpha)
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
    _y_train, _y_pred = _train_pred_reshape(
        y_pred=y_pred, dtrain=dtrain, len_alpha=_len_alpha
    )
    grads = []
    for alpha_inx in range(len(alphas)):
        _err_for_alpha = _y_train[alpha_inx] - _y_pred[alpha_inx]
        _grad = grad_fn(u=_err_for_alpha, alpha=alphas[alpha_inx], **kwargs)
        grads.append(_grad)
    return np.concatenate(grads), np.ones_like(y_pred)


check_loss_grad_hess: Callable = partial(_compute_grads_hess, grad_fn=_grad_rho)
huber_loss_grad_hess: Callable = partial(_compute_grads_hess, grad_fn=_grad_huber)


def _eval_check_loss(
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
    Returns:
        float
    """
    _len_alpha = len(alphas)
    _y_train, _y_pred = _train_pred_reshape(
        y_pred=y_pred, dtrain=dtrain, len_alpha=_len_alpha
    )
    loss: float = 0.0
    for alpha_inx in range(_len_alpha):
        _err_for_alpha = _y_train[alpha_inx] - _y_pred[alpha_inx]
        _loss = _rho(u=_err_for_alpha, alpha=alphas[alpha_inx])
        loss = loss + np.mean(_loss)

    loss = loss / _len_alpha
    return loss


def _xgb_eval_loss(
    y_pred: np.ndarray,
    dtrain: DtrainLike,
    alphas: List[float],
) -> Tuple[str, float]:
    loss = _eval_check_loss(y_pred=y_pred, dtrain=dtrain, alphas=alphas)
    return CHECK_LOSS, loss


def _lgb_eval_loss(
    y_pred: np.ndarray,
    dtrain: DtrainLike,
    alphas: List[float],
) -> Tuple[str, float, bool]:
    loss = _eval_check_loss(y_pred=y_pred, dtrain=dtrain, alphas=alphas)
    return CHECK_LOSS, loss, False


class MQObjective:
    """
    Monotone quantile objective and evaluation function
    Attributes
    ----------
    alphas (List[float])
    objective (ObjectiveName)
    model (ModelName)
    delta (float)

    Property
    ----
    fobj
    feval
    eval_name
    """

    def __init__(
        self,
        alphas: List[float],
        objective: ObjectiveName,
        model: ModelName,
        delta: float,
    ) -> None:
        if objective == ObjectiveName.huber:
            delta_validate(delta=delta)
            self._fobj = partial(huber_loss_grad_hess, alphas=alphas, delta=delta)
        elif objective == ObjectiveName.check:
            self._fobj = partial(check_loss_grad_hess, alphas=alphas)

        self._eval_name = CHECK_LOSS
        if model == ModelName.lightgbm:
            self._feval = partial(_lgb_eval_loss, alphas=alphas)
        elif model == ModelName.xgboost:
            self._feval = partial(_xgb_eval_loss, alphas=alphas)

    @property
    def fobj(self) -> Callable:
        return self._fobj

    @property
    def feval(self) -> Callable:
        return self._feval

    @property
    def eval_name(self) -> str:
        return self._eval_name
