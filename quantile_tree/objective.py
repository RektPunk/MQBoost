from typing import List, Tuple, Union
import numpy as np

import lightgbm as lgb
import xgboost as xgb


_DtrainLike = Union[lgb.basic.Dataset, xgb.DMatrix]


def _grad_rho(u: np.ndarray, alpha: float) -> np.ndarray:
    return (u < 0).astype(int) - alpha


def _rho(u: np.ndarray, alpha: float) -> np.ndarray:
    return u * _grad_rho(u, alpha)


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
) -> Tuple[np.ndarray, np.ndarray]:
    _y_train = dtrain.get_label()
    return _y_train.reshape(len_alpha, -1), y_pred.reshape(len_alpha, -1)


def check_loss_grad_hess(
    y_pred: np.ndarray,
    dtrain: _DtrainLike,
    alphas: List[float],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return gradient and hessin of composite check quanitle loss
    Args:
        dtrain (_DtrainLike)
        y_pred (np.ndarray)
        alphas (List[float])

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            gradient
            hessian
    """
    _len_alpha = len(alphas)
    _y_train, _y_pred = _train_pred_reshape(y_pred, dtrain, _len_alpha)
    grads = []
    for alpha_inx in range(_len_alpha):
        _err_for_alpha = _y_train[alpha_inx] - _y_pred[alpha_inx]
        _grad = _grad_rho(_err_for_alpha, alphas[alpha_inx])
        grads.append(_grad)

    grad = np.concatenate(grads)
    hess = np.ones(y_pred.shape)

    return grad, hess


def huber_loss_grad_hess(
    y_pred: np.ndarray,
    dtrain: _DtrainLike,
    alphas: List[float],
    delta: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return gradient and hessin of composite huber quanitle loss
    Args:
        y_pred (np.ndarray)
        dtrain (_DtrainLike)
        alphas (List[float])
        delta (float)

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            gradient
            hessian
    """
    _len_alpha = len(alphas)
    _y_train, _y_pred = _train_pred_reshape(y_pred, dtrain, _len_alpha)

    grads = []
    for alpha_inx in range(_len_alpha):
        _err_for_alpha = _y_train[alpha_inx] - _y_pred[alpha_inx]
        _grad = _grad_huber(_err_for_alpha, alphas[alpha_inx], delta)
        grads.append(_grad)

    grad = np.concatenate(grads)
    hess = np.ones(y_pred.shape)

    return grad, hess


# def check_loss_eval(
#     y_pred: np.ndarray,dtrain: _DtrainLike,  alphas: List[float],
# ) -> Tuple[str, np.ndarray, bool]:
#     """
#     Return composite quantile loss
#     Args:
#         dtrain (_DtrainLike)
#         y_pred (np.ndarray)
#         alphas (List[float])

#     Returns:
#         Tuple[str, np.ndarray, bool]
#     """
#     _len_alpha = len(alphas)
#     _y_train, _y_pred = _train_pred_reshape( y_pred, dtrain,_len_alpha)
#     loss = []
#     for alpha_inx in range(_len_alpha):
#         _err_for_alpha = _y_train[alpha_inx] - _y_pred[alpha_inx]
#         loss.append(_rho(_err_for_alpha, alphas[alpha_inx]))
#     loss = np.concatenate(loss)
#     return "loss", loss.mean(), False
