from typing import List, Tuple
import numpy as np
import lightgbm as lgb


def _rho(u, alpha) -> np.ndarray:
    return u * _grad_rho(u, alpha)


def _grad_rho(u, alpha) -> np.ndarray:
    return -(alpha - (u < 0).astype(float))


def check_loss_grad_hess(
    y_pred: np.ndarray, dtrain: lgb.basic.Dataset, alphas: List[float]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return gradient and hessin of composite quanitle loss
    Args:
        y_pred (np.ndarray)
        dtrain (lgb.basic.Dataset)
        alphas (List[float])

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            gradient
            hessian
    """
    _len_alpha = len(alphas)
    _y_train = dtrain.get_label()
    _y_pred_reshaped = y_pred.reshape(_len_alpha, -1)
    _y_train_reshaped = _y_train.reshape(_len_alpha, -1)

    grads = []
    for alpha_inx in range(_len_alpha):
        _err_for_alpha = _y_train_reshaped[alpha_inx] - _y_pred_reshaped[alpha_inx]
        grad = _grad_rho(_err_for_alpha, alphas[alpha_inx])
        grads.append(grad)

    grad = np.concatenate(grads)
    hess = np.ones(_y_train.shape)

    return grad, hess


def check_loss_eval(
    y_pred: np.ndarray, dtrain: lgb.basic.Dataset, alphas: List[float]
) -> Tuple[str, np.ndarray, bool]:
    """
    Return composite quantile loss
    Args:
        y_pred (np.ndarray) 
        dtrain (lgb.basic.Dataset) 
        alphas (List[float]) 

    Returns:
        Tuple[str, np.ndarray, bool]
    """
    _len_alpha = len(alphas)
    _y_train = dtrain.get_label()
    _y_pred_reshaped = y_pred.reshape(_len_alpha, -1)
    _y_train_reshaped = _y_train.reshape(_len_alpha, -1)

    loss = []
    for alpha_inx in range(_len_alpha):
        _err_for_alpha = _y_train_reshaped[alpha_inx] - _y_pred_reshaped[alpha_inx]
        loss.append(_rho(_err_for_alpha, alphas[alpha_inx]))
    loss = np.concatenate(loss)
    return "loss", loss.mean(), False
