from functools import partial
from typing import Any, Callable, List, Optional, Tuple

import numpy as np

from mqboost.base import DtrainLike, ModelName, ObjectiveName
from mqboost.utils import delta_validate

CHECK_LOSS: str = "check_loss"


def _grad_rho(u: np.ndarray, alpha: float) -> np.ndarray:
    """
    Compute the gradient of the check loss function.
    Args:
        u (np.ndarray): The error term.
        alpha (float): The quantile level.
    Returns:
        np.ndarray: The gradient of the check loss function.
    """
    return (u < 0).astype(int) - alpha


def _hess_rho(u: np.ndarray, alpha: float, delta: Optional[float] = None) -> np.ndarray:
    """
    Compute the Hessian of the check and huber loss function.
    Args:
        u (np.ndarray): The error term.
        alpha (float): The quantile level.
    Returns:
        np.ndarray: The Hessian of the check and huber loss function, which is a constant array of ones.
    """
    _h = np.ones_like(u)
    return _h


def _rho(u: np.ndarray, alpha: float) -> np.ndarray:
    """
    Compute the check loss function.
    Args:
        u (np.ndarray): The error term.
        alpha (float): The quantile level.
    Returns:
        np.ndarray: The check loss.
    """
    return -u * _grad_rho(u=u, alpha=alpha)


def _grad_approx(u: np.ndarray, alpha: float, epsilon: float = 1e-5):
    """
    Compute the gradient of the approx of the smooth approximated check loss function.
    Args:
        u (np.ndarray): The error term.
        alpha (float): The quantile level.
        epsilon (float, optional): The perturbation imposing smoothness. Defaults to 1e-5.
    Returns:
        np.ndarray: The gradient of the approx of the smooth approximated check loss function.
    """
    _grad = 0.5 * (1 - 2 * alpha - u / (epsilon + np.abs(u)))
    return _grad


def _hess_approx(u: np.ndarray, alpha: float, epsilon: float = 1e-5):
    """
    Compute the Hessian of the approx of the smooth approximated check loss function.
    Args:
        u (np.ndarray): The error term.
        alpha (float): The quantile level.
        epsilon (float, optional): The perturbation imposing smoothness. Defaults to 1e-5.
    Returns:
        np.ndarray: The Hessian of the approx of the smooth approximated check loss function.
    """
    _hess = 1 / (2 * (epsilon + np.abs(u)))
    return _hess


def _error_delta_compare(u: np.ndarray, delta: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compare absolute errors with delta.
    Args:
        u (np.ndarray): The error term.
        delta (float): The delta parameter.
    Returns:
        tuple: Two boolean arrays indicating where the errors are smaller or larger than delta.
    """
    _abs_error = np.abs(u)
    return (_abs_error <= delta).astype(int), (_abs_error > delta).astype(int)


def _grad_huber(u: np.ndarray, alpha: float, delta: float) -> np.ndarray:
    """
    Compute the gradient of the huber loss function.
    Args:
        u (np.ndarray): The error term.
        alpha (float): The quantile level.
        delta (float): The delta parameter.
    Returns:
        np.ndarray: The gradient of the huber loss function.
    """
    _smaller_delta, _bigger_delta = _error_delta_compare(u=u, delta=delta)
    _grad = _grad_rho(u=u, alpha=alpha)
    _r = _rho(u=u, alpha=alpha)
    return _r * _smaller_delta + _grad * _bigger_delta


def _train_pred_reshape(
    y_pred: np.ndarray,
    dtrain: DtrainLike,
    len_alpha: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Reshape training predictions and labels to match the number of quantile levels.
    Args:
        y_pred (np.ndarray): The predicted values.
        dtrain (DtrainLike): The training data.
        len_alpha (int): The number of quantile levels.
    Returns:
        Tuple[np.ndarray, np.ndarray]: Reshaped training labels and predictions.
    """
    _y_train: np.ndarray = dtrain.get_label()
    return _y_train.reshape(len_alpha, -1), y_pred.reshape(len_alpha, -1)


def _compute_grads_hess(
    y_pred: np.ndarray,
    dtrain: DtrainLike,
    alphas: List[float],
    grad_fn: Callable[[np.ndarray, float, Any], np.ndarray],
    hess_fn: Callable[[np.ndarray, float, Any], np.ndarray],
    **kwargs: Any,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute gradients and hessians for the given loss function.
    Args:
        y_pred (np.ndarray): The predicted values.
        dtrain (DtrainLike): The training data.
        alphas (List[float]): List of quantile levels.
        grad_fn (Callable): The gradient function to be used.
        hess_fn (Callable): The Hessian function to be used.
        **kwargs (Any): Additional arguments for the gradient function.
    Returns:
        Tuple[np.ndarray, np.ndarray]: The computed gradients and hessians.
    """
    _len_alpha = len(alphas)
    _y_train, _y_pred = _train_pred_reshape(
        y_pred=y_pred, dtrain=dtrain, len_alpha=_len_alpha
    )
    grads = []
    hess = []
    for alpha_inx in range(len(alphas)):
        _err_for_alpha = _y_train[alpha_inx] - _y_pred[alpha_inx]
        _grad = grad_fn(u=_err_for_alpha, alpha=alphas[alpha_inx], **kwargs)
        _hess = hess_fn(u=_err_for_alpha, alpha=alphas[alpha_inx], **kwargs)
        grads.append(_grad)
        hess.append(_hess)

    return np.concatenate(grads), np.concatenate(hess)


check_loss_grad_hess: Callable = partial(
    _compute_grads_hess, grad_fn=_grad_rho, hess_fn=_hess_rho
)
huber_loss_grad_hess: Callable = partial(
    _compute_grads_hess, grad_fn=_grad_huber, hess_fn=_hess_rho
)
approx_loss_grad_hess: Callable = partial(
    _compute_grads_hess, grad_fn=_grad_approx, hess_fn=_hess_approx
)


def _eval_check_loss(
    y_pred: np.ndarray,
    dtrain: DtrainLike,
    alphas: List[float],
) -> float:
    """
    Evaluate the check loss function.
    Args:
        y_pred (np.ndarray): The predicted values.
        dtrain (DtrainLike): The training data.
        alphas (List[float]): List of quantile levels.
    Returns:
        float: The computed check loss.
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
    """
    Evaluation function for XGBoost.
    Args:
        y_pred (np.ndarray): The predicted values.
        dtrain (DtrainLike): The training data.
        alphas (List[float]): List of quantile levels.
    Returns:
        Tuple[str, float]: The evaluation metric name and the computed loss.
    """
    loss = _eval_check_loss(y_pred=y_pred, dtrain=dtrain, alphas=alphas)
    return CHECK_LOSS, loss


def _lgb_eval_loss(
    y_pred: np.ndarray,
    dtrain: DtrainLike,
    alphas: List[float],
) -> Tuple[str, float, bool]:
    """
    Evaluation function for LightGBM.
    Args:
        y_pred (np.ndarray): The predicted values.
        dtrain (DtrainLike): The training data.
        alphas (List[float]): List of quantile levels.
    Returns:
        Tuple[str, float, bool]: The evaluation metric name, the computed loss, and a boolean flag.
    """
    loss = _eval_check_loss(y_pred=y_pred, dtrain=dtrain, alphas=alphas)
    return CHECK_LOSS, loss, False


class MQObjective:
    """
    MQObjective provides a monotone quantile objective and evaluation function for models.

    Attributes:
        alphas (List[float]): List of quantile levels for the model.
        objective (ObjectiveName): The objective function type (either 'huber' or 'check').
        model (ModelName): The model type (either 'lightgbm' or 'xgboost').
        delta (float): The delta parameter used for the 'huber' loss.

    Properties:
        fobj (Callable): The objective function to be minimized.
        feval (Callable): The evaluation function used during training.
        eval_name (str): The name of the evaluation metric.
        delta (float): The delta parameter value.
    """

    def __init__(
        self,
        alphas: List[float],
        objective: ObjectiveName,
        model: ModelName,
        delta: float,
        epsilon: float,
    ) -> None:
        """Initialize the MQObjective."""
        if objective == ObjectiveName.huber:
            self._delta = delta_validate(delta=delta)
            self._fobj = partial(huber_loss_grad_hess, alphas=alphas, delta=self._delta)
        elif objective == ObjectiveName.check:
            self._fobj = partial(check_loss_grad_hess, alphas=alphas)
        elif objective == ObjectiveName.approx:
            self._fobj = partial(approx_loss_grad_hess, alphas=alphas, epsilon=epsilon)

        self._eval_name = CHECK_LOSS
        if model == ModelName.lightgbm:
            self._feval = partial(_lgb_eval_loss, alphas=alphas)
        elif model == ModelName.xgboost:
            self._feval = partial(_xgb_eval_loss, alphas=alphas)

    @property
    def fobj(self) -> Callable:
        """
        Get the objective function to be minimized.
        Returns:
            Callable: The objective function.
        """
        return self._fobj

    @property
    def feval(self) -> Callable:
        """
        Get the evaluation function used during training.
        Returns:
            Callable: The evaluation function.
        """
        return self._feval

    @property
    def eval_name(self) -> str:
        """
        Get the name of the evaluation metric.
        Returns:
            str: The evaluation metric name.
        """
        return self._eval_name

    @property
    def delta(self) -> float:
        """
        Get the delta parameter for the huber loss.
        Returns:
            float: The delta parameter value.
        """
        return self._delta
