import lightgbm as lgb
import numpy as np
import numpy.typing as npt
import xgboost as xgb

from mqboost.base import ModelName, ObjectiveName, ValidationException


def calc_rho(error: npt.NDArray, alpha: npt.NDArray | float) -> npt.NDArray:
    """Compute the pinball loss (check loss) for a given error and quantile level alpha.

    The pinball loss is defined as: L(error, alpha) = (alpha - I(error < 0)) * error."""
    return (alpha - (error < 0).astype(int)) * error


def calc_check_grad_hess(
    error: npt.NDArray, alpha: npt.NDArray | float
) -> tuple[npt.NDArray, npt.NDArray]:
    """Compute the gradient and Hessian for the standard check loss.

    The gradient is dL/dp = I(error < 0) - alpha.
    A constant proxy of 1.0 is used for the Hessian to facilitate optimization."""
    return (error < 0).astype(int) - alpha, np.ones_like(error)


def calc_huber_grad_hess(
    error: npt.NDArray, alpha: npt.NDArray | float, epsilon: float
) -> tuple[npt.NDArray, npt.NDArray]:
    """Compute the gradient and Hessian for the Huber-like Smooth Quantile Loss.

    This objective provides a smooth approximation to the check loss near zero, controlled by the epsilon parameter.
    It behaves quadratically for |error| <= epsilon and linearly for |error| > epsilon."""
    abs_error = np.abs(error)
    mask = (abs_error <= epsilon).astype(float)

    # Gradient for the linear part (Standard Check Loss)
    check_grad, check_hess = calc_check_grad_hess(error=error, alpha=alpha)

    # Gradient for the Huber part (Quadratic approximation)
    # dL/dp = check_grad * (|error| / epsilon)
    huber_grad = check_grad * (abs_error / epsilon)
    grad = mask * huber_grad + (1 - mask) * check_grad

    # Hessian for the Huber part
    # d2L/dp2 = |check_grad| / epsilon
    huber_hess = np.abs(check_grad) / epsilon
    # For the linear part, we use check_hess (1.0) as a proxy
    hess = mask * huber_hess + (1 - mask) * check_hess

    return grad, hess


def calc_approx_grad_hess(
    error: npt.NDArray, alpha: npt.NDArray | float, epsilon: float
) -> tuple[npt.NDArray, npt.NDArray]:
    """Compute the gradient and Hessian for the Smooth Quantile Approximation.

    This uses a smooth approximation derived from the Majorization-Minimization
    approach for quantile regression."""
    # dL/dp = 0.5 * (1 - 2 * alpha - error / (epsilon + |error|))
    approx_grad = 0.5 * (1 - 2 * alpha - error / (epsilon + np.abs(error)))

    # d2L/dp2 = 1 / (2 * (epsilon + |error|))
    approx_hess = 1 / (2 * (epsilon + np.abs(error)))
    return approx_grad, approx_hess


def _get_alpha_expanded(alphas: list[float], total_len: int) -> tuple[npt.NDArray, int]:
    """Expand the list of alphas to match the stacked dataset size."""
    n = total_len // len(alphas)
    return np.repeat(alphas, n), n


def eval_check_loss(
    y_pred: npt.NDArray,
    dtrain: lgb.Dataset | xgb.DMatrix,
    alphas: list[float],
) -> float:
    """Evaluate the mean check loss across all quantiles."""
    y_true = dtrain.get_label()
    if not isinstance(y_true, np.ndarray):
        y_true = np.array(y_true)

    alphas_expanded, n = _get_alpha_expanded(alphas, len(y_true))
    error = y_true - y_pred
    loss_all = calc_rho(error=error, alpha=alphas_expanded)

    # Return the sum of mean losses across all quantiles
    loss_reshaped = loss_all.reshape(len(alphas), n)
    return float(np.sum(np.mean(loss_reshaped, axis=1)))


def validate_epsilon(epsilon: float) -> None:
    """Ensure epsilon is a positive float."""
    if not isinstance(epsilon, float):
        raise ValidationException("Epsilon is not float type")

    if epsilon <= 0:
        raise ValidationException("Epsilon must be positive")


class MQObjective:
    """Encapsulates custom objective and evaluation functions for Multi-Quantile regression.
    This class handles the interface with LightGBM and XGBoost, providing the gradients and Hessians required for training."""

    def __init__(
        self,
        alphas: list[float],
        objective: ObjectiveName,
        model: ModelName,
        epsilon: float,
        weight: npt.NDArray | None = None,
    ) -> None:
        """Initialize the multi-quantile objective."""
        self.alphas = alphas
        self.objective = objective
        self.model = model
        self.epsilon = epsilon
        self.weight = weight

        # Pre-validate parameters
        if self.objective in (ObjectiveName.approx, ObjectiveName.huber):
            validate_epsilon(self.epsilon)

    def fobj(
        self, y_pred: npt.NDArray, dtrain: lgb.Dataset | xgb.DMatrix
    ) -> tuple[npt.NDArray, npt.NDArray]:
        """Standard interface for custom objective functions in LightGBM and XGBoost."""
        y_true = dtrain.get_label()
        if not isinstance(y_true, np.ndarray):
            y_true = np.array(y_true)

        alphas_expanded, n = _get_alpha_expanded(self.alphas, len(y_true))
        error = y_true - y_pred

        # Calculate gradients and Hessians based on objective
        if self.objective == ObjectiveName.check:
            grads, hess = calc_check_grad_hess(error, alphas_expanded)
        elif self.objective == ObjectiveName.huber:
            grads, hess = calc_huber_grad_hess(error, alphas_expanded, self.epsilon)
        elif self.objective == ObjectiveName.approx:
            grads, hess = calc_approx_grad_hess(error, alphas_expanded, self.epsilon)
        else:
            raise ValueError(f"Unknown objective: {self.objective}")

        # Normalize by original sample size
        grads /= n
        hess /= n

        if isinstance(self.weight, np.ndarray):
            return grads * self.weight, hess * self.weight
        return grads, hess

    def feval(
        self, y_pred: npt.NDArray, dtrain: lgb.Dataset | xgb.DMatrix
    ) -> tuple[str, float, bool] | tuple[str, float]:
        """Unified interface for custom evaluation functions."""
        if self.model == ModelName.lightgbm and isinstance(dtrain, lgb.Dataset):
            return self.lgb_feval(y_pred, dtrain)
        elif self.model == ModelName.xgboost and isinstance(dtrain, xgb.DMatrix):
            return self.xgb_feval(y_pred, dtrain)
        else:
            raise ValueError(f"Cannot evaluate {self.model}, got type {type(dtrain)}")

    def lgb_feval(
        self, y_pred: npt.NDArray, dtrain: lgb.Dataset
    ) -> tuple[str, float, bool]:
        """Specific evaluation function for LightGBM."""
        loss = eval_check_loss(y_pred, dtrain, self.alphas)
        return "check_loss", loss, False

    def xgb_feval(self, y_pred: npt.NDArray, dtrain: xgb.DMatrix) -> tuple[str, float]:
        """Specific evaluation function for XGBoost."""
        loss = eval_check_loss(y_pred, dtrain, self.alphas)
        return "check_loss", loss
