import lightgbm as lgb
import numpy as np
import numpy.typing as npt
import xgboost as xgb

from mqboost.base import ModelName, ObjectiveName, ValidationException


def calc_rho(error: npt.NDArray, alpha: npt.NDArray | float) -> npt.NDArray:
    """Compute rho (pinball loss) for the given error and alpha."""
    # L = (alpha - I(error < 0)) * error
    return (alpha - (error < 0).astype(int)) * error


def calc_check_grad_hess(
    error: npt.NDArray, alpha: npt.NDArray | float
) -> tuple[npt.NDArray, npt.NDArray]:
    """Compute gradient and Hessian for the check loss."""
    # dL/dp = I(error < 0) - alpha
    # d2L/dp2 = 1 as a proxy for Hessian
    return (error < 0).astype(int) - alpha, np.ones_like(error)


def calc_huber_grad_hess(
    error: npt.NDArray, alpha: npt.NDArray | float, epsilon: float
) -> tuple[npt.NDArray, npt.NDArray]:
    """Compute gradient and Hessian for the Huber loss (Smooth Quantile Loss)."""
    abs_error = np.abs(error)
    mask = (abs_error <= epsilon).astype(float)

    # Gradient for linear part
    check_grad, check_hess = calc_check_grad_hess(error=error, alpha=alpha)
    # Gradient for Huber part
    # dL/dp = check_grad * (abs_error / epsilon)
    huber_grad = check_grad * (abs_error / epsilon)
    grad = mask * huber_grad + (1 - mask) * check_grad

    # Hessian for Huber part
    # d2L/dp2 = |check_grad| / epsilon
    huber_hess = np.abs(check_grad) / epsilon
    # For linear part, we use check_hess as a proxy for Hessian
    hess = mask * huber_hess + (1 - mask) * check_hess

    return grad, hess


def calc_approx_grad_hess(
    error: npt.NDArray, alpha: npt.NDArray | float, epsilon: float
) -> tuple[npt.NDArray, npt.NDArray]:
    """Compute gradient and Hessian for the approximate loss (MM loss)."""
    # dL/dp = 0.5 * (1 - 2 * alpha - error / (epsilon + |error|))
    approx_grad = 0.5 * (1 - 2 * alpha - error / (epsilon + np.abs(error)))

    # d2L/dp2 = 1 / (2 * (epsilon + |error|))
    approx_hess = 1 / (2 * (epsilon + np.abs(error)))
    return approx_grad, approx_hess


def _get_alpha_expanded(alphas: list[float], total_len: int) -> tuple[npt.NDArray, int]:
    """Helper to expand alphas and get original dataset size."""
    n = total_len // len(alphas)
    return np.repeat(alphas, n), n


def eval_check_loss(
    y_pred: npt.NDArray,
    dtrain: lgb.Dataset | xgb.DMatrix,
    alphas: list[float],
) -> float:
    """Evaluate the check loss function using vectorized operations."""
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
    """Validate epsilon parameter ensuring it is a positive float."""
    if not isinstance(epsilon, float):
        raise ValidationException("Epsilon is not float type")

    if epsilon <= 0:
        raise ValidationException("Epsilon must be positive")


class MQObjective:
    """MQObjective encapsulates the objective and evaluation functions for the MQRegressor."""

    def __init__(
        self,
        alphas: list[float],
        objective: ObjectiveName,
        model: ModelName,
        epsilon: float,
        weight: npt.NDArray | None = None,
    ) -> None:
        """Initialize the MQObjective."""
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
        """Custom objective function for LightGBM and XGBoost."""
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

        # Normalize and apply weights
        grads /= n
        hess /= n

        if isinstance(self.weight, np.ndarray):
            return grads * self.weight, hess * self.weight
        return grads, hess

    def feval(
        self, y_pred: npt.NDArray, dtrain: lgb.Dataset | xgb.DMatrix
    ) -> tuple[str, float, bool] | tuple[str, float]:
        """Custom evaluation function for LightGBM and XGBoost."""
        if self.model == ModelName.lightgbm:
            return self.lgb_feval(y_pred, dtrain)  # type: ignore
        return self.xgb_feval(y_pred, dtrain)  # type: ignore

    def lgb_feval(
        self, y_pred: npt.NDArray, dtrain: lgb.Dataset
    ) -> tuple[str, float, bool]:
        """Custom evaluation function for LightGBM."""
        loss = eval_check_loss(y_pred, dtrain, self.alphas)
        return "check_loss", loss, False

    def xgb_feval(self, y_pred: npt.NDArray, dtrain: xgb.DMatrix) -> tuple[str, float]:
        """Custom evaluation function for XGBoost."""
        loss = eval_check_loss(y_pred, dtrain, self.alphas)
        return "check_loss", loss
