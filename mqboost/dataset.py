from typing import Callable

import lightgbm as lgb
import numpy as np
import numpy.typing as npt
import pandas as pd
import xgboost as xgb

from mqboost.base import (
    FUNC_TYPE,
    FittingException,
    ModelName,
    TypeName,
    ValidationException,
)


def validate_alpha(
    alphas: list[float] | float,
) -> list[float]:
    """Validate target quantiles (alphas). Ensures alphas are in (0, 1), in ascending order, and contain no duplicates."""
    if isinstance(alphas, float):
        alphas = [alphas]

    if not isinstance(alphas, list):
        raise ValidationException("Alpha must be a list or float")

    if 0.0 in alphas or 1.0 in alphas:
        raise ValidationException("Alpha cannot be 0 or 1")

    _len_alphas = len(alphas)
    if _len_alphas == 0:
        raise ValidationException("Input alpha is not valid")

    if _len_alphas >= 2 and any(
        alphas[i] > alphas[i + 1] for i in range(_len_alphas - 1)
    ):
        raise ValidationException("Alpha is not ascending order")

    if _len_alphas != len(set(alphas)):
        raise ValidationException("Duplicated alpha exists")

    return alphas


def prepare_x(
    x: pd.DataFrame,
    alphas: list[float],
) -> pd.DataFrame:
    """Prepare the feature matrix for multi-quantile training by stacking the dataset
    and adding a '_tau' column to indicate the quantile level."""
    if "_tau" in x.columns:
        raise ValidationException("Column name '_tau' is not allowed.")

    num_alphas = len(alphas)
    num_rows = len(x)

    _alpha_repeat_list = np.repeat(alphas, num_rows)
    _repeated_x = pd.concat([x] * num_alphas, axis=0).reset_index(drop=True)
    _repeated_x["_tau"] = _alpha_repeat_list

    return _repeated_x


def prepare_y(
    y: pd.Series | npt.NDArray,
    alphas: list[float],
) -> npt.NDArray:
    """Prepare the target vector by repeating it for each target quantile."""
    return np.tile(y, len(alphas))


def to_dataframe(x: pd.DataFrame | pd.Series | npt.NDArray) -> pd.DataFrame:
    """Convert numpy array or pandas Series to a pandas DataFrame."""
    if isinstance(x, np.ndarray) or isinstance(x, pd.Series):
        _x = pd.DataFrame(x)
    else:
        _x = x.copy()
    return _x


class MQDataset:
    """A container for multi-quantile datasets, handling the transformation into
    a stacked format suitable for LightGBM and XGBoost training.

    Attributes:
        alphas (list[float] | float):
            List of quantile levels.
            Must be in ascending order and contain no duplicates.
        data (pd.DataFrame | pd.Series | np.ndarray): The input features.
        label (pd.Series | np.ndarray): The target labels (if provided).
        weight (list[float] | list[int] | np.ndarray | pd.Series): Weight for each instance (if provided).
        model (str): The model type (LightGBM or XGBoost)."""

    def __init__(
        self,
        alphas: list[float] | float,
        data: pd.DataFrame | pd.Series | npt.NDArray,
        label: pd.Series | npt.NDArray | None = None,
        weight: list[float] | list[int] | npt.NDArray | pd.Series | None = None,
        model: str = ModelName.lightgbm.value,
    ) -> None:
        """Initialize the MQDataset."""
        self._model = ModelName[model]
        self.nrow = len(data)
        self.alphas = validate_alpha(alphas)

        _funcs = FUNC_TYPE[self._model]
        self.train_dtype = _funcs[TypeName.train_dtype]
        self.predict_dtype = _funcs[TypeName.predict_dtype]

        _data = to_dataframe(data)
        self.data = prepare_x(x=_data, alphas=self.alphas)
        self.columns = self.data.columns

        self._label_raw = label
        self._label_mean = None
        if label is not None:
            self._label_mean = label.mean()
            self._label = prepare_y(y=label - self._label_mean, alphas=self.alphas)
            self._is_none_label = False
        else:
            self._is_none_label = True

        self._weight_raw = weight
        if weight is not None:
            _weight = np.array(weight) if not isinstance(weight, np.ndarray) else weight
            self._weight = prepare_y(y=_weight, alphas=self.alphas)

    def set_label_mean(self, label_mean: float) -> None:
        """Re-center labels using a new mean."""
        if self._label_raw is None:
            raise ValidationException("Cannot set label mean when labels are None")
        self._label_mean = label_mean
        self._label = prepare_y(
            y=self._label_raw - self._label_mean, alphas=self.alphas
        )

    @property
    def label(self) -> npt.NDArray:
        """Get the raw target labels."""
        self.__label_available()
        return self._label

    @property
    def label_mean(self) -> float:
        """Get the label mean."""
        self.__label_available()
        if self._label_mean is None:
            raise ValidationException("Label mean is None")
        return float(self._label_mean)

    @property
    def weight(self) -> npt.NDArray | None:
        """Get the weights."""
        return getattr(self, "_weight", None)

    @property
    def dtrain(self) -> lgb.Dataset | xgb.DMatrix:
        """Get the training data in the required format for the model."""
        self.__label_available()
        return self.train_dtype(data=self.data, label=self._label, weight=self.weight)

    @property
    def dpredict(self) -> lgb.Dataset | xgb.DMatrix | Callable:
        """Get the prediction data in the required format for the model."""
        return self.predict_dtype(data=self.data)

    def __label_available(self) -> None:
        """Check if the label is available, raises an exception if not."""
        if getattr(self, "_is_none_label", True):
            raise FittingException("Fitting is impossible since label is None")
