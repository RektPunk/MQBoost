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
)
from mqboost.encoder import MQLabelEncoder
from mqboost.utils import alpha_validate, prepare_x, prepare_y, to_dataframe


class MQDataset:
    """MQDataset encapsulates the dataset used for training and predicting with the MQRegressor.
    It supports both LightGBM and XGBoost models, handling data preparation, validation, and conversion for training and prediction.

    Attributes:
        alphas (list[float] | float):
            List of quantile levels.
            Must be in ascending order and contain no duplicates.
        data (pd.DataFrame | pd.Series | np.ndarray): The input features.
        label (pd.Series | np.ndarray): The target labels (if provided).
        weight (list[float] | list[int] | np.ndarray | pd.Series): Weight for each instance (if provided).
        model (str): The model type (LightGBM or XGBoost).
    """

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
        self.alphas = alpha_validate(alphas)

        _funcs = FUNC_TYPE[self._model]
        self.train_dtype = _funcs[TypeName.train_dtype]
        self.predict_dtype = _funcs[TypeName.predict_dtype]

        _data = to_dataframe(data)
        self.encoders: dict[str, MQLabelEncoder] = {}
        for col in _data.select_dtypes(exclude="number").columns:
            _series = _data[col]
            if not isinstance(_series, pd.Series):
                continue
            _encoder = MQLabelEncoder()
            _data[col] = _encoder.fit_transform(_series)
            self.encoders.update({col: _encoder})

        self.data = prepare_x(x=_data, alphas=self.alphas)
        self.columns = self.data.columns
        if label is not None:
            self._label_mean = label.mean()
            self._label = prepare_y(y=label - self._label_mean, alphas=self.alphas)
            self._is_none_label = False

        if weight is not None:
            _weight = np.array(weight) if not isinstance(weight, np.ndarray) else weight
            self._weight = prepare_y(y=_weight, alphas=self.alphas)

    @property
    def label(self) -> npt.NDArray:
        """Get the raw target labels."""
        self.__label_available()
        return self._label

    @property
    def label_mean(self) -> float:
        """Get the label mean."""
        self.__label_available()
        return self._label_mean

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
