from typing import Callable, Optional

import numpy as np
import pandas as pd

from mqboost.base import (
    FUNC_TYPE,
    AlphaLike,
    DtrainLike,
    FittingException,
    ModelName,
    TypeName,
    WeightLike,
    XdataLike,
    YdataLike,
)
from mqboost.encoder import MQLabelEncoder
from mqboost.utils import alpha_validate, prepare_x, prepare_y, to_dataframe


def _compare_datasets(data: pd.DataFrame, reference_data: pd.DataFrame) -> None:
    if data.shape[1] != reference_data.shape[1] - 1:
        raise ValueError("Number of columns do not match")


class MQDataset:
    """
    MQDataset encapsulates the dataset used for training and predicting with the MQRegressor.
    It supports both LightGBM and XGBoost models, handling data preparation, validation, and conversion for training and prediction.

    Attributes:
        alphas (list[float] | float):
            List of quantile levels.
            Must be in ascending order and contain no duplicates.
        data (pd.DataFrame | pd.Series | np.ndarray): The input features.
        label (pd.Series | np.ndarray): The target labels (if provided).
        weight (list[float] | list[int] | np.ndarray | pd.Series): Weight for each instance (if provided).
        model (str): The model type (LightGBM or XGBoost).
        reference (MQBoost | None): Reference dataset for label encoding and label mean.

    Property:
        train_dtype: Returns the data type function for training data.
        predict_dtype: Returns the data type function for prediction data.
        columns: Returns the column names of the input features.
        nrow: Returns the number of rows in the dataset.
        data: Returns the input features.
        label: Returns the target labels.
        alphas: Returns the list of quantile levels.
        dtrain: Returns the training data in the required format for the model.
        dpredict: Returns the prediction data in the required format for the model.
    """

    def __init__(
        self,
        alphas: AlphaLike,
        data: XdataLike,
        label: YdataLike | None = None,
        weight: WeightLike | None = None,
        model: str = ModelName.lightgbm.value,
        reference: Optional["MQDataset"] = None,
    ) -> None:
        """Initialize the MQDataset."""
        self._model = ModelName.get(model)
        self._nrow = len(data)
        self._alphas = alpha_validate(alphas)

        _funcs = FUNC_TYPE.get(self._model)
        self._train_dtype: Callable = _funcs.get(TypeName.train_dtype)
        self._predict_dtype: Callable = _funcs.get(TypeName.predict_dtype)

        _data = to_dataframe(data)
        if reference is None:
            self.encoders: dict[str, MQLabelEncoder] = {}
            for col in _data.columns:
                if _data[col].dtype == "object":
                    _encoder = MQLabelEncoder()
                    _data[col] = _encoder.fit_transform(_data[col])
                    self.encoders.update({col: _encoder})
        else:
            _compare_datasets(_data, reference.data)
            self.encoders = reference.encoders.copy()
            for col, _encoder in self.encoders.items():
                _data[col] = _encoder.transform(_data[col])

        self._data = prepare_x(x=_data, alphas=self._alphas)
        self._columns = self._data.columns
        if label is not None:
            self._label_mean = reference.label_mean if reference else label.mean()
            self._label = prepare_y(y=label - self._label_mean, alphas=self._alphas)
            self._is_none_label = False

        if weight is not None:
            _weight = np.array(weight) if not isinstance(weight, np.ndarray) else weight
            self._weight = prepare_y(y=_weight, alphas=self._alphas)

    @property
    def train_dtype(self) -> Callable:
        """Get the data type function for training data."""
        return self._train_dtype

    @property
    def predict_dtype(self) -> Callable:
        """Get the data type function for prediction data."""
        return self._predict_dtype

    @property
    def columns(self) -> pd.Index:
        """Get the column names of the input features."""
        return self._columns

    @property
    def nrow(self) -> int:
        """Get the number of rows in the dataset."""
        return self._nrow

    @property
    def data(self) -> pd.DataFrame:
        """Get the raw input features."""
        return self._data

    @property
    def alphas(self) -> list[float]:
        """Get the list of quantile levels."""
        return self._alphas

    @property
    def label(self) -> pd.DataFrame:
        """Get the raw target labels."""
        self.__label_available()
        return self._label

    @property
    def label_mean(self) -> float:
        """Get the label mean."""
        self.__label_available()
        return self._label_mean

    @property
    def weight(self) -> WeightLike | None:
        """Get the weights."""
        return getattr(self, "_weight", None)

    @property
    def dtrain(self) -> DtrainLike:
        """Get the training data in the required format for the model."""
        self.__label_available()
        return self._train_dtype(data=self._data, label=self._label, weight=self.weight)

    @property
    def dpredict(self) -> DtrainLike | Callable:
        """Get the prediction data in the required format for the model."""
        return self._predict_dtype(data=self._data)

    def __label_available(self) -> None:
        """Check if the label is available, raises an exception if not."""
        if getattr(self, "_is_none_label", True):
            raise FittingException("Fitting is impossible since label is None")
