from typing import Callable, List, Optional, Union

import pandas as pd

from mqboost.base import (
    FUNC_TYPE,
    AlphaLike,
    DtrainLike,
    FittingException,
    ModelName,
    TypeName,
    XdataLike,
    YdataLike,
)
from mqboost.utils import alpha_validate, prepare_x, prepare_y


class MQDataset:
    """
    MQDataset encapsulates the dataset used for training and predicting with the MQRegressor.
    It supports both LightGBM and XGBoost models, handling data preparation, validation, and conversion for training and prediction.

    Attributes:
        alphas (List[float]):
            List of quantile levels.
            Must be in ascending order and contain no duplicates.
        data (pd.DataFrame): The input features.
        label (pd.DataFrame): The target labels (if provided).
        model (ModelName): The model type (LightGBM or XGBoost).

    Property:
        train_dtype: Returns the data type function for training data.
        predict_dtype: Returns the data type function for prediction data.
        model: Returns the model type.
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
        label: Optional[YdataLike] = None,
        model: str = ModelName.lightgbm.value,
    ) -> None:
        """Initialize the MQDataset."""
        self._model = ModelName.get(model)
        self._nrow = len(data)
        self._alphas = alpha_validate(alphas)

        _funcs = FUNC_TYPE.get(self._model)
        self._train_dtype: Callable = _funcs.get(TypeName.train_dtype)
        self._predict_dtype: Callable = _funcs.get(TypeName.predict_dtype)

        self._data = prepare_x(x=data, alphas=self._alphas)
        self._columns = self._data.columns

        if label is not None:
            self._label = prepare_y(y=label, alphas=self._alphas)
            self._is_none_label = False

    @property
    def train_dtype(self) -> Callable:
        """
        Get the data type function for training data.
        Returns:
            Callable: The function that converts data to the required training data type.
        """
        return self._train_dtype

    @property
    def predict_dtype(self) -> Callable:
        """
        Get the data type function for prediction data.
        Returns:
            Callable: The function that converts data to the required prediction data type.
        """
        return self._predict_dtype

    @property
    def model(self) -> ModelName:
        """
        Get the model type.
        Returns:
            ModelName: The model type (LightGBM or XGBoost).
        """
        return self._model

    @property
    def columns(self) -> pd.Index:
        """
        Get the column names of the input features.
        Returns:
            pd.Index: The column names.
        """
        return self._columns

    @property
    def nrow(self) -> int:
        """
        Get the number of rows in the dataset.
        Returns:
            int: The number of rows.
        """
        return self._nrow

    @property
    def data(self) -> pd.DataFrame:
        """
        Get the raw input features.
        Returns:
            pd.DataFrame: The input features.
        """
        return self._data

    @property
    def label(self) -> pd.DataFrame:
        """
        Get the raw target labels.
        Returns:
            pd.DataFrame: The target labels.
        """
        self.__label_available()
        return self._label

    @property
    def alphas(self) -> List[float]:
        """
        Get the list of quantile levels.
        Returns:
            List[float]: The quantile levels.
        """
        return self._alphas

    @property
    def dtrain(self) -> DtrainLike:
        """
        Get the training data in the required format for the model.
        Returns:
            DtrainLike: The training data.
        """
        self.__label_available()
        return self._train_dtype(data=self._data, label=self._label)

    @property
    def dpredict(self) -> Union[DtrainLike, Callable]:
        """
        Get the prediction data in the required format for the model.
        Returns:
            Union[DtrainLike, Callable]: The prediction data.
        """
        return self._predict_dtype(data=self._data)

    def __label_available(self) -> None:
        """Check if the label is available, raises an exception if not."""
        if getattr(self, "_is_none_label", True):
            raise FittingException("Fitting is impossible since label is None")
