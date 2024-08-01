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
    def __init__(
        self,
        alphas: AlphaLike,
        data: XdataLike,
        label: Optional[YdataLike] = None,
        model: str = ModelName.lightgbm.value,
    ) -> None:
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
        return self._train_dtype

    @property
    def predict_dtype(self) -> Callable:
        return self._predict_dtype

    @property
    def model(self) -> ModelName:
        return self._model

    @property
    def columns(self) -> pd.Index:
        return self._columns

    @property
    def nrow(self) -> int:
        return self._nrow

    @property
    def data(self) -> pd.DataFrame:
        return self._data

    @property
    def label(self) -> pd.DataFrame:
        self.__label_available()
        return self._label

    @property
    def dtrain(self) -> DtrainLike:
        self.__label_available()
        return self._train_dtype(data=self._data, label=self._label)

    @property
    def dpredict(self) -> Union[DtrainLike, Callable]:
        return self._predict_dtype(data=self._data)

    @property
    def alphas(self) -> List[float]:
        return self._alphas

    def __label_available(self) -> None:
        if getattr(self, "_is_none_label", True):
            raise FittingException("Fitting is impossible since label is None")
