from typing import Callable, Optional, Union

import numpy as np
import pandas as pd

from mqboost.base import FUNC_TYPE, ModelName, TypeName, XdataLike, YdataLike, AlphaLike
from mqboost.utils import prepare_x, prepare_y


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

        _funcs = FUNC_TYPE.get(self._model)
        _train_dtype: Callable = _funcs.get(TypeName.train_dtype)
        _predict_dtype: Callable = _funcs.get(TypeName.predict_dtype)

        _data = prepare_x(x=data, alphas=alphas)
        self._columns: pd.Index[str] = self._data.columns

        if label is None:
            self._train = None
            self._predict = _predict_dtype(data=_data)
        else:
            self._label = prepare_y(y=label, alphas=self._alphas)
            self._train = _train_dtype(data=self._data, label=self._label)
            self._predict = _predict_dtype(data=self._data, label=self._label)

    @property
    def model(self):
        return self._model

    @property
    def columns(self):
        return self._columns

    @property
    def nrow(self):
        return self._nrow

    @property
    def train(self):
        return self._train

    @property
    def predict(self):
        return self._predict
