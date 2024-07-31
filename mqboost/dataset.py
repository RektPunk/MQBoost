from typing import Callable, Optional, Union

import numpy as np
import pandas as pd

from mqboost.base import FUNC_TYPE, ModelName, TypeName, XdataLike, YdataLike
from mqboost.utils import prepare_x, prepare_y


class MQDataset:
    def __init__(
        self,
        data: XdataLike,
        label: Optional[YdataLike] = None,
        model: str = ModelName.lightgbm.value,
    ) -> None:
        self._model = ModelName.get(model)

        _funcs = FUNC_TYPE.get(self._model)
        self._train_dtype: Callable = _funcs.get(TypeName.train_dtype)
        self._predict_dtype: Callable = _funcs.get(TypeName.predict_dtype)
