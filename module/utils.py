from typing import Any, List, Union, Dict
from itertools import repeat, chain

import numpy as np
import pandas as pd


__all__ = [
    "alpha_validate",
    "prepare_x",
    "prepare_train",
]


def alpha_validate(
    alphas: Union[List[float], float],
) -> List[float]:
    if isinstance(alphas, float):
        alphas = [alphas]
    return alphas


def prepare_x(
    x: Union[pd.DataFrame, pd.Series, np.ndarray],
    alphas: List[float],
) -> pd.DataFrame:
    if isinstance(x, np.ndarray) or isinstance(x, pd.Series):
        x = pd.DataFrame(x)
    assert "_tau" not in x.columns, "Column name '_tau' is not allowed."
    _alpha_repeat_count_list = [list(repeat(alpha, len(x))) for alpha in alphas]
    _alpha_repeat_list = list(chain.from_iterable(_alpha_repeat_count_list))
    _repeated_x = pd.concat([x] * len(alphas), axis=0)

    _repeated_x = _repeated_x.assign(
        _tau=_alpha_repeat_list,
    )
    return _repeated_x


def prepare_train(
    x: Union[pd.DataFrame, pd.Series, np.ndarray],
    y: Union[pd.Series, np.ndarray],
    alphas: List[float],
) -> Dict[str, Union[pd.DataFrame, np.ndarray]]:
    _train_df = prepare_x(x, alphas)
    _repeated_y = np.concatenate(list(repeat(y, len(alphas))))
    return (_train_df, _repeated_y)
