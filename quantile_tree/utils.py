from typing import List, Union, Tuple
from itertools import repeat, chain

import numpy as np
import pandas as pd

from .base import XdataLike, YdataLike, AlphaLike


def alpha_validate(
    alphas: AlphaLike,
) -> List[float]:
    """
    Validate alphas
    Args:
        alphas (AlphaLike)

    Returns:
        List[float]
    """
    if isinstance(alphas, float):
        alphas = [alphas]
    assert len(alphas) > 0, "alpha is not valid"

    return alphas


def prepare_x(
    x: XdataLike,
    alphas: List[float],
) -> pd.DataFrame:
    """
    Return stacked X
    Args:
        x (XdataLike)
        alphas (List[float])

    Returns:
        pd.DataFrame
    """
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
    x: XdataLike,
    y: YdataLike,
    alphas: List[float],
) -> Tuple[str, Union[pd.DataFrame, np.ndarray]]:
    """
    Return stacked X, y for training
    Args:
        x (XdataLike)
        y (YdataLike)
        alphas (List[float])

    Returns:
        Dict[str, Union[pd.DataFrame, np.ndarray]]
    """
    _train_df = prepare_x(x, alphas)
    _repeated_y = np.concatenate(list(repeat(y, len(alphas))))
    return (_train_df, _repeated_y)


def delta_validate(delta: float) -> None:
    assert delta <= 0.1, "Delta smaller than 0.1 highly recommended"
