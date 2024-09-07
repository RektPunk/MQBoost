import warnings
from itertools import chain, repeat

import numpy as np
import pandas as pd

from mqboost.base import AlphaLike, ValidationException, XdataLike, YdataLike


def alpha_validate(
    alphas: AlphaLike,
) -> list[float]:
    """Validates the list of alphas ensuring they are in ascending order and contain no duplicates."""
    if isinstance(alphas, float):
        alphas = [alphas]

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
    x: XdataLike,
    alphas: list[float],
) -> pd.DataFrame:
    """Prepares and returns a stacked DataFrame of features repeated for each alpha, with an additional column indicating the alpha value.
    Raises:
        ValidationException: If the input data contains a column named '_tau'.
    """
    if isinstance(x, np.ndarray) or isinstance(x, pd.Series):
        x = pd.DataFrame(x)

    if "_tau" in x.columns:
        raise ValidationException("Column name '_tau' is not allowed.")

    _alpha_repeat_count_list = [list(repeat(alpha, len(x))) for alpha in alphas]
    _alpha_repeat_list = list(chain.from_iterable(_alpha_repeat_count_list))

    _repeated_x = pd.concat([x] * len(alphas), axis=0)
    _repeated_x = _repeated_x.assign(
        _tau=_alpha_repeat_list,
    )
    return _repeated_x


def prepare_y(
    y: YdataLike,
    alphas: list[float],
) -> np.ndarray:
    """Prepares and returns a stacked array of target values repeated for each alpha."""
    return np.concatenate(list(repeat(y, len(alphas))))


def delta_validate(delta: float) -> float:
    """Validates the delta parameter ensuring it is a positive float and less than or equal to 0.05."""
    _delta_upper_bound: float = 0.05

    if not isinstance(delta, float):
        raise ValidationException("Delta is not float type")

    if delta <= 0:
        raise ValidationException("Delta must be positive")

    if delta > _delta_upper_bound:
        warnings.warn("Delta should be 0.05 or less.")

    return delta
