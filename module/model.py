from typing import List, Union, Dict, Any
from functools import partial
from itertools import repeat, chain

import numpy as np
import lightgbm as lgb
import pandas as pd

from module.objective import check_loss_eval, check_loss_grad_hess


def _alpha_validate(
    alphas: Union[List[float], float],
) -> List[float]:
    if isinstance(alphas, float):
        alphas = [alphas]
    return alphas


def _prepare_x(
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


def _prepare_train(
    x: Union[pd.DataFrame, pd.Series, np.ndarray],
    y: Union[pd.Series, np.ndarray],
    alphas: List[float],
) -> Dict[str, Union[pd.DataFrame, np.ndarray]]:
    _train_df = _prepare_x(x, alphas)
    _repeated_y = np.concatenate(list(repeat(y, len(alphas))))
    return (_train_df, _repeated_y)


class MonotonicQuantileRegressor:
    def __init__(
        self,
        x: Union[pd.DataFrame, pd.Series, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        alphas: Union[List[float], float],
    ):
        alphas = _alpha_validate(alphas)
        self.x_train, self.y_train = _prepare_train(x, y, alphas)
        self.dataset = lgb.Dataset(data=self.x_train, label=self.y_train)
        self.fobj = partial(check_loss_grad_hess, alphas=alphas)
        self.feval = partial(check_loss_eval, alphas=alphas)

    def train(self, params: Dict[str, Any]) -> lgb.basic.Booster:
        self._params = params.copy()
        if "monotone_constraints" in self._params:
            self._params["monotone_constraints"].append(1)
        else:
            self._params.update(
                {
                    "monotone_constraints": [
                        1 if "_tau" == col else 0 for col in self.x_train.columns
                    ]
                }
            )
        self.model = lgb.train(
            train_set=self.dataset,
            verbose_eval=False,
            params=self._params,
            fobj=self.fobj,
            feval=self.feval,
        )
        return self.model

    def predict(
        self,
        x: Union[pd.DataFrame, pd.Series, np.ndarray],
        alphas: Union[List[float], float],
    ) -> np.ndarray:
        alphas = _alpha_validate(alphas)
        _x = _prepare_x(x, alphas)
        _pred = self.model.predict(_x)
        _pred = _pred.reshape(len(alphas), len(x))
        return _pred
