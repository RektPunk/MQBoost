from typing import Any, Dict, Optional

import lightgbm as lgb
import numpy as np
import xgboost as xgb

from mqboost.base import AlphaLike, FittingException, ModelName, MQStr, ObjectiveName
from mqboost.constraints import set_monotone_constraints
from mqboost.dataset import MQDataset
from mqboost.objective import MQObjective
from mqboost.utils import alpha_validate

__all__ = ["MQRegressor"]


class MQRegressor:
    """
    Monotone quantile regressor which preserving monotonicity among quantiles
    Attributes
    ----------
    x: Union[pd.DataFrame, pd.Series, np.ndarray]
    y: Union[pd.Series, np.ndarray]
    alphas: Union[List[float], float]
        It must be in ascending order and not contain duplicates.
    objective: str, optional
        Determine objective function. Defaults to "check", another option is "huber".
    model: str, optioal
        Determine base model. Defaults to "lightgbm", another option is "xgboost".
    delta: float, optional
        Only used with "huber" objective.
        Defaults to 0.05 and must be smaller than 0.1.

    Methods
    ----------
    train
    predict
    """

    def __init__(
        self,
        alphas: AlphaLike,
        params: Dict[str, Any],
        model: str = ModelName.lightgbm.value,
        objective: str = ObjectiveName.check.value,
        delta: float = 0.05,
    ) -> None:
        self._model = ModelName.get(model)
        self._objective = ObjectiveName.get(objective)
        self._alphas = alpha_validate(alphas)
        self._params = set_monotone_constraints(
            params=params,
            columns=self._dataset.columns,
            model_name=self._model,
        )
        self._MQObj = MQObjective(
            alphas=self._alphas,
            objective=self._objective,
            model=self._model,
            delta=delta,
        )

    def fit(
        self,
        dataset: MQDataset,
        eval_set: Optional[MQDataset] = None,
    ) -> None:
        """
        fit regressor
        """
        eval_set = dataset if eval_set is None else eval_set
        if self.__is_lgb:
            self._params.update({MQStr.obj: self._MQObj.fobj})
            self.model = lgb.train(
                train_set=dataset,
                params=self._params,
                feval=self._MQObj.feval,
                valid_sets=[eval_set],
            )
        elif self.__is_xgb:
            self.model = xgb.train(
                dtrain=dataset,
                verbose_eval=False,
                params=self._params,
                obj=self._MQObj.fobj,
                custom_metric=self._MQObj.feval,
                evals=[(eval_set, "eval")],
            )
        else:
            raise FittingException("model name is invalid")
        self._fitted = True

    def predict(
        self,
        dataset: MQDataset,
        alphas: Optional[AlphaLike] = None,
    ) -> np.ndarray:
        """
        Return predicted quantiles
        """
        self.__is_fitted
        if alphas is None:
            alphas = self._alphas
        else:
            alphas = alpha_validate(alphas=alphas)
        _pred = self.model.predict(data=dataset.predict)
        _pred = _pred.reshape(len(alphas), dataset.nrow)
        return _pred

    @property
    def MQObj(self) -> MQObjective:
        return self._MQObj

    @property
    def __is_lgb(self) -> bool:
        return self._model == ModelName.lightgbm

    @property
    def __is_xgb(self) -> bool:
        return self._model == ModelName.xgboost

    @property
    def __is_fitted(self) -> None:
        if not getattr(self, "_fitted", False):
            raise FittingException("train must be executed before predict")
