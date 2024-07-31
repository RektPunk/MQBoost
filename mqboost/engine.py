from typing import Any, Dict, Optional

import lightgbm as lgb
import numpy as np
import xgboost as xgb

from mqboost.base import FittingException, ModelName, MQStr, ObjectiveName
from mqboost.constraints import set_monotone_constraints
from mqboost.dataset import MQDataset
from mqboost.objective import MQObjective

__all__ = ["MQRegressor"]


class MQRegressor:
    """
    Monotone quantile regressor which preserving monotonicity among quantiles
    Attributes
    ----------
    params: Dict[str, Any]
        Train parameter.
    model: str, optioal
        Determine base model. Defaults to "lightgbm", another option is "xgboost".
    objective: str, optional
        Determine objective function. Defaults to "check", another option is "huber".
    delta: float, optional
        Only used with "huber" objective.
        Defaults to 0.05 and must be smaller than 0.1.
    Methods
    ----------
    fit
    predict
    """

    def __init__(
        self,
        params: Dict[str, Any],
        model: str = ModelName.lightgbm.value,
        objective: str = ObjectiveName.check.value,
        delta: float = 0.05,
    ) -> None:
        self._params = params
        self._model = ModelName.get(model)
        self._objective = ObjectiveName.get(objective)
        self._delta = delta

    def fit(
        self,
        dataset: MQDataset,
        eval_set: Optional[MQDataset] = None,
    ) -> None:
        """
        Fit regressor
        Args:
            dataset (MQDataset)
            eval_set (MQDataset, optional)
                Defaults to None. If None, dataset input is used.
        """
        if eval_set is None:
            _eval_set = dataset.train
        else:
            _eval_set = eval_set.train

        params = set_monotone_constraints(
            params=self._params,
            columns=dataset.columns,
            model_name=self._model,
        )
        self._MQObj = MQObjective(
            alphas=dataset.alphas,
            objective=self._objective,
            model=self._model,
            delta=self._delta,
        )
        if self.__is_lgb:
            params.update({MQStr.obj: self._MQObj.fobj})
            self.model = lgb.train(
                train_set=dataset.train,
                params=params,
                feval=self._MQObj.feval,
                valid_sets=[_eval_set],
            )
        elif self.__is_xgb:
            self.model = xgb.train(
                dtrain=dataset.train,
                verbose_eval=False,
                params=params,
                obj=self._MQObj.fobj,
                custom_metric=self._MQObj.feval,
                evals=[(_eval_set, "eval")],
            )
        self._fitted = True

    def predict(
        self,
        dataset: MQDataset,
    ) -> np.ndarray:
        """
        Return predicted quantiles
        Args:
            dataset (MQDataset)
        Returns:
            np.ndarray
        """
        self.__predict_available()
        _pred = self.model.predict(data=dataset.predict)
        _pred = _pred.reshape(len(dataset.alphas), dataset.nrow)
        return _pred

    def __predict_available(self) -> None:
        if not getattr(self, "_fitted", False):
            raise FittingException("train must be executed before predict")

    @property
    def MQObj(self) -> MQObjective:
        return self._MQObj

    @property
    def __is_lgb(self) -> bool:
        return self._model == ModelName.lightgbm

    @property
    def __is_xgb(self) -> bool:
        return self._model == ModelName.xgboost
