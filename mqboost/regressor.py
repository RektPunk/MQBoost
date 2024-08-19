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
    MQRegressor is a custom multiple quantile estimator that supports LightGBM and XGBoost models with
    preserving monotonicity among quantiles.

    Attributes:
        params (Dict[str, Any]):
            Parameters for the model.
            Any params related to model can be used except "objective".
        model (str): The model type (either 'lightgbm' or 'xgboost'). Default is 'lightgbm'.
        objective (str): The objective function (either 'check', 'huber', or 'approx'). Default is 'check'.
        delta (float):
            Parameter for the 'huber' objective function.
            Default is 0.01 and must be smaller than 0.05.
        epsilon (float):
            Parameter for the 'smooth approximated check' objective function.
            Default is 1e-5.
    Methods:
        fit(dataset, eval_set):
            Fits the regressor to the provided dataset, optionally evaluating on a separate validation set.
        predict(dataset):
            Predicts quantiles for the given dataset.

    Property:
        MQObj: Returns the MQObjective instance.
    """

    def __init__(
        self,
        params: Dict[str, Any],
        model: str = ModelName.lightgbm.value,
        objective: str = ObjectiveName.check.value,
        delta: float = 0.01,
        epsilon: float = 1e-5,
    ) -> None:
        """Initialize the MQRegressor."""
        self._params = params
        self._model = ModelName.get(model)
        self._objective = ObjectiveName.get(objective)
        self._delta = delta
        self._epsilon = epsilon

    def fit(
        self,
        dataset: MQDataset,
        eval_set: Optional[MQDataset] = None,
    ) -> None:
        """
        Fit the regressor to the dataset.
        Args:
            dataset (MQDataset): The dataset to fit the model on.
            eval_set (Optional[MQDataset]):
                The validation dataset. If None, the dataset is used for evaluation.
        """
        if eval_set is None:
            _eval_set = dataset.dtrain
        else:
            _eval_set = eval_set.dtrain

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
            epsilon=self._epsilon,
        )
        if self.__is_lgb:
            params.update({MQStr.obj.value: self._MQObj.fobj})
            self.model = lgb.train(
                train_set=dataset.dtrain,
                params=params,
                feval=self._MQObj.feval,
                valid_sets=[_eval_set],
            )
        elif self.__is_xgb:
            self.model = xgb.train(
                dtrain=dataset.dtrain,
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
        Predict quantiles for the dataset.
        Args:
            dataset (MQDataset): The dataset to make predictions on.
        Returns:
            np.ndarray: The predicted quantiles.
        """
        self.__predict_available()
        _pred = self.model.predict(data=dataset.dpredict)
        _pred = _pred.reshape(len(dataset.alphas), dataset.nrow)
        return _pred

    def __predict_available(self) -> None:
        """
        Check if the model has been fitted before making predictions.
        Raises:
            FittingException: If the model has not been fitted.
        """
        if not getattr(self, "_fitted", False):
            raise FittingException("Fit must be executed before predict")

    @property
    def MQObj(self) -> MQObjective:
        """
        Get the MQObjective instance.
        Returns:
            MQObjective: The MQObjective instance.
        """
        return self._MQObj

    @property
    def __is_lgb(self) -> bool:
        """Check if the model is LightGBM."""
        return self._model == ModelName.lightgbm

    @property
    def __is_xgb(self) -> bool:
        """Check if the model is XGBoost."""
        return self._model == ModelName.xgboost
