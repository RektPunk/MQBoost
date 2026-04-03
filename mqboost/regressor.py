from typing import Any

import lightgbm as lgb
import numpy as np
import numpy.typing as npt
import xgboost as xgb

from mqboost.base import FittingException, ModelName, ObjectiveName
from mqboost.constraints import set_monotone_constraints
from mqboost.dataset import MQDataset
from mqboost.objective import MQObjective
from mqboost.utils import params_validate

__all__ = ["MQRegressor"]


class MQRegressor:
    """MQRegressor is a custom multiple quantile estimator that supports LightGBM and XGBoost models with
    preserving monotonicity among quantiles.

    Attributes:
        params (dict[str, Any]):
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
    """

    def __init__(
        self,
        params: dict[str, Any],
        model: str = ModelName.lightgbm.value,
        objective: str = ObjectiveName.check.value,
        delta: float = 0.01,
        epsilon: float = 1e-5,
    ) -> None:
        """Initialize the MQRegressor."""
        params_validate(params=params)
        self.params = params
        self.model_name = ModelName[model]
        self.objective = ObjectiveName[objective]
        self.delta = delta
        self.epsilon = epsilon

    def fit(
        self,
        dataset: MQDataset,
        eval_set: MQDataset | None = None,
        **kwargs,
    ) -> None:
        """Fit the regressor to the dataset.
        Args:
            dataset (MQDataset): The dataset to fit the model on.
            eval_set (Optional[MQDataset]):
                The validation dataset. If None, the dataset is used for evaluation.
            **kwargs:
                train parameters.
        """
        if eval_set:
            eval_set_dtrain = eval_set.dtrain
        else:
            eval_set_dtrain = dataset.dtrain

        self._label_mean = dataset.label_mean

        params = set_monotone_constraints(
            params=self.params,
            columns=dataset.columns,
            model_name=self.model_name,
        )
        self.MQObj = MQObjective(
            alphas=dataset.alphas,
            objective=self.objective,
            weight=dataset.weight,
            model=self.model_name,
            delta=self.delta,
            epsilon=self.epsilon,
        )
        if self.__is_lgb:
            params.update({"objective": self.MQObj.fobj})
            if not (
                isinstance(dataset.dtrain, lgb.Dataset)
                and isinstance(eval_set_dtrain, lgb.Dataset)
            ):
                raise ValueError("dtrain must be a lightgbm Dataset")

            self.model = lgb.train(
                train_set=dataset.dtrain,
                params=params,
                feval=self.MQObj.feval,
                valid_sets=[eval_set_dtrain],
                **kwargs,
            )
        elif self.__is_xgb:
            self.model = xgb.train(
                dtrain=dataset.dtrain,
                verbose_eval=False,
                params=params,
                obj=self.MQObj.fobj,
                custom_metric=self.MQObj.feval,
                evals=[(eval_set_dtrain, "eval")],
                **kwargs,
            )
        self._colnames = dataset.columns.to_list()
        self._fitted = True

    def predict(
        self,
        dataset: MQDataset,
    ) -> npt.NDArray:
        """Predict quantiles for the dataset.
        Args:
            dataset (MQDataset): The dataset to make predictions on.
        Returns:
            np.ndarray: The predicted quantiles.
        """
        self.__predict_available()
        _pred = (
            np.asanyarray(self.model.predict(data=dataset.dpredict)) + self._label_mean
        )
        _pred = _pred.reshape(len(dataset.alphas), dataset.nrow)
        return _pred

    def __predict_available(self) -> None:
        """Check if the model has been fitted before making predictions."""
        if not getattr(self, "_fitted", False):
            raise FittingException("Fit must be executed first.")

    @property
    def feature_importance(self) -> dict[str, Any]:
        self.__predict_available()
        importances: dict[str, Any] = {str(k): 0 for k in self._colnames}
        if self.__is_lgb:
            if not isinstance(self.model, lgb.Booster):
                raise TypeError("model must be a lightgbm Booster")
            _importance = self.model.feature_importance(importance_type="gain").tolist()
            importances.update({str(k): v for k, v in zip(self._colnames, _importance)})
            return importances
        else:
            if not isinstance(self.model, xgb.Booster):
                raise TypeError("model must be a xgboost Booster")
            importances.update(self.model.get_score(importance_type="gain"))
            return importances

    @property
    def __is_lgb(self) -> bool:
        """Check if the model is LightGBM."""
        return self.model_name == ModelName.lightgbm

    @property
    def __is_xgb(self) -> bool:
        """Check if the model is XGBoost."""
        return self.model_name == ModelName.xgboost
