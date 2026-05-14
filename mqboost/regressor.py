from typing import Any

import lightgbm as lgb
import numpy as np
import numpy.typing as npt
import xgboost as xgb

from mqboost.base import FittingException, ModelName, ObjectiveName, ValidationException
from mqboost.constraints import set_monotone_constraints
from mqboost.dataset import MQDataset
from mqboost.objective import MQObjective

__all__ = ["MQRegressor"]


def validate_params(params: dict[str, Any]) -> None:
    """Validate that model parameters do not contain an 'objective' key."""
    if "objective" in params:
        raise ValidationException(
            "The parameter named 'objective' must be excluded in params"
        )


class MQRegressor:
    """Multiple Quantile Regressor using GBDT (LightGBM or XGBoost).

    This regressor implements a multi-quantile estimation strategy by stacking
    the dataset and using monotone constraints on the special '_tau' feature
    to ensure non-crossing quantiles."""

    def __init__(
        self,
        params: dict[str, Any],
        model: str = ModelName.lightgbm.value,
        objective: str = ObjectiveName.check.value,
        epsilon: float = 1e-5,
    ) -> None:
        """Initialize the MQRegressor with specified model parameters and objective."""
        validate_params(params=params)
        self.params = params
        self.model_name = ModelName[model]
        self.objective = ObjectiveName[objective]
        self.epsilon = epsilon

    def fit(
        self,
        dataset: MQDataset,
        eval_set: MQDataset | None = None,
        **kwargs,
    ) -> None:
        """Fit the multi-quantile regressor to the dataset."""
        self._label_mean = dataset.label_mean
        if eval_set:
            eval_set.set_label_mean(self._label_mean)
            eval_set_dtrain = eval_set.dtrain
        else:
            eval_set_dtrain = dataset.dtrain

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
                feval=self.MQObj.lgb_feval,
                valid_sets=[eval_set_dtrain],
                **kwargs,
            )
        elif self.__is_xgb:
            self.model = xgb.train(
                dtrain=dataset.dtrain,
                verbose_eval=False,
                params=params,
                obj=self.MQObj.fobj,
                custom_metric=self.MQObj.xgb_feval,
                evals=[(eval_set_dtrain, "eval")],
                **kwargs,
            )
        self._colnames = dataset.columns.to_list()
        self._fitted = True

    def predict(
        self,
        dataset: MQDataset,
    ) -> npt.NDArray:
        """Predict multiple quantiles for the given dataset."""
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
        """Get feature importance scores from the fitted model."""
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
