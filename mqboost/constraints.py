from typing import Any, Dict

import pandas as pd

from mqboost.base import FUNC_TYPE, ModelName, MQStr, TypeName, ValidationException


def set_monotone_constraints(
    params: Dict[str, Any],
    columns: pd.Index,
    model_name: ModelName,
) -> Dict[str, Any]:
    """
    Set monotone constraints in params
    Args:
        params (Dict[str, Any])
        columns (pd.Index)
        model_name (ModelName)
    Raises:
        ValidationException: when "objective" is in params.keys()
    Returns:
        Dict[str, Any]
    """
    constraints_fucs = FUNC_TYPE.get(model_name).get(TypeName.constraints_type)
    if MQStr.obj.value in params:
        raise ValidationException(
            "The parameter named 'objective' must be excluded in params"
        )
    _params = params.copy()
    if MQStr.mono.value in _params:
        _monotone_constraints = list(_params[MQStr.mono.value])
        _monotone_constraints.append(1)
        _params.update({MQStr.mono.value: constraints_fucs(_monotone_constraints)})
    else:
        _params.update(
            {
                MQStr.mono.value: constraints_fucs(
                    [1 if "_tau" == col else 0 for col in columns]
                )
            }
        )
    return _params
