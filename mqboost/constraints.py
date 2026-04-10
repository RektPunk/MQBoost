from typing import Any

import pandas as pd

from mqboost.base import FUNC_TYPE, ModelName, TypeName


def set_monotone_constraints(
    params: dict[str, Any],
    columns: pd.Index,
    model_name: ModelName,
) -> dict[str, Any]:
    """Set monotone constraints in params"""
    MONOTONE_CONSTRAINTS: str = "monotone_constraints"

    constraints_fucs = FUNC_TYPE[model_name][TypeName.constraints_type]
    _params = params.copy()
    if MONOTONE_CONSTRAINTS in _params:
        _monotone_constraints = _params.get(MONOTONE_CONSTRAINTS)
        if not isinstance(_monotone_constraints, list):
            raise TypeError(f"{MONOTONE_CONSTRAINTS} must be a list")

        _monotone_constraints.append(1)
        _params.update({MONOTONE_CONSTRAINTS: constraints_fucs(_monotone_constraints)})
    else:
        _params.update(
            {
                MONOTONE_CONSTRAINTS: constraints_fucs(
                    [1 if "_tau" == col else 0 for col in columns]
                )
            }
        )
    return _params
