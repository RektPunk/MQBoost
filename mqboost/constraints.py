from typing import Any

import pandas as pd

from mqboost.base import FUNC_TYPE, ModelName, TypeName


def set_monotone_constraints(
    params: dict[str, Any],
    columns: pd.Index,
    model_name: ModelName,
) -> dict[str, Any]:
    """Configure monotone constraints for the GBDT model.

    To ensure that predicted quantiles are non-decreasing with respect to the
    quantile level (alpha), a monotone constraint of '1' is applied to the
    special '_tau' feature.
    """
    MONOTONE_CONSTRAINTS: str = "monotone_constraints"

    constraints_funcs = FUNC_TYPE[model_name][TypeName.constraints_type]
    _params = params.copy()
    num_columns = len(columns)

    if MONOTONE_CONSTRAINTS in _params:
        _monotone_constraints = _params.get(MONOTONE_CONSTRAINTS)
        if not isinstance(_monotone_constraints, list):
            raise TypeError(f"{MONOTONE_CONSTRAINTS} must be a list")

        # If user provided constraints for all columns including _tau
        if len(_monotone_constraints) == num_columns:
            pass
        # If user provided constraints for original columns only
        elif len(_monotone_constraints) == num_columns - 1:
            _monotone_constraints.append(1)
        else:
            raise ValueError(
                f"Length of {MONOTONE_CONSTRAINTS} must be {num_columns} or {num_columns - 1}"
            )

        _params.update({MONOTONE_CONSTRAINTS: constraints_funcs(_monotone_constraints)})
    else:
        # Default: only _tau is monotonic (1)
        _constraints = [1 if col == "_tau" else 0 for col in columns]
        _params.update({MONOTONE_CONSTRAINTS: constraints_funcs(_constraints)})

    return _params
