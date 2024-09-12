import pandas as pd

from mqboost.base import FUNC_TYPE, ModelName, ParamsLike, TypeName


def set_monotone_constraints(
    params: ParamsLike,
    columns: pd.Index,
    model_name: ModelName,
) -> ParamsLike:
    """
    Set monotone constraints in params
    Args:
        params (ParamsLike)
        columns (pd.Index)
        model_name (ModelName)
    Raises:
        ValidationException: when "objective" is in params.keys()
    Returns:
        ParamsLike
    """
    MONOTONE_CONSTRAINTS: str = "monotone_constraints"

    constraints_fucs = FUNC_TYPE.get(model_name).get(TypeName.constraints_type)
    _params = params.copy()
    if MONOTONE_CONSTRAINTS in _params:
        _monotone_constraints = list(_params[MONOTONE_CONSTRAINTS])
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
