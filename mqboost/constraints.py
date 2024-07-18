from typing import Any, Callable, Dict

from mqboost.base import MQStr, ValidationException, XdataLike


def set_monotone_constraints(
    params: Dict[str, Any], x_train: XdataLike, constraints_fucs: Callable
) -> Dict[str, Any]:
    """
    Set monotone constraints in params
    Args:
        params (Dict[str, Any])
    """
    if MQStr.obj in params:
        raise ValidationException(
            "The parameter named 'objective' must not be included in params"
        )
    _params = params.copy()
    if MQStr.mono in _params:
        _monotone_constraints = list(_params[MQStr.mono])
        _monotone_constraints.append(1)
        _params[MQStr.mono] = constraints_fucs(_monotone_constraints)
    else:
        _params.update(
            {
                MQStr.mono: constraints_fucs(
                    [1 if "_tau" == col else 0 for col in x_train.columns]
                )
            }
        )
    return _params
