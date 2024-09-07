import numpy as np
import pandas as pd
import pytest

from mqboost.base import MQStr, ValidationException
from mqboost.utils import (
    alpha_validate,
    delta_validate,
    params_validate,
    prepare_x,
    prepare_y,
)


# Test for alpha_validate
def test_alpha_validate_single_alpha():
    alphas = 0.3
    result = alpha_validate(alphas)
    assert result == [0.3]


def test_alpha_validate_multiple_alphas():
    alphas = [0.1, 0.2, 0.3]
    result = alpha_validate(alphas)
    assert result == alphas


def test_alpha_validate_raises_on_zero_or_one_alpha():
    with pytest.raises(ValidationException, match="Alpha cannot be 0 or 1"):
        alpha_validate([0.0, 0.3])
    with pytest.raises(ValidationException, match="Alpha cannot be 0 or 1"):
        alpha_validate([0.3, 1.0])


def test_alpha_validate_raises_on_non_ascending_alphas():
    with pytest.raises(ValidationException, match="Alpha is not ascending order"):
        alpha_validate([0.3, 0.2, 0.1])


def test_alpha_validate_raises_on_duplicate_alphas():
    with pytest.raises(ValidationException, match="Duplicated alpha exists"):
        alpha_validate([0.1, 0.2, 0.2])


def test_alpha_validate_raises_on_empty_alphas():
    with pytest.raises(ValidationException, match="Input alpha is not valid"):
        alpha_validate([])


# Test for prepare_x
def test_prepare_x_with_dataframe():
    x = pd.DataFrame(
        {
            "feature_1": [1, 2],
            "feature_2": [3, 4],
        }
    )
    alphas = [0.1, 0.2]
    result = prepare_x(x, alphas)

    expected = pd.DataFrame(
        {
            "feature_1": [1, 2, 1, 2],
            "feature_2": [3, 4, 3, 4],
            "_tau": [0.1, 0.1, 0.2, 0.2],
        }
    )

    pd.testing.assert_frame_equal(result, expected)


def test_prepare_x_with_series():
    x = pd.Series([1, 2, 3])
    alphas = [0.1, 0.2]
    result = prepare_x(x, alphas)

    expected = pd.DataFrame(
        {
            0: [1, 2, 3, 1, 2, 3],
            "_tau": [0.1, 0.1, 0.1, 0.2, 0.2, 0.2],
        }
    )

    pd.testing.assert_frame_equal(result, expected)


def test_prepare_x_with_array():
    x = np.array([[1, 2], [3, 4]])
    alphas = [0.1, 0.2]
    result = prepare_x(x, alphas)

    expected = pd.DataFrame(
        {
            0: [1, 3, 1, 3],
            1: [2, 4, 2, 4],
            "_tau": [0.1, 0.1, 0.2, 0.2],
        }
    )

    pd.testing.assert_frame_equal(result, expected)


def test_prepare_x_raises_on_invalid_column_name():
    x = pd.DataFrame({"_tau": [1, 2], "feature_1": [3, 4]})
    alphas = [0.1, 0.2]

    with pytest.raises(ValidationException, match="Column name '_tau' is not allowed."):
        prepare_x(x, alphas)


# Test for prepare_y
def test_prepare_y_with_list():
    y = [1, 2, 3]
    alphas = [0.1, 0.2]
    result = prepare_y(y, alphas)

    expected = np.array([1, 2, 3, 1, 2, 3])

    np.testing.assert_array_equal(result, expected)


def test_prepare_y_with_array():
    y = np.array([1, 2, 3])
    alphas = [0.1, 0.2]
    result = prepare_y(y, alphas)

    expected = np.array([1, 2, 3, 1, 2, 3])

    np.testing.assert_array_equal(result, expected)


def test_prepare_y_with_series():
    y = pd.Series([1, 2, 3])
    alphas = [0.1, 0.2]
    result = prepare_y(y, alphas)

    expected = np.array([1, 2, 3, 1, 2, 3])

    np.testing.assert_array_equal(result, expected)


# Test for delta_validate
def test_delta_validate_valid_delta():
    delta = 0.04
    result = delta_validate(delta)
    assert result == 0.04


def test_delta_validate_invalid_type():
    with pytest.raises(ValidationException, match="Delta is not float type"):
        delta_validate(1)


def test_delta_validate_negative_delta():
    with pytest.raises(ValidationException, match="Delta must be positive"):
        delta_validate(-0.01)


def test_delta_validate_exceeds_upper_bound():
    delta = 0.06
    with pytest.warns(UserWarning, match="Delta should be 0.05 or less."):
        result = delta_validate(delta)
    assert result == 0.06


# Test for params validate
def test_set_params_validate_raises_validation_exception():
    params = {
        MQStr.obj.value: "regression",
        MQStr.mono.value: [1, -1],
    }
    with pytest.raises(
        ValidationException,
        match="The parameter named 'objective' must be excluded in params",
    ):
        params_validate(params)
