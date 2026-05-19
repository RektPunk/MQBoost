import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest
import xgboost as xgb

from mqboost.base import (
    FittingException,
    ModelName,
    ValidationException,
)
from mqboost.dataset import (
    MQDataset,
    prepare_x,
    prepare_y,
    to_dataframe,
    validate_alpha,
)


# Test for to_dataframe
def test_to_dataframe_with_dataframe():
    x = pd.DataFrame(
        {
            "feature_1": [1, 2],
            "feature_2": [3, 4],
        }
    )
    pd.testing.assert_frame_equal(x, to_dataframe(x))


def test_to_dataframe_with_series():
    x = pd.Series([1, 2, 3])
    expected = pd.DataFrame(
        {
            0: [1, 2, 3],
        }
    )
    pd.testing.assert_frame_equal(expected, to_dataframe(x))


def test_to_dataframe_with_array():
    x = np.array([[1, 2], [3, 4]])
    expected = pd.DataFrame(
        {
            0: [1, 3],
            1: [2, 4],
        }
    )
    pd.testing.assert_frame_equal(expected, to_dataframe(x))


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
    result = prepare_x(to_dataframe(x), alphas)
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
    result = prepare_x(to_dataframe(x), alphas)
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


# Test for validate_alpha
def test_validate_alpha_single_alpha():
    alphas = 0.3
    result = validate_alpha(alphas)
    assert result == [0.3]


def test_validate_alpha_multiple_alphas():
    alphas = [0.1, 0.2, 0.3]
    result = validate_alpha(alphas)
    assert result == alphas


def test_validate_alpha_raises_on_zero_or_one_alpha():
    with pytest.raises(ValidationException, match="Alpha cannot be 0 or 1"):
        validate_alpha([0.0, 0.3])
    with pytest.raises(ValidationException, match="Alpha cannot be 0 or 1"):
        validate_alpha([0.3, 1.0])


def test_validate_alpha_raises_on_non_ascending_alphas():
    with pytest.raises(ValidationException, match="Alpha is not ascending order"):
        validate_alpha([0.3, 0.2, 0.1])


def test_validate_alpha_raises_on_duplicate_alphas():
    with pytest.raises(ValidationException, match="Duplicated alpha exists"):
        validate_alpha([0.1, 0.2, 0.2])


def test_validate_alpha_raises_on_empty_alphas():
    with pytest.raises(ValidationException, match="Input alpha is not valid"):
        validate_alpha([])


def _concat(df: pd.DataFrame, concat_count: int):
    return pd.concat([df] * concat_count, axis=0).reset_index(drop=True)


# Test for MQDataset initialization
def test_mqdataset_initialization_with_lgb():
    data = pd.DataFrame({"feature_1": [1, 2, 3], "feature_2": [4, 5, 6]})
    label = pd.Series([1, 2, 3])
    alphas = [0.1, 0.2, 0.3]
    dataset = MQDataset(
        alphas=alphas, data=data, label=label, model=ModelName.lightgbm.value
    )

    assert dataset.nrow == 3
    assert dataset.alphas == alphas
    pd.testing.assert_frame_equal(
        dataset.data,
        _concat(data, 3).assign(_tau=[0.1, 0.1, 0.1, 0.2, 0.2, 0.2, 0.3, 0.3, 0.3]),
    )
    np.testing.assert_array_equal(
        dataset.label, np.array([-1, 0, 1, -1, 0, 1, -1, 0, 1])
    )


def test_mqdataset_initialization_with_xgb():
    data = pd.DataFrame({"feature_1": [1, 2, 3], "feature_2": [4, 5, 6]})
    label = pd.Series([1, 2, 3])
    alphas = [0.1, 0.2]
    dataset = MQDataset(
        alphas=alphas, data=data, label=label, model=ModelName.xgboost.value
    )

    assert dataset.nrow == 3
    assert dataset.alphas == alphas
    pd.testing.assert_frame_equal(
        dataset.data, _concat(data, 2).assign(_tau=[0.1, 0.1, 0.1, 0.2, 0.2, 0.2])
    )
    np.testing.assert_array_equal(dataset.label, np.array([-1, 0, 1, -1, 0, 1]))


def test_mqdataset_initialization_with_invalid_alpha():
    data = pd.DataFrame({"feature_1": [1, 2, 3], "feature_2": [4, 5, 6]})

    with pytest.raises(ValidationException, match="Alpha is not ascending order"):
        MQDataset(alphas=[0.3, 0.2], data=data)


def test_mqdataset_initialization_without_label():
    data = pd.DataFrame({"feature_1": [1, 2, 3], "feature_2": [4, 5, 6]})
    alphas = [0.1, 0.2]
    dataset = MQDataset(alphas=alphas, data=data, model=ModelName.lightgbm.value)

    assert dataset.nrow == 3
    assert dataset.alphas == alphas

    pd.testing.assert_frame_equal(
        dataset.data, _concat(data, 2).assign(_tau=[0.1, 0.1, 0.1, 0.2, 0.2, 0.2])
    )

    # Ensure label is not available
    with pytest.raises(
        FittingException, match="Fitting is impossible since label is None"
    ):
        dataset.label


# Test properties
def test_mqdataset_train_predict_dtype():
    data = pd.DataFrame({"feature_1": [1, 2, 3], "feature_2": [4, 5, 6]})
    alphas = [0.1, 0.2]
    dataset = MQDataset(alphas=alphas, data=data, model=ModelName.lightgbm.value)
    assert dataset.train_dtype == lgb.Dataset
    pd.testing.assert_frame_equal(dataset.predict_dtype(data), data)

    dataset = MQDataset(alphas=alphas, data=data, model=ModelName.xgboost.value)
    assert dataset.train_dtype == xgb.DMatrix
    assert dataset.predict_dtype == xgb.DMatrix


def test_mqdataset_columns_property():
    data = pd.DataFrame({"feature_1": [1, 2, 3], "feature_2": [4, 5, 6]})
    alphas = [0.1, 0.2]

    dataset = MQDataset(alphas=alphas, data=data, model=ModelName.lightgbm.value)
    assert list(dataset.columns) == [
        "feature_1",
        "feature_2",
        "_tau",
    ]


def test_mqdataset_dtype_lgb():
    data = pd.DataFrame({"feature_1": [1, 2, 3], "feature_2": [4, 5, 6]})
    label = pd.Series([1, 2, 3])
    alphas = [0.1, 0.2]
    dataset = MQDataset(
        alphas=alphas, data=data, label=label, model=ModelName.lightgbm.value
    )

    dtrain = dataset.dtrain
    dpredict = dataset.dpredict
    assert isinstance(dtrain, lgb.Dataset)
    assert isinstance(dpredict, pd.DataFrame)


def test_mqdataset_dtype_xgb():
    data = pd.DataFrame({"feature_1": [1, 2, 3], "feature_2": [4, 5, 6]})
    label = pd.Series([1, 2, 3])
    alphas = [0.1, 0.2]
    dataset = MQDataset(
        alphas=alphas, data=data, label=label, model=ModelName.xgboost.value
    )

    dtrain = dataset.dtrain
    dpredict = dataset.dpredict
    assert isinstance(dtrain, xgb.DMatrix)
    assert isinstance(dpredict, xgb.DMatrix)
