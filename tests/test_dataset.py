import numpy as np
import pandas as pd
import pytest

from mqboost.base import FittingException, ModelName, ValidationException
from mqboost.dataset import MQDataset
from mqboost.encoder import MQLabelEncoder


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
    np.testing.assert_array_equal(dataset.label, np.concatenate([label] * len(alphas)))


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
    np.testing.assert_array_equal(dataset.label, np.concatenate([label] * len(alphas)))


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
    assert dataset.train_dtype == dataset._train_dtype
    assert dataset.predict_dtype == dataset._predict_dtype

    dataset = MQDataset(alphas=alphas, data=data, model=ModelName.xgboost.value)
    assert dataset.train_dtype == dataset._train_dtype
    assert dataset.predict_dtype == dataset._predict_dtype


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
    assert isinstance(dtrain, dataset.train_dtype)
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
    assert isinstance(dtrain, dataset.train_dtype)
    assert isinstance(dpredict, dataset.predict_dtype)


def test_MQDataset_reference():
    data = pd.DataFrame(
        {
            "col1": ["A", "B", "C"],
            "col2": [1, 2, 3],
            "col3": ["2", "3", "1"],
        }
    )
    label = pd.Series([0, 1, 0])
    alphas = [0.1, 0.2]
    dataset = MQDataset(data=data, label=label, alphas=alphas)

    assert isinstance(dataset.encoders["col1"], MQLabelEncoder)
    assert isinstance(dataset.encoders["col3"], MQLabelEncoder)
    transformed_data = pd.DataFrame(
        {
            "col1": [0, 1, 2] * 2,
            "col2": [1, 2, 3] * 2,
            "col3": [1, 2, 0] * 2,
            "_tau": [0.1, 0.1, 0.1, 0.2, 0.2, 0.2],
        }
    )
    pd.testing.assert_frame_equal(dataset.data, transformed_data)
    new_data = pd.DataFrame(
        {
            "col1": ["A", "C", "B"],
            "col2": [1, 3, 2],
            "col3": ["X", "Y", "X"],
        }
    )

    new_dataset = MQDataset(data=new_data, alphas=alphas, reference=dataset)
    transformed_new_data = pd.DataFrame(
        {
            "col1": [0, 2, 1] * 2,
            "col2": [1, 3, 2] * 2,
            "col3": [4, 4, 4] * 2,
            "_tau": [0.1, 0.1, 0.1, 0.2, 0.2, 0.2],
        }
    )
    print(new_dataset.data)
    pd.testing.assert_frame_equal(new_dataset.data, transformed_new_data)
    assert new_dataset.encoders == dataset.encoders


def test_MQDataset_reference_with_missing_columns():
    data = pd.DataFrame(
        {
            "col1": ["A", "B", "C"],
            "col2": [1, 2, 3],
            "col3": ["2", "3", "1"],
        }
    )
    label = pd.Series([0, 1, 0])
    alphas = [0.1, 0.2]
    dataset = MQDataset(data=data, label=label, alphas=alphas)

    new_data = pd.DataFrame(
        {
            "col1": ["A", "C"],
            "col2": [1, 3],  # col3 is missing
        }
    )

    with pytest.raises(ValueError):
        MQDataset(data=new_data, alphas=alphas, reference=dataset)
