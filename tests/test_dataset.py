import numpy as np
import pandas as pd
import pytest

from mqboost.base import FittingException, ModelName, ValidationException
from mqboost.dataset import MQDataset


def _concat(df: pd.DataFrame, concat_count: int):
    return pd.concat([df] * concat_count, axis=0).reset_index(drop=True)


# Test for MQDataset initialization
def test_mqdataset_initialization_with_lightgbm():
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


def test_mqdataset_initialization_with_xgboost():
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


def test_mqdataset_columns_property():
    data = pd.DataFrame({"feature_1": [1, 2, 3], "feature_2": [4, 5, 6]})
    alphas = [0.1, 0.2]

    dataset = MQDataset(alphas=alphas, data=data, model=ModelName.lightgbm.value)
    assert list(dataset.columns) == [
        "feature_1",
        "feature_2",
        "_tau",
    ]


def test_mqdataset_dtrain():
    data = pd.DataFrame({"feature_1": [1, 2, 3], "feature_2": [4, 5, 6]})
    label = pd.Series([1, 2, 3])
    alphas = [0.1, 0.2]
    dataset = MQDataset(
        alphas=alphas, data=data, label=label, model=ModelName.lightgbm.value
    )

    dtrain = dataset.dtrain
    assert isinstance(dtrain, dataset.train_dtype)


def test_mqdataset_dpredict():
    data = pd.DataFrame({"feature_1": [1, 2, 3], "feature_2": [4, 5, 6]})
    alphas = [0.1, 0.2]
    dataset = MQDataset(alphas=alphas, data=data, model=ModelName.xgboost.value)

    dpredict = dataset.dpredict
    assert isinstance(dpredict, dataset.predict_dtype)
