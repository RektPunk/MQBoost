import numpy as np
import pandas as pd
import pytest

from mqboost.encoder import MQLabelEncoder


# Test data for categorical variables
@pytest.fixture
def sample_data():
    return pd.Series(["apple", "banana", "orange", None, "kiwi", np.nan])


# Test data for label encoding
@pytest.fixture
def sample_label_data():
    return np.array([2, 3, 5, 0, 4, 0])


def test_fit_transform(sample_data):
    encoder = MQLabelEncoder()
    transformed = encoder.fit_transform(sample_data)

    # Check that the transformed result is numeric
    assert transformed is not None
    assert transformed.dtype == int
    assert len(transformed) == len(sample_data)


def test_unseen_and_nan_values(sample_data):
    encoder = MQLabelEncoder()
    encoder.fit(sample_data)

    # Include new unseen value and check behavior
    test_data = pd.Series(["apple", "unknown", None, "melon", np.nan])
    transformed = encoder.transform(test_data)

    # Check for correct handling of unseen and NaN values
    assert (
        transformed
        == encoder.label_encoder.transform(["apple", "Unseen", "NaN", "Unseen", "NaN"])
    ).all()
