import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


class MQLabelEncoder:
    def __init__(self) -> None:
        self.label_encoder = LabelEncoder()

    def fit(self, series: pd.Series) -> None:
        self.label_encoder.fit(list(series[~series.isna()]) + ["Unseen", "NaN"])

    def transform(self, series: pd.Series) -> pd.Series:
        return self.label_encoder.transform(
            np.select(
                [series.isna(), ~series.isin(self.label_encoder.classes_)],
                ["NaN", "Unseen"],
                series,
            )
        )

    def fit_transform(self, series: pd.Series) -> pd.Series:
        self.fit(series=series)
        return self.transform(series=series)
