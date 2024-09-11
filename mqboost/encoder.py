import numpy as np
from sklearn.preprocessing import LabelEncoder

from mqboost.base import XdataLike


class MQLabelEncoder:
    def __init__(self) -> None:
        self.label_encoder = LabelEncoder()

    def fit(self, series: XdataLike) -> None:
        self.label_encoder.fit(list(series[~series.isna()]) + ["Unseen", "NaN"])

    def transform(self, series: XdataLike) -> XdataLike:
        return self.label_encoder.transform(
            np.select(
                [series.isna(), ~series.isin(self.label_encoder.classes_)],
                ["NaN", "Unseen"],
                series,
            )
        )

    def fit_transform(self, series: XdataLike) -> XdataLike:
        self.fit(series=series)
        return self.transform(series=series)
