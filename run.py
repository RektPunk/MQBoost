import numpy as np
import plotly.graph_objects as go
from module.model import MonotonicQuantileRegressor

if __name__ == "__main__":
    sample_size = 500
    alphas = [0.3, 0.4, 0.5, 0.6, 0.7]
    params = {
        "max_depth": 4,
        "num_leaves": 15,
        "learning_rate": 0.1,
        "n_estimators": 100,
        "boosting_type": "gbdt",
    }
    x = np.linspace(-10, 10, sample_size)
    y = np.sin(x) + np.random.uniform(-0.4, 0.4, sample_size)
    x_test = np.linspace(-10, 10, sample_size)
    y_test = np.sin(x_test) + np.random.uniform(-0.4, 0.4, sample_size)

    monotonic_quantile_regressor = MonotonicQuantileRegressor(
        x=x, y=y_test, alphas=alphas
    )
    model = monotonic_quantile_regressor.train(params=params)
    preds = monotonic_quantile_regressor.predict(x=x_test, alphas=alphas)

    fig = go.Figure(
        go.Scatter(
            x=x_test,
            y=y_test,
            mode="markers",
        )
    )
    for _pred in preds:
        fig.add_trace(go.Scatter(x=x_test, y=_pred, mode="lines"))
    fig.show()
