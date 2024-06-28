import numpy as np
import plotly.graph_objects as go
from module.model import QuantileRegressorLgb, QuantileRegressorXgb


if __name__ == "__main__":
    sample_size = 500
    alphas = [0.3, 0.4, 0.5, 0.6, 0.7]
    x = np.linspace(-10, 10, sample_size)
    y = np.sin(x) + np.random.uniform(-0.4, 0.4, sample_size)
    x_test = np.linspace(-10, 10, sample_size)
    y_test = np.sin(x_test) + np.random.uniform(-0.4, 0.4, sample_size)

    monotonic_quantile_lgb = QuantileRegressorLgb(x=x, y=y_test, alphas=alphas)
    lgb_params = {
        "max_depth": 4,
        "num_leaves": 15,
        "learning_rate": 0.1,
        "n_estimators": 100,
        "boosting_type": "gbdt",
    }
    monotonic_quantile_lgb.train(params=lgb_params)
    preds_lgb = monotonic_quantile_lgb.predict(x=x_test, alphas=alphas)

    lgb_fig = go.Figure(
        go.Scatter(
            x=x_test,
            y=y_test,
            mode="markers",
        )
    )
    for _pred in preds_lgb:
        lgb_fig.add_trace(go.Scatter(x=x_test, y=_pred, mode="lines"))
    lgb_fig.show()

    monotonic_quantile_xgb = QuantileRegressorXgb(x=x, y=y_test, alphas=alphas)
    params = {
        "learning_rate": 0.65,
        "max_depth": 10,
    }
    monotonic_quantile_xgb.train(params=params)
    preds = monotonic_quantile_xgb.predict(x=x_test, alphas=alphas)

    xgb_fig = go.Figure(
        go.Scatter(
            x=x_test,
            y=y_test,
            mode="markers",
        )
    )
    for _pred in preds:
        xgb_fig.add_trace(go.Scatter(x=x_test, y=_pred, mode="lines"))
    xgb_fig.show()
