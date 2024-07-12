import numpy as np
import plotly.graph_objects as go

from mqboost import MQRegressor

## Generate sample
sample_size = 500
x = np.linspace(-10, 10, sample_size)
y = np.sin(x) + np.random.uniform(-0.4, 0.4, sample_size)
x_test = np.linspace(-10, 10, sample_size)
y_test = np.sin(x_test) + np.random.uniform(-0.4, 0.4, sample_size)

## target quantiles
alphas = [0.3, 0.4, 0.5, 0.6, 0.7]

### Ex1: LightGBM + check loss
mq_lgb = MQRegressor(
    x=x,
    y=y_test,
    alphas=alphas,
)
lgb_params = {
    "max_depth": 4,
    "num_leaves": 15,
    "learning_rate": 0.1,
    "boosting_type": "gbdt",
}
mq_lgb.train(params=lgb_params)
preds_lgb = mq_lgb.predict(x=x_test, alphas=alphas)

### EX2: XGBoost + check loss
mq_xgb = MQRegressor(
    x=x,
    y=y_test,
    alphas=alphas,
    objective="check",
    model="xgboost",
)
xgb_params = {
    "learning_rate": 0.65,
    "max_depth": 10,
}

mq_xgb.train(params=xgb_params)
preds_xgb = mq_xgb.predict(x=x_test, alphas=alphas)

# Ex3: Lightgbm + huber loss
mq_lgb = MQRegressor(
    x=x,
    y=y_test,
    alphas=alphas,
    objective="huber",
    model="lightgbm",
    delta=0.01,
)
mq_lgb.train(params=lgb_params)
preds_lgb = mq_lgb.predict(x=x_test, alphas=alphas)


### Visualization
lgb_fig = go.Figure(
    go.Scatter(
        x=x_test,
        y=y_test,
        name="test",
        mode="markers",
    )
)
xgb_fig = go.Figure(
    go.Scatter(
        x=x_test,
        y=y_test,
        name="test",
        mode="markers",
    )
)
for _pred_lgb, _pred_xgb, alpha in zip(preds_lgb, preds_xgb, alphas):
    lgb_fig.add_trace(
        go.Scatter(
            x=x_test,
            y=_pred_lgb,
            name=f"{alpha}-quantile",
            mode="lines",
        )
    )
    xgb_fig.add_trace(
        go.Scatter(
            x=x_test,
            y=_pred_xgb,
            name=f"{alpha}-quantile",
            mode="lines",
        )
    )
lgb_fig.update_layout(title="LightGBM Predictions")
lgb_fig.show()
xgb_fig.update_layout(title="XGBoost Predictions")
xgb_fig.show()
