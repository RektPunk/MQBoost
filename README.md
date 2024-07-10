# quantile-tree

Multiple quantiles estimation model maintaining non-crossing condition (or monotone quantile condition) using Lightgbm and XGBoost

# Usage
## Features
- **QuantileRegressorLgb**: quantile regressor preserving monotonicity among quantiles based on LightGBM
- **QuantileRegressorXgb**: quantile regressor preserving monotonicity among quantiles based on XGBoost

## Installation
You can install quantile-tree using pip:
```bash
pip install quantile-tree
```

## Example
```python
import numpy as np
from quantile_tree import QuantileRegressorLgb, QuantileRegressorXgb

## Generate sample
sample_size = 500
x = np.linspace(-10, 10, sample_size)
y = np.sin(x) + np.random.uniform(-0.4, 0.4, sample_size)
x_test = np.linspace(-10, 10, sample_size)
y_test = np.sin(x_test) + np.random.uniform(-0.4, 0.4, sample_size)

## target quantiles
alphas = [0.3, 0.4, 0.5, 0.6, 0.7]

## QuantileRegressorLgb
monotonic_quantile_lgb = QuantileRegressorLgb(x=x, y=y_test, alphas=alphas)
lgb_params = {
    "max_depth": 4,
    "num_leaves": 15,
    "learning_rate": 0.1,
    "boosting_type": "gbdt",
}
monotonic_quantile_lgb.train(params=lgb_params)
preds_lgb = monotonic_quantile_lgb.predict(x=x_test, alphas=alphas)

## QuantileRegressorLgb + huber loss (default; check loss)
## delta must be smaller than 0.1
# monotonic_quantile_lgb = QuantileRegressorLgb(x=x, y=y_test, alphas=alphas, objective = "huber", delta = 0.05)
# monotonic_quantile_lgb.train(params=lgb_params)

## QuantileRegressorXgb
monotonic_quantile_xgb = QuantileRegressorXgb(x=x, y=y_test, alphas=alphas)
params = {
    "learning_rate": 0.65,
    "max_depth": 10,
}
monotonic_quantile_xgb.train(params=params)
preds_xgb = monotonic_quantile_xgb.predict(x=x_test, alphas=alphas)

## QuantileRegressorLgb + huber loss (default; check loss)
## delta must be smaller than 0.1
# monotonic_quantile_xgb = QuantileRegressorXgb(x=x, y=y_test, alphas=alphas, objective = "huber", delta = 0.05)
# monotonic_quantile_xgb.train(params=xgb_params)
```

### Visualization
```python
import plotly.graph_objects as go


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
```
