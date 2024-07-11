# quantile-tree

Non-crossing quantile estimation with:
- LightGBM
- XGBoost

# Installation
Install using pip:
```bash
pip install quantile-tree
```

# Usage
## Features
- **QuantileRegressorLgb**: quantile regressor based on LightGBM
- **QuantileRegressorXgb**: quantile regressor based on XGBoost

## Parameters
```python
x         # Explanatory data (e.g. pd.DataFrame)
          # Column name '_tau' must be not included
y         # Response data (e.g. np.ndarray)
alphas    # Target quantiles
objective # [Optional] objective to minimize, "check"(default) or "huber"
delta     # [Optional] parameter in "huber" objective, used when objective == "huber"
          # delta must be smaller than 0.1
```
## Methods
```python
train     # train quantile model
          # Any params related to model can be used except "objective"
predict   # predict with input data
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
monotonic_quantile_lgb = QuantileRegressorLgb(
    x=x,
    y=y_test,
    alphas=alphas,
    objective="huber",
    delta=0.05,
)
lgb_params = {
    "max_depth": 4,
    "num_leaves": 15,
    "learning_rate": 0.1,
    "boosting_type": "gbdt",
}
monotonic_quantile_lgb.train(params=lgb_params)
preds_lgb = monotonic_quantile_lgb.predict(x=x_test, alphas=alphas)


## QuantileRegressorXgb
monotonic_quantile_xgb = QuantileRegressorXgb(
    x=x,
    y=y_test,
    alphas=alphas
)
params = {
    "learning_rate": 0.65,
    "max_depth": 10,
}
monotonic_quantile_xgb.train(params=params)
preds_xgb = monotonic_quantile_xgb.predict(x=x_test, alphas=alphas)
```
