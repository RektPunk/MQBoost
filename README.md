# MQBoost

A multiple quantiles estimation model that maintains the non-crossing condition (or monotone quantile condition) based on:
- [LightGBM](https://github.com/microsoft/LightGBM)
- [XGBoost](https://github.com/dmlc/xgboost)

with the hyperparameter optimization framework [Optuna](https://github.com/optuna/optuna).

# Installation
Install using pip:
```bash
pip install mqboost
```

# Usage
## Features
- **MQRegressor**: quantile regressor

## Parameters
```python
x         # Explanatory data (e.g. pd.DataFrame)
          # Column name '_tau' must be not included
y         # Response data (e.g. np.ndarray)
alphas    # Target quantiles
          # It must be in ascending order and not contain duplicates
objective # [Optional] objective to minimize, "check"(default) or "huber"
model     # [Optional] boost algorithm to use, "lightgbm"(default) or "xgboost"
delta     # [Optional] parameter in "huber" objective, only used when objective == "huber"
          # It must be smaller than 0.1
```

## Methods
```python
train           # train quantile model
                # Any params related to model can be used except "objective"
predict         # predict with input data
optimize_params # Optimize hyperparameter with using optuna
```

## Example
```python
import numpy as np
from mqboost import MQRegressor

## Generate sample
sample_size = 500
x = np.linspace(-10, 10, sample_size)
y = np.sin(x) + np.random.uniform(-0.4, 0.4, sample_size)
x_test = np.linspace(-10, 10, sample_size)
y_test = np.sin(x_test) + np.random.uniform(-0.4, 0.4, sample_size)

## target quantiles
alphas = [0.3, 0.4, 0.5, 0.6, 0.7]

## model name
model = "lightgbm" # "xgboost"

## objective funtion
objective = "huber" # "check"
delta = 0.01 # set when objective is huber default 0.05

## LightGBM based quantile regressor
mq_lgb = MQRegressor(
    x=x,
    y=y_test,
    alphas=alphas,
    objective=objective,
    model=model,
    delta=delta,
)

## train with fixed params
lgb_params = {
    "max_depth": 4,
    "num_leaves": 15,
    "learning_rate": 0.1,
    "boosting_type": "gbdt",
}
mq_lgb.train(params=lgb_params)

## train with optuna
mq_lgb.train(n_trials = 10) # the number of trials

## Same process
# best_params = mq_lgb.optimize_params(n_trials = 10)
# mq_lgb.train(params=best_params)

## predict
preds_lgb = mq_lgb.predict(x=x_test, alphas=alphas)
```
