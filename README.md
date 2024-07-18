<div style="text-align: center;">
  <img src="https://capsule-render.vercel.app/api?type=transparent&fontColor=ffffff&text=MQBoost&height=120&fontSize=90">
</div>
<p align="center">
  <a href="https://github.com/RektPunk/MQBoost/releases/latest">
    <img alt="release" src="https://img.shields.io/github/v/release/RektPunk/mqboost.svg">
  </a>
  <a href="https://pypi.python.org/pypi/mqboost/">
    <img alt="PyPI" src="https://badge.fury.io/py/mqboost.svg">
  </a>
<!--   <a href="LICENSE">
    <img alt="license" src="https://img.shields.io/badge/license-MIT-indigo.sv">
  </a> -->
</p>

**MQBoost** introduces an advanced model for estimating multiple quantiles while ensuring the non-crossing condition (monotone quantile condition). This model harnesses the capabilities of both [LightGBM](https://github.com/microsoft/LightGBM) and [XGBoost](https://github.com/dmlc/xgboost), two leading gradient boosting frameworks.

By implementing the hyperparameter optimization prowess of [Optuna](https://github.com/optuna/optuna), this model achieves great performance and precision. Optuna's optimization algorithms fine-tune the hyperparameters, ensuring the model operates efficiently.

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

# Generate sample data
sample_size = 500
x = np.linspace(-10, 10, sample_size)
y = np.sin(x) + np.random.uniform(-0.4, 0.4, sample_size)
x_test = np.linspace(-10, 10, sample_size)
y_test = np.sin(x_test) + np.random.uniform(-0.4, 0.4, sample_size)

# Define target quantiles
alphas = [0.3, 0.4, 0.5, 0.6, 0.7]

# Specify model type
model = "lightgbm"  # Options: "lightgbm" or "xgboost"

# Set objective function
objective = "huber"  # Options: "huber" or "check"
delta = 0.01  # Set when objective is "huber", default is 0.05

# Initialize the LightGBM-based quantile regressor
mq_lgb = MQRegressor(
    x=x,
    y=y_test,
    alphas=alphas,
    objective=objective,
    model=model,
    delta=delta,
)

# Train the model with fixed parameters
lgb_params = {
    "max_depth": 4,
    "num_leaves": 15,
    "learning_rate": 0.1,
    "boosting_type": "gbdt",
}
mq_lgb.train(params=lgb_params)

# Train the model with Optuna hyperparameter optimization
mq_lgb.train(n_trials=10)
# Alternatively, you can optimize parameters first and then train
# best_params = mq_lgb.optimize_params(n_trials=10)
# mq_lgb.train(params=best_params)

# Predict using the trained model
preds_lgb = mq_lgb.predict(x=x_test, alphas=alphas)
```
