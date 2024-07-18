<div style="text-align: center;">
  <img src="https://capsule-render.vercel.app/api?type=transparent&fontColor=0047AB&text=MQBoost&height=120&fontSize=90">
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
- **MQRegressor**: A model for quantile regression


## Parameters
```python
#-------------------------------------------------------------------------------------------------#
# init
x                 # Explanatory data (e.g., pd.DataFrame)
                  # Column named '_tau' must not be included
y                 # Response data (e.g., np.ndarray)
alphas            # Target quantiles
                  # Must be in ascending order and contain no duplicates
objective         # [Optional] Objective to minimize, "check" (default) or "huber"
model             # [Optional] Boosting algorithm to use, "lightgbm" (default) or "xgboost"
delta             # [Optional] Parameter for "huber" objective; used only when objective == "huber"
                  # Must be smaller than 0.1
#-------------------------------------------------------------------------------------------------#
# train           # train quantile model
                  #  Any params related to model can be used except "objective"

params            # [Optional] Model parameters; defaults to None.
                  # If None, hyperparameter optimization is executed.
n_trials          # [Optional] Number of hyperparameter optimization trials
#-------------------------------------------------------------------------------------------------#
# predict         # predict with input data

x                 # Explanatory data (e.g., pd.DataFrame)
alphas            # Target quantiles for prediction
#-------------------------------------------------------------------------------------------------#
# optimize_params

n_trials          # Number of hyperparameter optimization trials
get_params_func   # Manual hyperparameter function
#-------------------------------------------------------------------------------------------------#
```



## Example
```python
import numpy as np
from optuna import Trial

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
best_params = mq_lgb.optimize_params(n_trials=10)
mq_lgb.train(params=best_params)

# Moreover, you have the option to optimize parameters by implementing functions manually
def get_params(trial: Trial, model: str):
    return {
        "verbose": -1,
        "learning_rate": trial.suggest_float("learning_rate", 1e-2, 1.0, log=True),
        "max_depth": trial.suggest_int("max_depth", 1, 10),
        "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
        "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 2, 256),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.4, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.4, 1.0),
        "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
    }

best_params = mq_lgb.optimize_params(n_trials=10, get_params_func=get_params)
mq_lgb.train(params=best_params)

# Predict using the trained model
preds_lgb = mq_lgb.predict(x=x_test, alphas=alphas)
```
