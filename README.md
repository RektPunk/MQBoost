<div style="text-align: center;">
  <img src="https://capsule-render.vercel.app/api?type=transparent&fontColor=0047AB&text=MQBoost&height=120&fontSize=90">
</div>
<p align="center">
  <a href="https://github.com/RektPunk/MQBoost/releases/latest">
    <img alt="release" src="https://img.shields.io/github/v/release/RektPunk/mqboost.svg">
  </a>
<!--   <a href="LICENSE">
    <img alt="license" src="https://img.shields.io/badge/license-MIT-indigo.sv">
  </a> -->
</p>

**MQBoost** introduces an advanced model for estimating multiple quantiles while ensuring the non-crossing condition (monotone quantile condition). This model harnesses the capabilities of both [LightGBM](https://github.com/microsoft/LightGBM) and [XGBoost](https://github.com/dmlc/xgboost), two leading gradient boosting frameworks.

By implementing the hyperparameter optimization prowess of [Optuna](https://github.com/optuna/optuna), the model achieves great performance. Optuna's optimization algorithms fine-tune the hyperparameters, ensuring the model operates efficiently.

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
#--------------------------------------------------------------------------------------------#
>> MQBoost.__init__
x                   # Explanatory data (e.g., pd.DataFrame).
                    # Column named '_tau' must not be included.
y                   # Response data (e.g., np.ndarray).
alphas              # Target quantiles.
                    # Must be in ascending order and contain no duplicates.
objective           # [Optional] Objective to minimize, "check" (default) or "huber".
model               # [Optional] Boosting algorithm to use, "lightgbm" (default) or "xgboost".
delta               # [Optional] Parameter for "huber" objective.
                    # Used only when objective == "huber".
                    # Must be smaller than 0.1.
#--------------------------------------------------------------------------------------------#
>> MQBoost.train
params              # [Optional] Model parameters; defaults to None.
                    # Any params related to model can be used except "objective".
                    # If None, hyperparameter optimization is executed.
n_trials            # [Optional] Number of hyperparameter optimization trials.
                    # Defaults to 20.
#--------------------------------------------------------------------------------------------#
>> MQBoost.predict
x                   # Explanatory data (e.g., pd.DataFrame).
alphas              # [Optional] Target quantiles for prediction.
                    # Defaults to alphas used in train.
#--------------------------------------------------------------------------------------------#
>> MQBoost.optimize_params
n_trials            # Number of hyperparameter optimization trials
get_params_func     # [Optional] Manual hyperparameter function
valid_dict          # [Optional] Manually selected validation sets
                    # Keys must contain "data" and "label"
#--------------------------------------------------------------------------------------------#
```

## Example
See [**Examples**](https://github.com/RektPunk/MQBoost/tree/main/examples)
