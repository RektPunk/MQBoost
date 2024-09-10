<div style="text-align: center;">
  <img src="https://capsule-render.vercel.app/api?type=transparent&fontColor=0047AB&text=MQBoost&height=120&fontSize=90">
</div>
<p align="center">
  <a href="https://github.com/RektPunk/MQBoost/releases/latest">
    <img alt="release" src="https://img.shields.io/github/v/release/RektPunk/mqboost.svg">
  </a>
  <a href="https://pypi.org/project/MQBoost">
    <img alt="Pythonv" src="https://img.shields.io/pypi/pyversions/MQBoost.svg?logo=python&logoColor=white">
  </a>
  <a href="https://github.com/RektPunk/MQBoost/blob/main/LICENSE">
    <img alt="License" src="https://img.shields.io/github/license/RektPunk/MQboost.svg">
  </a>
  <a href="https://github.com/RektPunk/MQBoost/actions/workflows/lint.yaml">
    <img alt="Lint" src="https://github.com/RektPunk/MQBoost/actions/workflows/lint.yaml/badge.svg?branch=main">
  </a>
  <a href="https://github.com/RektPunk/MQBoost/actions/workflows/test.yaml">
    <img alt="Test" src="https://github.com/RektPunk/MQBoost/actions/workflows/test.yaml/badge.svg?branch=main">
  </a>
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
- **MQDataset**: Encapsulates the dataset used for MQRegressor and MQOptimizer.
- **MQRegressor**: Custom multiple quantile estimator with preserving monotonicity among quantiles.
- **MQOptimizer**: Optimize hyperparameters for MQRegressor with Optuna.


## Example
Please refer to the [**Examples**](https://github.com/RektPunk/MQBoost/tree/main/examples) provided for further clarification.
