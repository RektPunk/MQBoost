<div style="text-align: center;">
  <img src="https://capsule-render.vercel.app/api?type=transparent&fontColor=0047AB&text=MQBoost&height=120&fontSize=90">
</div>

**MQBoost** is a gradient boosting-based framework for simultaneous multi-quantile regression with monotonicity constraints (non-crossing quantiles).
It is built on top of  [LightGBM](https://github.com/microsoft/LightGBM) and [XGBoost](https://github.com/dmlc/xgboost), two leading gradient boosting frameworks, enabling efficient and scalable training while ensuring valid quantile estimates.

Standard quantile regression models often suffer from Quantile crossing (e.g., 90% quantile < 50% quantile) and independent training per quantile → inconsistent predictions. **MQBoost** solves this by:
- Learning multiple quantiles jointly
- Enforcing monotonicity across quantiles
- Leveraging efficient boosting frameworks

# Installation
Install using pip:
```bash
pip install mqboost
```

# Usage
## Features
- **MQDataset**: Encapsulates the dataset used for MQRegressor.
- **MQRegressor**: Custom multiple quantile estimator with preserving monotonicity among quantiles.

## Example
Please refer to the [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/RektPunk/MQBoost/blob/main/examples/mqregressor.ipynb) or [**Examples**](https://github.com/RektPunk/MQBoost/tree/main/examples/mqregressor.py) provided for further clarification.

# Citation
If you use MQBoost in your research or project, please cite it as follows:
```md
@article{Moon2026,
    title={Monotone Composite Quantile Regression via Second-Order Gradient Boosting Framework},
    author={Moon, Sangjun and Hong, Sungchul and Park, Beomjin},
    journal={Machine Learning},
    volume={115},
    number={6},
    pages={127},
    year={2026},
    month={may},
    issn={1573-0565},
    doi={10.1007/s10994-026-07058-2},
    url={https://doi.org/10.1007/s10994-026-07058-2}
}
```
