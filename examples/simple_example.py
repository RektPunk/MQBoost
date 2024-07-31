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
    y=y,
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

# Alternatively, optimize parameters first and then train
best_params = mq_lgb.optimize_params(n_trials=10)
mq_lgb.train(params=best_params)


# Moreover, you can optimize parameters by implementing functions manually
# Also, you can manually set the validation set
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


valid_dict = {
    "data": x_test,
    "label": y_test,
}

best_params = mq_lgb.optimize_params(
    n_trials=10, get_params_func=get_params, valid_dict=valid_dict
)
mq_lgb.train(params=best_params)

# Predict using the trained model
preds_lgb = mq_lgb.predict(x=x_test, alphas=alphas)
