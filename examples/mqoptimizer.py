import numpy as np
from optuna import Trial

from mqboost import MQDataset, MQOptimizer, MQRegressor

# Generate sample data
sample_size = 500
x = np.linspace(-10, 10, sample_size)
y = np.sin(x) + np.random.uniform(-0.4, 0.4, sample_size)
x_valid = np.linspace(-10, 10, sample_size)
y_valid = np.sin(x_valid) + np.random.uniform(-0.4, 0.4, sample_size)
x_test = np.linspace(-10, 10, sample_size)
y_test = np.sin(x_test) + np.random.uniform(-0.4, 0.4, sample_size)

# Define target quantiles
alphas = [0.3, 0.4, 0.5, 0.6, 0.7]

# Specify model type
model = "lightgbm"  # Options: "lightgbm" or "xgboost"

# Set objective function
objective = "check"  # Options: "huber" or "check"

# Set dataset
train_dataset = MQDataset(data=x, label=y, alphas=alphas, model=model)
valid_dataset = MQDataset(data=x_valid, label=y_valid, alphas=alphas, model=model)
test_dataset = MQDataset(data=x_test, label=y_test, alphas=alphas, model=model)

# Initialize the optimizer
mq_optimizer = MQOptimizer(
    model=model,
    objective=objective,
)

# Optimize params using Optuna
mq_optimizer.optimize_params(
    dataset=train_dataset,
    n_trials=10,
)


# Moreover, you can optimize parameters by implementing functions manually
# Also, you can manually set the validation set
def get_params(trial: Trial):
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


mq_optimizer.optimize_params(
    dataset=train_dataset,
    n_trials=10,
    get_params_func=get_params,
    valid_set=valid_dataset,
)

# Init MQRegressor with best params
mq_regressor = MQRegressor(**mq_optimizer.best_params)
mq_regressor.fit(dataset=train_dataset, eval_set=valid_dataset)

# Predict using the trained model
mq_regressor.predict(dataset=test_dataset)
