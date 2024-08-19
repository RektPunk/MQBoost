import numpy as np

from mqboost import MQDataset, MQRegressor

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
objective = "huber"  # Options: "check", "huber", or "approx"
delta = 0.01  # Set when objective is "huber", default is 0.01

# Train the model with fixed parameters
# Initialize the LightGBM-based quantile regressor
lgb_params = {
    "max_depth": 4,
    "num_leaves": 15,
    "learning_rate": 0.1,
    "boosting_type": "gbdt",
}

mq_regressor = MQRegressor(
    params=lgb_params,
    objective=objective,
    model=model,
    delta=delta,
)

# Fit the model
train_dataset = MQDataset(data=x, label=y, alphas=alphas, model=model)
mq_regressor.fit(dataset=train_dataset)

# Predict using the fitted model
test_dataset = MQDataset(data=x_test, alphas=alphas, model=model)
preds_lgb = mq_regressor.predict(test_dataset)
