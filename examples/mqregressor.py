"""These following example show how to set up and use `mqboost` for quantile regression with fixed parameters.
Adjust the parameters and settings as needed for your specific use case.
To use the code, make sure you have `mqboost` and other required dependencies installed."""

# import matplotlib.pyplot as plt
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
objective = "check"  # Options: "check", "approx", or "huber"
# epsilon = 1e-5  # Set when objective is "approx" or "huber", default is 1e-5

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
)

# Fit the model
train_dataset = MQDataset(data=x, label=y, alphas=alphas, model=model)
mq_regressor.fit(dataset=train_dataset)

# Predict using the fitted model
test_dataset = MQDataset(data=x_test, alphas=alphas, model=model)
preds_lgb = mq_regressor.predict(test_dataset)

# # For visualization of predictions vs. actual values
# plt.figure(figsize=(12, 6))
# plt.scatter(x_test, y_test, label="Actual y_test", alpha=0.5, s=10)

# # Plot each quantile prediction
# for i, alpha in enumerate(alphas):
#     plt.plot(x_test, preds_lgb[i, :], label=f"Quantile {alpha}")

# plt.title("Quantile Regression Predictions vs. Actual Values")
# plt.xlabel("x_test")
# plt.ylabel("y_values")
# plt.legend()
# plt.grid(True)
# plt.show()
