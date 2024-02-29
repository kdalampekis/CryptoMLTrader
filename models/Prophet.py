import pandas as pd
import numpy as np
from ..utils import data_splitting
from fbprophet import Prophet
from sklearn.metrics import mean_squared_error


# Assuming datasets are loaded as follows:
datasets = {
    "original": (data_splitting.train_data, data_splitting.val_data, data_splitting.test_data),
    "standard_scaled": (data_splitting.train_data_standard, data_splitting.val_data_standard, data_splitting.test_data_standard),
    "minmax_scaled": (data_splitting.train_data_minmax, data_splitting.val_data_minmax, data_splitting.test_data_minmax),
    "filtered": (data_splitting.train_data_filtered, data_splitting.val_data_filtered, data_splitting.test_data_filtered),
    "standard_scaled_filtered": (data_splitting.train_data_standard_filtered, data_splitting.val_data_standard_filtered, data_splitting.test_data_standard_filtered),
    "minmax_scaled_filtered": (data_splitting.train_data_minmax_filtered, data_splitting.val_data_minmax_filtered, data_splitting.test_data_minmax_filtered)
}


# Initialize dictionaries to store RMSE for each dataset variant for both targets
rmse_scores_high = {}
rmse_scores_low = {}


# Function to train and evaluate a Prophet model for a specific target
def train_evaluate_prophet(train, test, target):
    df_prophet = pd.DataFrame({
        'ds': train['ds'],
        'y': train[target]
    })

    # Initialize and fit the Prophet model
    model = Prophet()
    model.fit(df_prophet)

    # Create a dataframe for predictions
    future = model.make_future_dataframe(periods=len(test), freq='D')
    forecast = model.predict(future)

    # Evaluate the model
    y_true = test[target].values  # Use the specific target for evaluation
    y_pred = forecast['yhat'][-len(test):].values
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    return rmse


# Loop over each dataset variant
for name, (train, val, test) in datasets.items():
    # Combine train and validation for training, and use test for evaluation
    combined_train_val = pd.concat([train, val])

    # Evaluate for 'high' target
    rmse_high = train_evaluate_prophet(combined_train_val, test, 'y')
    rmse_scores_high[name] = rmse_high

    # Evaluate for 'low' target
    rmse_low = train_evaluate_prophet(combined_train_val, test, 'low')
    rmse_scores_low[name] = rmse_low

    print(f"{name}: RMSE High = {rmse_high}, RMSE Low = {rmse_low}")

# Compare RMSE scores to find the best dataset variant for both targets
best_dataset_high = min(rmse_scores_high, key=rmse_scores_high.get)
best_dataset_low = min(rmse_scores_low, key=rmse_scores_low.get)

print(f"Best dataset variant for High: {best_dataset_high} with RMSE = {rmse_scores_high[best_dataset_high]}")
print(f"Best dataset variant for Low: {best_dataset_low} with RMSE = {rmse_scores_low[best_dataset_low]}")

