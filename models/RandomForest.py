import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import data_splitting
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from math import sqrt
import numpy as np

# Load your datasets
datasets = {
    "original": (data_splitting.train_data, data_splitting.val_data, data_splitting.test_data),
    "standard_scaled": (
        data_splitting.train_data_standard, data_splitting.val_data_standard, data_splitting.test_data_standard),
    "minmax_scaled": (
        data_splitting.train_data_minmax, data_splitting.val_data_minmax, data_splitting.test_data_minmax),
    "filtered": (
        data_splitting.train_data_filtered, data_splitting.val_data_filtered, data_splitting.test_data_filtered),
    "standard_scaled_filtered": (data_splitting.train_data_standard_filtered, data_splitting.val_data_standard_filtered,
                                 data_splitting.test_data_standard_filtered),
    "minmax_scaled_filtered": (data_splitting.train_data_minmax_filtered, data_splitting.val_data_minmax_filtered,
                               data_splitting.test_data_minmax_filtered)
}

# Define your target variables
targets = ['y', 'low']  # Adjust as needed

# Initialize a dictionary to store RMSE results
results = {'single_output': {}, 'multi_output': {}}

# Iterate over datasets
for dataset_name, dataset in datasets.items():
    train_df, val_df, test_df = dataset  # Unpack the dataset tuple

    # Prepare feature and target matrices
    X_train = train_df.drop(['ds', 'y', 'low'], axis=1)
    y_train = train_df[targets]
    X_test = test_df.drop(['ds', 'y', 'low'], axis=1)
    y_test = test_df[targets]

    # Single-output models
    for target in targets:
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train[target])
        predictions = model.predict(X_test)
        rmse = sqrt(mean_squared_error(y_test[target], predictions))
        results['single_output'].setdefault(dataset_name, {})[target] = rmse

    # Multi-output model
    multi_output_model = RandomForestRegressor(n_estimators=100, random_state=42)
    multi_output_model.fit(X_train, y_train)
    predictions = multi_output_model.predict(X_test)

    # Calculate RMSE for each target separately and store the results
    rmse_scores = np.sqrt(mean_squared_error(y_test, predictions, multioutput='raw_values'))
    results['multi_output'][dataset_name] = dict(zip(targets, rmse_scores))

# Display results
for model_type, datasets_results in results.items():
    print(f"Results for {model_type}:")
    for dataset_name, scores in datasets_results.items():
        print(f"\t{dataset_name}: {scores}")
