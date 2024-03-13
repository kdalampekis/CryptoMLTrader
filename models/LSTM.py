import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import data_splitting
import numpy as np
from sklearn.metrics import mean_squared_error
from math import sqrt
from tensorflow import keras


# Load datasets
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

# Define target variables
targets = ['y', 'low']  # Adjust as needed

# Initialize a dictionary to store RMSE results
results = {'single_output': {}, 'multi_output': {}}

# Define LSTM model architecture
def create_lstm_model(input_shape):
    model = keras.Sequential()
    model.add(keras.layers.LSTM(units=50, input_shape=input_shape, return_sequences=True))
    model.add(keras.layers.Dense(1))
    return model

# Train and evaluate LSTM model
def train_evaluate_lstm(X_train, y_train, X_val, y_val, X_test, y_test, target):
    input_shape = (X_train.shape[1], X_train.shape[2])
    model = create_lstm_model(input_shape)
    model.compile(optimizer='adam', loss='mse')
    early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    model.fit(X_train, y_train[target], epochs=100, batch_size=32, validation_data=(X_val, y_val[target]), callbacks=[early_stopping], verbose=0)
    val_predictions = model.predict(X_val)
    test_predictions = model.predict(X_test)

    # Reshape predictions to match target dimensions
    val_predictions = val_predictions.squeeze()  # Remove extra dimensions
    test_predictions = test_predictions.squeeze()

    val_rmse = sqrt(mean_squared_error(y_val[target], val_predictions))
    test_rmse = sqrt(mean_squared_error(y_test[target], test_predictions))

    return val_rmse, test_rmse

# Train and evaluate LSTM model for multi-output
def train_evaluate_multi_lstm(X_train, y_train, X_val, y_val, X_test, y_test):
    input_shape = (X_train.shape[1], X_train.shape[2])
    model = create_lstm_model(input_shape)
    model.add(keras.layers.Dense(2))  # For two outputs
    model.compile(optimizer='adam', loss='mse')
    early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_val, y_val), callbacks=[early_stopping],
              verbose=0)
    val_predictions = model.predict(X_val)
    test_predictions = model.predict(X_test)

    # Reshape predictions if necessary
    if val_predictions.ndim > 2:
        val_predictions = val_predictions.squeeze()
    if test_predictions.ndim > 2:
        test_predictions = test_predictions.squeeze()

    # Compute RMSE for each target
    val_rmse = sqrt(mean_squared_error(y_val, val_predictions))
    test_rmse = sqrt(mean_squared_error(y_test, test_predictions))

    return val_rmse, test_rmse

def main():
    best_dataset = None
    best_avg_val_rmse = float('inf')
    best_dataset_type = None  # To track whether single or multi-output is better

    # Loop over datasets
    for dataset_name, (train_df, val_df, test_df) in datasets.items():
        X_train, y_train = train_df.drop(['ds', 'y', 'low'], axis=1), train_df[['y', 'low']]
        X_val, y_val = val_df.drop(['ds', 'y', 'low'], axis=1), val_df[['y', 'low']]
        X_test, y_test = test_df.drop(['ds', 'y', 'low'], axis=1), test_df[['y', 'low']]

        # Reshape input data for LSTM
        X_train = X_train.values.reshape((X_train.shape[0], 1, X_train.shape[1]))
        X_val = X_val.values.reshape((X_val.shape[0], 1, X_val.shape[1]))
        X_test = X_test.values.reshape((X_test.shape[0], 1, X_test.shape[1]))

        # Single-output models
        for target in targets:
            val_rmse, test_rmse = train_evaluate_lstm(X_train, y_train, X_val, y_val, X_test, y_test, target)
            results['single_output'].setdefault(dataset_name, {}).setdefault(target, []).append((val_rmse, test_rmse))

        # Calculate average validation RMSE for single-output models
        avg_val_rmse_single = np.mean([val_rmse for target in targets for val_rmse, _ in results['single_output'][dataset_name].get(target, [])])

        # Compare with best dataset
        if avg_val_rmse_single < best_avg_val_rmse:
            best_avg_val_rmse = avg_val_rmse_single
            best_dataset = dataset_name
            best_dataset_type = 'single_output'

        # Multi-output models
        val_rmse, test_rmse = train_evaluate_multi_lstm(X_train, y_train, X_val, y_val, X_test, y_test)
        results['multi_output'][dataset_name] = [(val_rmse, test_rmse)]

        # Calculate average validation RMSE for multi-output model
        avg_val_rmse_multi = np.mean([val_rmse for val_rmse, _ in results['multi_output'][dataset_name]])

        # Compare with best dataset
        if avg_val_rmse_multi < best_avg_val_rmse:
            best_avg_val_rmse = avg_val_rmse_multi
            best_dataset = dataset_name
            best_dataset_type = 'multi_output'

    # Print results
    print(f"Best dataset: {best_dataset} with {best_dataset_type} model")
    print("Results:")
    for model_type, datasets_results in results.items():
        print(f"Results for {model_type}:")
        for dataset_name, scores in datasets_results.items():
            if model_type == 'single_output':
                for target in targets:
                    avg_val_rmse = np.mean([val_rmse for val_rmse, _ in scores[target]])
                    print(f"\t{dataset_name} ({target}): Avg Validation RMSE = {avg_val_rmse}")
            else:
                avg_val_rmse = np.mean([val_rmse for val_rmse, _ in scores])
                print(f"\t{dataset_name}: Avg Validation RMSE = {avg_val_rmse}")


if __name__ == '__main__':
    main()
