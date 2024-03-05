import sys
import os
from sklearn.metrics import mean_squared_error
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Dense, Conv1D, Flatten
from tensorflow.python.keras.callbacks import EarlyStopping
from math import sqrt
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import data_splitting


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


# Define the CNN model architecture
def create_cnn_model(input_shape, output_units):
    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=1, activation='relu', input_shape=input_shape))
    model.add(Flatten())
    model.add(Dense(50, activation='relu'))
    model.add(Dense(output_units))
    model.compile(optimizer='adam', loss='mse')
    return model


# Train and evaluate CNN model
def train_evaluate_cnn(X_train, y_train, X_val, y_val, output_units):
    input_shape = (X_train.shape[1], X_train.shape[2])
    model = create_cnn_model(input_shape, output_units)

    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_val, y_val), callbacks=[early_stopping], verbose=0)

    val_predictions = model.predict(X_val)
    val_rmse = sqrt(mean_squared_error(y_val, val_predictions))
    return val_rmse


# Main function to evaluate datasets
def main():
    best_dataset = None
    best_avg_val_rmse = float('inf')
    best_dataset_type = None

    for dataset_name, (train_df, val_df, test_df) in datasets.items():

        num_features = train_df.shape[1] - 3  # Adjust based on the number of dropped columns
        X_train = train_df.drop(['ds', 'y', 'low'], axis=1).values.reshape((train_df.shape[0], 1, num_features))
        X_val = val_df.drop(['ds', 'y', 'low'], axis=1).values.reshape((val_df.shape[0], 1, num_features))

        # Assuming 'y' is your target variable for single output and ['y1', 'y2'] for multi-output
        y_train_single = train_df['y'].values
        y_val_single = val_df['y'].values

        y_train_multi = train_df[['y', 'low']].values
        y_val_multi = val_df[['y', 'low']].values

        # Train and evaluate CNN model for single output
        val_rmse_single = train_evaluate_cnn(X_train, y_train_single, X_val, y_val_single, 1)

        # Train and evaluate CNN model for multi output
        val_rmse_multi = train_evaluate_cnn(X_train, y_train_multi, X_val, y_val_multi, 2)

        # Print evaluation metrics for each dataset
        print(f"Dataset: {dataset_name}")
        print(f"  Single Output Val RMSE: {val_rmse_single}")
        print(f"  Multi Output Val RMSE: {val_rmse_multi}")

        # Compare RMSE and determine best dataset and model type
        avg_val_rmse = (val_rmse_single + val_rmse_multi) / 2
        if avg_val_rmse < best_avg_val_rmse:
            best_avg_val_rmse = avg_val_rmse
            best_dataset = dataset_name
            best_dataset_type = 'single_output' if val_rmse_single < val_rmse_multi else 'multi_output'

    print(f"Best dataset: {best_dataset} with {best_dataset_type} model")
    print(f"Best Average Validation RMSE: {best_avg_val_rmse}")


if __name__ == '__main__':
    main()
