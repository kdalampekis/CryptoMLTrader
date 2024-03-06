import sys
import os
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Conv1D, Flatten
from keras.layers import BatchNormalization
from tensorflow.python.keras.callbacks import EarlyStopping
from math import sqrt
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import data_splitting


def create_cnn_model(input_shape, filters, kernel_size, dense_units, output_units):
    model = Sequential()
    model.add(Conv1D(filters=filters, kernel_size=kernel_size, activation='relu', input_shape=input_shape))
    model.add(BatchNormalization())  # Batch normalization layer
    model.add(Flatten())
    model.add(Dense(dense_units, activation='relu'))
    model.add(Dense(output_units))
    model.compile(optimizer='adam', loss='mse')
    return model


def main():

    # Load the minmax_scaled_filtered dataset
    train_data = data_splitting.train_data_minmax_filtered
    val_data = data_splitting.val_data_minmax_filtered
    test_data = data_splitting.test_data_minmax_filtered

    # Assuming 'y' and 'low' are the target variables
    num_features = train_data.drop(['ds', 'y', 'low'], axis=1).shape[1]
    X = train_data.drop(['ds', 'y', 'low'], axis=1).values.reshape((train_data.shape[0], 1, num_features))
    y = train_data[['y', 'low']].values

    # Define hyperparameters to tune
    filters_options = [32, 64]
    kernel_size_options = [1]
    dense_units_options = [50, 100]

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    best_rmse = float('inf')
    best_params = {}

    for filters in filters_options:
        for kernel_size in kernel_size_options:
            for dense_units in dense_units_options:
                rmse_scores = []

                for train_index, val_index in kf.split(X):
                    X_train, X_val = X[train_index], X[val_index]
                    y_train, y_val = y[train_index], y[val_index]

                    model = create_cnn_model((1, num_features), filters, kernel_size, dense_units, 2)
                    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

                    model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_val, y_val),
                              callbacks=[early_stopping], verbose=0)
                    val_predictions = model.predict(X_val)
                    rmse = sqrt(mean_squared_error(y_val, val_predictions))
                    rmse_scores.append(rmse)

                avg_rmse = np.mean(rmse_scores)
                print(f"Filters: {filters}, Kernel Size: {kernel_size}, Dense Units: {dense_units}, Avg RMSE: {avg_rmse}")

                if avg_rmse < best_rmse:
                    best_rmse = avg_rmse
                    best_params = {'filters': filters, 'kernel_size': kernel_size, 'dense_units': dense_units}

    print(f"Best RMSE: {best_rmse}")
    print(f"Best Parameters: {best_params}")


if __name__ == '__main__':
    main()
