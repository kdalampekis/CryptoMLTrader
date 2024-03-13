from tensorflow import keras
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import data_splitting


# Load the minmax_scaled_filtered dataset
X_train, y_train = data_splitting.train_data_minmax_filtered.drop(['ds', 'y', 'low'], axis=1), data_splitting.train_data_minmax_filtered[['y']]
X_val, y_val = data_splitting.val_data_minmax_filtered.drop(['ds', 'y', 'low'], axis=1), data_splitting.val_data_minmax_filtered[['y']]
X_test, y_test = data_splitting.test_data_minmax_filtered.drop(['ds', 'y', 'low'], axis=1), data_splitting.test_data_minmax_filtered[['y']]

# Reshape input data for LSTM
X_train = X_train.values.reshape((X_train.shape[0], 1, X_train.shape[1]))
X_val = X_val.values.reshape((X_val.shape[0], 1, X_val.shape[1]))
X_test = X_test.values.reshape((X_test.shape[0], 1, X_test.shape[1]))

# Define the model-building function (updated for new hyperparameters)
def build_model(lstm_units=50, num_layers=1, optimizer='adam', learning_rate=0.001, dropout_rate=0.0, activation='relu', regularizer_l1=0.0, regularizer_l2=0.0):
    model = keras.Sequential()
    for i in range(num_layers):
        model.add(keras.layers.LSTM(units=lstm_units, activation=activation, return_sequences=(i < num_layers - 1), input_shape=[1, X_train.shape[2]]))
        model.add(keras.layers.Dropout(dropout_rate))
    model.add(keras.layers.Dense(1, activation=activation, kernel_regularizer=keras.regularizers.l1_l2(l1=regularizer_l1, l2=regularizer_l2)))
    optimizer_instance = keras.optimizers.get(optimizer)
    keras.backend.set_value(optimizer_instance.learning_rate, learning_rate)
    model.compile(optimizer=optimizer_instance, loss='mse')
    return model


def main():

    # Wrap the model with KerasRegressor
    model = keras.wrappers.scikit_learn.KerasRegressor(build_fn=build_model, verbose=1)

    # Define the parameter grid to search
    param_grid = {
        'lstm_units': [30, 50, 100],  # Number of neurons in LSTM layer
        'num_layers': [1, 2, 3],      # Number of LSTM layers
        'optimizer': ['adam', 'sgd'],
        'learning_rate': [0.001, 0.01, 0.1],  # Learning rate
        'dropout_rate': [0.0, 0.2, 0.5],  # Dropout for regularization
        'batch_size': [16, 32, 64],   # Size of each batch
        'epochs': [50, 100],          # Number of epochs
        'activation': ['relu', 'tanh'],  # Activation function
        'regularizer_l1': [0.0, 0.01],  # L1 regularization
        'regularizer_l2': [0.0, 0.01],  # L2 regularization
    }

    # Use TimeSeriesSplit for cross-validation
    tscv = TimeSeriesSplit(n_splits=5)

    # Grid search
    grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=tscv, scoring='neg_mean_squared_error')
    grid_result = grid.fit(X_train, y_train)

    # Summarize results
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))


# Retrain the best model
    best_params = grid_result.best_params_
    best_model = build_model(**best_params)
    best_model.fit(X_train, y_train, epochs=best_params['epochs'], batch_size=best_params['batch_size'], verbose=0)

    # Save the best model
    model_save_path = '../Trained_Models/best_lstm_model.h5'
    best_model.save(model_save_path)

    print(f"Model saved to {model_save_path}")


if __name__ == '__main__':
    main()
