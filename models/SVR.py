import numpy as np
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import mean_squared_error
from math import sqrt
import sys
import os
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


# Example of preparing data for SVR
def prepare_data(df):
    X = df.drop(['ds', 'y'], axis=1).values  # Drop non-feature columns
    y = df['y'].values  # Target variable
    return X, y


prepared_datasets = {name: prepare_data(df) for name, (df, _, _) in datasets.items()}


def train_evaluate_svr(X_train, y_train, X_val, y_val, params):
    model = SVR(**params)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    rmse = sqrt(mean_squared_error(y_val, y_pred))
    return rmse


best_rmse = float('inf')
best_dataset = None
params = {'C': 1.0, 'epsilon': 0.1}  # Default parameters


def main():
    global best_rmse, best_dataset  # to modify global variables

    for name, (df_train, df_val, _) in datasets.items():
        # Prepare the data
        X_train, y_train = prepare_data(df_train)
        X_val, y_val = prepare_data(df_val)

        # Train and evaluate the SVR model
        rmse = train_evaluate_svr(X_train, y_train, X_val, y_val, params)
        print(f"Dataset {name} - RMSE: {rmse}")

        if rmse < best_rmse:
            best_rmse = rmse
            best_dataset = name

    print(f"Best Dataset: {best_dataset} with RMSE: {best_rmse}")


if __name__ == '__main__':
    main()
