import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
from math import sqrt
import pickle
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import data_splitting

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


def train_evaluate_xgb(X_train, y_train, X_val, y_val, params):
    model = xgb.XGBRegressor(**params)
    model.fit(X_train, y_train)
    y_val_pred = model.predict(X_val)
    val_rmse = sqrt(mean_squared_error(y_val, y_val_pred))
    return val_rmse


def prepare_data(df, target):
    X = df.drop(['ds', target], axis=1).values
    y = df[target].values
    return X, y


def main():
    best_dataset = None
    best_rmse = float('inf')
    params = {'objective': 'reg:squarederror'}  # Default parameters for XGBoost

    for name, (train_df, val_df, test_df) in datasets.items():
        X_train, y_train = prepare_data(train_df, 'y')
        X_val, y_val = prepare_data(val_df, 'y')

        val_rmse = train_evaluate_xgb(X_train, y_train, X_val, y_val, params)
        if val_rmse < best_rmse:
            best_rmse = val_rmse
            best_dataset = name

    print(f"Best dataset: {best_dataset} with RMSE: {best_rmse}")

    param_grid = {
        'max_depth': [3, 5, 7],
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 0.2]
    }

    X_train, y_train = prepare_data(datasets[best_dataset][0], 'y')  # Adjust index as per datasets structure

    grid_search = GridSearchCV(xgb.XGBRegressor(objective='reg:squarederror'),
                               param_grid, scoring='neg_mean_squared_error', cv=5, verbose=2)
    grid_search.fit(X_train, y_train)

    best_params = grid_search.best_params_
    print("Best hyperparameters:", best_params)

    # Saving the best XGBoost model
    best_model = grid_search.best_estimator_
    model_save_path = os.path.join('../Trained_Models', 'best_xgb_model.pkl')
    with open(model_save_path, 'wb') as pkl:
        pickle.dump(best_model, pkl)

    print(f"Model saved to {model_save_path}")

    X_test, y_test = prepare_data(datasets[best_dataset][2], 'y')  # Adjust index as per datasets structure
    best_model = grid_search.best_estimator_
    y_test_pred = best_model.predict(X_test)
    test_rmse = sqrt(mean_squared_error(y_test, y_test_pred))
    print(f"Test RMSE: {test_rmse}")


if __name__ == '__main__':
    main()
