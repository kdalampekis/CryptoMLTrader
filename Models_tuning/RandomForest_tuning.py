from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import numpy as np
import pickle
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import data_splitting


def main():
    # Load the minmax_scaled_filtered dataset
    X_train, y_train = data_splitting.X_train_minmax_filtered, data_splitting.y_train_minmax_filtered
    X_val, y_val = data_splitting.X_val_minmax_filtered, data_splitting.y_val_minmax_filtered
    X_test, y_test = data_splitting.X_test_minmax_filtered, data_splitting.y_test_minmax_filtered

    X_train = X_train.drop(['ds'], axis=1)
    X_val = X_val.drop(['ds'], axis=1)
    X_test = X_test.drop(['ds'], axis=1)

    # Define parameter grid
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
        'max_features': [1.0, 'sqrt', 'log2', 0.5],
    }

    # Initialize the model
    rf = RandomForestRegressor()

    # Initialize the Grid Search model
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, scoring='neg_mean_squared_error', verbose=2, n_jobs=-1)

    # Fit the grid search to the data
    grid_search.fit(X_train, y_train)

    # Print the best parameters found
    print("Best parameters found: ", grid_search.best_params_)

    # Retrieve the best model
    best_rf = grid_search.best_estimator_

    # Save the best model
    model_save_path = '../Trained_Models/best_rf_model.pkl'
    with open(model_save_path, 'wb') as pkl:
        pickle.dump(best_rf, pkl)

    print(f"Model saved to {model_save_path}")

    # Evaluate on the validation set
    best_rf = grid_search.best_estimator_
    val_predictions = best_rf.predict(X_val)
    val_rmse = np.sqrt(mean_squared_error(y_val, val_predictions, multioutput='raw_values'))
    print(f"Validation RMSE: {val_rmse}")

    # Evaluate on the test set
    test_predictions = best_rf.predict(X_test)
    test_rmse = np.sqrt(mean_squared_error(y_test, test_predictions, multioutput='raw_values'))
    print(f"Test RMSE: {test_rmse}")


if __name__ == '__main__':
    main()
