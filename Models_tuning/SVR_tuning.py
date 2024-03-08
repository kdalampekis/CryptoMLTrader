from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
from math import sqrt
import pickle
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import data_splitting

# Function to prepare data
def prepare_data(df):
    X = df.drop(['ds', 'y'], axis=1).values  # Adjust according to your dataset
    y = df['y'].values
    return X, y


def main():

    # Preparing the data
    X_train, y_train = prepare_data(data_splitting.train_data_minmax_filtered)
    X_val, y_val = prepare_data(data_splitting.val_data_minmax_filtered)
    X_test, y_test = prepare_data(data_splitting.test_data_minmax_filtered)

    # Hyperparameter grid
    param_grid = {
        'C': [0.1, 1, 10, 100],
        'gamma': ['scale', 'auto'],
        'epsilon': [0.01, 0.1, 1]
    }

    # Grid search with cross-validation
    grid_search = GridSearchCV(SVR(), param_grid, cv=5, scoring='neg_mean_squared_error', verbose=2)
    grid_search.fit(X_train, y_train)

    # Best parameters and model
    best_params = grid_search.best_params_
    best_svr = grid_search.best_estimator_
    print("Best Parameters:", best_params)

    # Save the best model
    model_save_path = '../Trained_Models/best_svr_model.pkl'
    with open(model_save_path, 'wb') as pkl:
        pickle.dump(best_svr, pkl)

    print(f"Model saved to {model_save_path}")

    # Validation and test evaluation
    y_val_pred = best_svr.predict(X_val)
    val_rmse = sqrt(mean_squared_error(y_val, y_val_pred))
    print(f"Validation RMSE: {val_rmse}")

    y_test_pred = best_svr.predict(X_test)
    test_rmse = sqrt(mean_squared_error(y_test, y_test_pred))
    print(f"Test RMSE: {test_rmse}")


if __name__ == '__main__':
    main()
