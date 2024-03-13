from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
from math import sqrt
import pandas as pd
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import data_splitting
import pickle


# Assuming 'y' is the target variable

# Load the minmax_scaled dataset or the dataset you found to be the best
train_data = data_splitting.train_data_minmax['y']
val_data = data_splitting.val_data_minmax['y']
test_data = data_splitting.test_data_minmax['y']


# Function to train and evaluate ARIMA model
def train_evaluate_arima(train_data, val_data, order):
    model = ARIMA(train_data, order=order)
    model_fit = model.fit()
    predictions = model_fit.forecast(steps=len(val_data))
    rmse = sqrt(mean_squared_error(val_data, predictions))
    return rmse


# Function to perform a grid search over ARIMA parameters
def grid_search_arima(train_data, val_data, p_values, d_values, q_values):
    best_score, best_cfg = float("inf"), None
    for p in p_values:
        for d in d_values:
            for q in q_values:
                order = (p, d, q)
                try:
                    rmse = train_evaluate_arima(train_data, val_data, order)
                    if rmse < best_score:
                        best_score, best_cfg = rmse, order
                    print('ARIMA%s RMSE=%.3f' % (order,rmse))
                except:
                    continue
    return best_cfg, best_score


def main():

    # Hyperparameters ranges
    p_values = range(0, 3)
    d_values = range(0, 2)
    q_values = range(0, 3)

    # Grid search
    # Grid search
    best_cfg, best_score = grid_search_arima(train_data, val_data, p_values, d_values, q_values)
    print('Best ARIMA%s RMSE=%.3f' % (best_cfg, best_score))

    # Train the best model on the combined train and validation data
    combined_data = pd.concat([train_data, val_data])  # Use pd.concat to combine Series
    model_fit = ARIMA(combined_data, order=best_cfg).fit()

    # Evaluate on the test set
    predictions = model_fit.forecast(steps=len(test_data))
    test_rmse = sqrt(mean_squared_error(test_data, predictions))
    print('Test RMSE: %.3f' % test_rmse)

    # Save the best model
    model_save_path = os.path.join('../Trained_Models', 'best_arima_model.pkl')
    with open(model_save_path, 'wb') as pkl:
        pickle.dump(model_fit, pkl)

    print(f"Model saved to {model_save_path}")


if __name__ == '__main__':
    main()
