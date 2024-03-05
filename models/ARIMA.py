import sys
import os
import pandas as pd
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
from math import sqrt


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import data_splitting


# Load and preprocess datasets
datasets = {
    "original": (data_splitting.train_data, data_splitting.val_data, data_splitting.test_data),
    "standard_scaled": (data_splitting.train_data_standard, data_splitting.val_data_standard, data_splitting.test_data_standard),
    "minmax_scaled": (data_splitting.train_data_minmax, data_splitting.val_data_minmax, data_splitting.test_data_minmax),
    "filtered": (data_splitting.train_data_filtered, data_splitting.val_data_filtered, data_splitting.test_data_filtered),
    "standard_scaled_filtered": (data_splitting.train_data_standard_filtered, data_splitting.val_data_standard_filtered, data_splitting.test_data_standard_filtered),
    "minmax_scaled_filtered": (data_splitting.train_data_minmax_filtered, data_splitting.val_data_minmax_filtered, data_splitting.test_data_minmax_filtered)
}

for key, (train_df, val_df, test_df) in datasets.items():
    for df in [train_df, val_df, test_df]:
        if 'ds' in df.columns:
            df['ds'] = pd.to_datetime(df['ds'])
            df.set_index('ds', inplace=True)


# Test stationarity function
def test_stationarity(timeseries):
    dftest = adfuller(timeseries, autolag='AIC')
    return dftest[1]  # p-value


# Testing stationarity on the training part of datasets
for name, (train_df, _, _) in datasets.items():
    p_value = test_stationarity(train_df['y'])
    print(f"{name} Stationarity Test P-Value: {p_value}")


# ARIMA training and evaluation functions
def train_evaluate_arima(train_data, val_data, order):
    model_fit = train_arima_model(train_data, order)
    predictions = model_fit.forecast(steps=len(val_data))
    rmse = sqrt(mean_squared_error(val_data, predictions))
    return rmse


def train_arima_model(train_data, order):
    model = ARIMA(train_data, order=order)
    model_fit = model.fit()
    return model_fit


def main():

    best_rmse = float('inf')
    best_dataset_name = None
    best_order = None

    p_values = range(0, 3)
    d_values = range(0, 2)
    q_values = range(0, 3)

    for dataset_name, (train_df, val_df, test_df) in datasets.items():
        train_data = train_df['y']
        val_data = val_df['y']
        test_data = test_df['y']

        for p in p_values:
            for d in d_values:
                for q in q_values:
                    order = (p, d, q)
                    try:
                        rmse = train_evaluate_arima(train_data, val_data, order)
                        if rmse < best_rmse:
                            best_rmse = rmse
                            best_dataset_name = dataset_name
                            best_order = order
                    except Exception as e:
                        print(f"Error {e} with order {order}")

    # Testing the best model on the test set
    best_train_data = datasets[best_dataset_name][0]['y']
    best_test_data = datasets[best_dataset_name][2]['y']
    final_model = train_arima_model(best_train_data, best_order)
    test_rmse = sqrt(mean_squared_error(best_test_data, final_model.forecast(steps=len(best_test_data))))

    print(f"Best dataset: {best_dataset_name}")
    print(f"Best ARIMA Order: {best_order}")
    print(f"Validation RMSE: {best_rmse}")
    print(f"Test RMSE: {test_rmse}")


if __name__ == '__main__':
    main()
