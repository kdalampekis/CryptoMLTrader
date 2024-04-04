import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import data_splitting
import pickle
from tensorflow.python.keras.models import load_model
import xgboost as xgb
from prophet.serialize import model_from_json

# Load Keras models
best_lstm_model = load_model('../Trained_Models/best_lstm_model.h5')
best_cnn_model = load_model('../Trained_Models/best_cnn_model.h5')

# Load scikit-learn and XGBoost models
with open('../Trained_Models/best_svr_model.pkl', 'rb') as file:
    best_svr_model = pickle.load(file)
with open('../Trained_Models/best_rf_model.pkl', 'rb') as file:
    best_rf_model = pickle.load(file)
best_xgb_model = xgb.XGBRegressor()  # Assuming the model was saved
best_xgb_model.load_model('../Trained_Models/best_xgb_model.json')

# Load Prophet models
with open('../Trained_Models/best_prophet_low.pkl', 'rb') as file:
    best_prophet_low = pickle.load(file)

# ARIMA doesn't use .h5 format and might need custom handling based on save/load method
# Example for loading a pickled ARIMA model
with open('../Trained_Models/best_arima_model.pkl', 'rb') as file:
    best_arima_model = pickle.load(file)

# Generate predictions. This part will need to be adjusted based on your actual models and data
predictions_lstm = best_lstm_model.predict(X_test)
predictions_cnn = best_cnn_model.predict(X_test)
predictions_svr = best_svr_model.predict(X_test)
predictions_rf = best_rf_model.predict(X_test)
predictions_xgb = best_xgb_model.predict(X_test)

# Prophet and ARIMA might require their specific input format, so this step is illustrative
predictions_prophet = best_prophet_low.predict(X_test)  # Adjust based on actual usage
predictions_arima = best_arima_model.predict(n_periods=len(X_test))  # Adjust based on actual usage

# Ensemble predictions (make sure all predictions are in a compatible format, e.g., numpy arrays)
ensemble_predictions = (
    0.7 * predictions_lstm +
    0.05 * predictions_arima.reshape(-1, 1) +  # Assuming ARIMA predictions need reshaping
    0.05 * predictions_cnn +
    0.05 * predictions_rf.reshape(-1, 1) +  # Reshape if needed
    0.05 * predictions_svr.reshape(-1, 1) +  # Reshape if needed
    0.05 * predictions_xgb.reshape(-1, 1) +  # Reshape if needed
    0.05 * predictions_prophet['yhat'].values.reshape(-1, 1)  # Assuming 'yhat' is the prediction column
)

