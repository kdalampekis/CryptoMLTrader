from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.callbacks import EarlyStopping
import itertools
import warnings
from prophet import Prophet
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
import pandas as pd
import numpy as np
import talib
from statsmodels.tsa.seasonal import STL
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt


def seasonal_decomposition(df):
    df.set_index('ds', inplace=True)

    # Perform seasonal decomposition
    decomposition = STL(df['close'], seasonal=13).fit()

    # Extract trend, seasonal, and residual components
    trend = decomposition.trend
    seasonal = decomposition.seasonal
    residual = decomposition.resid

    # Remove seasonal component from the original data
    df['close_detrended'] = df['close'] - seasonal
    df.reset_index(inplace=True)

    return df


def calculate_technical_indicators(df, sma_window=20, ema_short=12, ema_long=26, rsi_window=14):
    # Calculate moving averages (SMA)
    df['SMA_' + str(sma_window)] = df['close'].rolling(window=sma_window).mean()

    # Calculate exponential moving averages (EMA)
    df['EMA_' + str(ema_short)] = df['close'].ewm(span=ema_short, adjust=False).mean()
    df['EMA_' + str(ema_long)] = df['close'].ewm(span=ema_long, adjust=False).mean()

    # Calculate RSI
    delta = df['close'].diff()
    gain = delta.clip(lower=0).rolling(window=rsi_window).mean()
    loss = -delta.clip(upper=0).rolling(window=rsi_window).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # Calculate Bollinger Bands
    rolling_mean = df['close'].rolling(window=sma_window).mean()
    rolling_std = df['close'].rolling(window=sma_window).std()
    df['BB_upper'] = rolling_mean + (rolling_std * 2)
    df['BB_lower'] = rolling_mean - (rolling_std * 2)

    df = calculate_additional_features(df)

    return df


def calculate_additional_features(df):
    # Price Rate of Change (ROC)
    roc_periods = [1, 7, 30]
    for period in roc_periods:
        df[f'ROC_{period}'] = (df['close'].pct_change(periods=period) * 100).fillna(0)

    # Volume Rate of Change (VROC)
    for period in roc_periods:
        df[f'VROC_{period}'] = (df['volumefrom'].pct_change(periods=period) * 100).fillna(0)

    # Price Momentum Oscillator (PMO)
    df['PMO_12_26'] = talib.EMA(df['close'], timeperiod=12) - talib.EMA(df['close'], timeperiod=26)

    # On-Balance Volume (OBV)
    df['OBV'] = (np.sign(df['close'].diff()) * df['volumefrom']).cumsum()

    # Price and Volume Trend (PVT)
    df['PVT'] = df['close'].diff() / df['close'].shift()

    # Relative Volatility Index (RVI)
    up = df['close'].diff()
    down = -up
    df['RVI'] = talib.EMA(up, timeperiod=14) / talib.EMA(down, timeperiod=14)

    # Chaikin Money Flow (CMF)
    df['CMF'] = talib.ADOSC(df['y'], df['low'], df['close'], df['volumefrom'], fastperiod=3, slowperiod=10)

    # Average True Range (ATR)
    df['ATR'] = talib.ATR(df['y'], df['low'], df['close'], timeperiod=14)

    # Price Oscillators
    df['Stochastic_Oscillator'] = talib.STOCH(df['y'], df['low'], df['close'])[1]
    macd, signal, hist = talib.MACD(df['close'])
    df['MACD_Histogram'] = hist

    # Lag features
    lag_periods = [1, 7, 30]  # Lag by 1 day, 1 week, and 1 month
    for period in lag_periods:
        df[f'close_lag_{period}'] = df['close'].shift(periods=period)

    # Ratio features
    df['close_to_volume_ratio'] = df['close'] / df['volumefrom']

    # Statistical features
    df['close_mean'] = df['close'].rolling(window=20).mean()
    df['close_std'] = df['close'].rolling(window=20).std()

    # Date-based features
    df['ds'] = pd.to_datetime(df['ds'])
    df['year'] = df['ds'].dt.year
    df['month'] = df['ds'].dt.month
    df['day'] = df['ds'].dt.day
    df['day_of_week'] = df['ds'].dt.dayofweek
    df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'])
    df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'])

    df = seasonal_decomposition(df)

    return df


def preprocess_data(df):

    # Handle missing values by filling with the mean of each column
    df.fillna(df.mean(), inplace=True)

    # Remove outliers: remove rows with close prices beyond three standard deviations from the mean
    mean_close = df['close'].mean()
    std_close = df['close'].std()
    df = df[abs(df['close'] - mean_close) < 3 * std_close]

    # Perform feature scaling: normalize the 'volumefrom' column
    df['volumefrom'] = (df['volumefrom'] - df['volumefrom'].min()) / (df['volumefrom'].max() - df['volumefrom'].min())

    return df


pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

# Load historical data from the CSV file
csv_file_path = 'Solana_20_2_2023-20_2_2024_historical_data_coinmarketcap.csv'
df_csv = pd.read_csv(csv_file_path, delimiter=';', parse_dates=['timeOpen', 'timeClose', 'timeHigh', 'timeLow'])
df_csv.rename(columns={
        'timestamp': 'ds',
        'open': 'open',
        'high': 'y',
        'low': 'low',
        'close': 'close',
        'volume': 'volumefrom',
        'marketCap': 'volumeto'
    }, inplace=True)
df_csv.drop(columns=['timeOpen', 'timeClose'], inplace=True)
df_csv['ds'] = pd.to_datetime(df_csv['ds'])
df_csv['date'] = df_csv['ds'].dt.date
df_csv.drop(columns=['ds'], inplace=True)
df_csv.rename(columns={'date': 'ds'}, inplace=True)
df_csv['ds'] = pd.to_datetime(df_csv['ds'])
df_csv['timeLow'] = pd.to_datetime(df_csv['timeLow'])
df_csv['timeHigh'] = pd.to_datetime(df_csv['timeHigh'])

# Extract relevant features
df_csv['hour_low'] = df_csv['timeLow'].dt.hour
df_csv['minute_low'] = df_csv['timeLow'].dt.minute
df_csv['second_low'] = df_csv['timeLow'].dt.second

df_csv['hour_high'] = df_csv['timeHigh'].dt.hour
df_csv['minute_high'] = df_csv['timeHigh'].dt.minute
df_csv['second_high'] = df_csv['timeHigh'].dt.second
df_csv.drop(columns=['timeLow', 'timeHigh'], inplace=True)


# Extract relevant features
df_csv = calculate_technical_indicators(df_csv)
df_csv = preprocess_data(df_csv)
df_csv_sorted = df_csv.sort_values(by='ds', ascending=True)
df_csv.drop(columns=['day_of_week'], inplace=True)
df_csv_standard = df_csv_sorted.copy()
df_csv_minmax = df_csv_sorted.copy()
df_csv_help1 = df_csv_sorted.copy()
df_csv_help2 = df_csv_sorted.copy()

features_to_scale = ['open', 'y', 'low', 'close', 'volumefrom', 'volumeto', 'SMA_20', 'EMA_12', 'EMA_26',
                     'RSI', 'BB_upper', 'BB_lower', 'ROC_1', 'ROC_7', 'ROC_30', 'VROC_1', 'VROC_7', 'VROC_30',
                     'PMO_12_26', 'OBV', 'PVT', 'RVI', 'CMF', 'ATR', 'Stochastic_Oscillator', 'MACD_Histogram',
                     'close_lag_1', 'close_lag_7', 'close_lag_30', 'close_to_volume_ratio', 'close_mean',
                     'close_std', 'close_detrended', 'year', 'month', 'day', 'day_of_week_cos', 'day_of_week_sin', 'hour_low', 'minute_low', 'second_low', 'hour_high', 'minute_high', 'second_high']

features_to_scale1 = ['open', 'y', 'low', 'close', 'volumefrom', 'volumeto', 'SMA_20', 'EMA_12', 'EMA_26',
                     'RSI', 'BB_upper', 'BB_lower', 'ROC_1', 'ROC_7', 'ROC_30', 'VROC_1', 'VROC_7', 'VROC_30',
                     'PMO_12_26', 'OBV', 'PVT', 'RVI', 'CMF', 'ATR', 'Stochastic_Oscillator', 'MACD_Histogram',
                     'close_lag_1', 'close_lag_7', 'close_lag_30', 'close_to_volume_ratio', 'close_mean',
                     'close_std', 'close_detrended', 'year', 'month', 'day', 'day_of_week_cos', 'day_of_week_sin', 'hour_low', 'minute_low', 'second_low', 'hour_high', 'minute_high', 'second_high']

# Initialize the scaler
scaler = StandardScaler()
# Scale the selected features
df_csv_standard[features_to_scale] = scaler.fit_transform(df_csv_help1[features_to_scale1])

# Initialize the scaler
minmax_scaler = MinMaxScaler()
df_csv_minmax[features_to_scale] = minmax_scaler.fit_transform(df_csv_help2[features_to_scale])

# Determine the index for splitting the data (e.g., 80% training, 20% testing)
split_index = int(0.8 * len(df_csv_sorted))
split_index_standard = int(0.8 * len(df_csv_standard))
split_index_minmax = int(0.8 * len(df_csv_minmax))


# Split the dataset into training and testing sets
train_data = df_csv_sorted.iloc[:split_index]
test_data = df_csv_sorted.iloc[split_index:]
train_data_standard = df_csv_standard.iloc[:split_index_standard]
test_data_standard = df_csv_standard.iloc[split_index_standard:]
train_data_minmax = df_csv_minmax.iloc[:split_index_minmax]
test_data_minmax = df_csv_minmax.iloc[split_index_minmax:]


# Verify the shape of the training and testing sets
print("Training data shape:", train_data.shape)
print("Testing data shape:", test_data.shape)
print("Training standard data shape:", train_data_standard.shape)
print("Testing standard data shape:", test_data_standard.shape)
print("Training minmax data shape:", train_data_minmax.shape)


# Features (X) are all columns except "high" and "low"
X_train = train_data.drop(columns=["y", "low"])
X_test = test_data.drop(columns=["y", "low"])

# Target variables (y) are "high" and "low"
y_train = train_data[["y", "low"]]
y_test = test_data[["y", "low"]]

# Features (X) are all columns except "high" and "low"
X_train_standard = train_data_standard.drop(columns=["y", "low"])
X_test_standard = test_data_standard.drop(columns=["y", "low"])

# Target variables (y) are "high" and "low"
y_train_standard = train_data_standard[["y", "low"]]
y_test_standard = test_data_standard[["y", "low"]]

# Features (X) are all columns except "high" and "low"
X_train_minmax = train_data_minmax.drop(columns=["y", "low"])
X_test_minmax = test_data_minmax.drop(columns=["y", "low"])

# Target variables (y) are "high" and "low"
y_train_minmax = train_data_minmax[["y", "low"]]
y_test_minmax = test_data_minmax[["y", "low"]]



# Example parameters to iterate over
param_grid = {
    'changepoint_prior_scale': [0.001, 0.01, 0.1, 0.5],
    'seasonality_prior_scale': [0.01, 0.1, 1.0, 10.0],
}

# Placeholder for model results
best_params = None
lowest_rmse = float('inf')

# Cross-validation setup
tscv = TimeSeriesSplit(n_splits=3)  # Adjust based on your dataset size and time frame

for cps in param_grid['changepoint_prior_scale']:
    for sps in param_grid['seasonality_prior_scale']:
        rmses = []
        for train_index, test_index in tscv.split(df_csv):
            train, test = df_csv.iloc[train_index], df_csv.iloc[test_index]
            # Initialize and fit Prophet model
            m = Prophet(changepoint_prior_scale=cps, seasonality_prior_scale=sps)
            m.fit(train[['ds', 'y']])
            future = m.make_future_dataframe(periods=len(test), freq='D')
            forecast = m.predict(future)
            # Calculate RMSE
            rmse = mean_squared_error(test['y'], forecast['yhat'].tail(len(test)), squared=False)
            rmses.append(rmse)
        # Update best params based on average RMSE
        avg_rmse = sum(rmses) / len(rmses)
        if avg_rmse < lowest_rmse:
            lowest_rmse = avg_rmse
            best_params = {'cps': cps, 'sps': sps}

print(f"Best Parameters: {best_params}, RMSE: {lowest_rmse}")

def build_lstm_model(input_shape, layers=2, neurons=50, dropout_rate=0.2):
    model = Sequential()
    model.add(LSTM(neurons, activation='relu', return_sequences=(layers > 1), input_shape=input_shape))
    model.add(Dropout(dropout_rate))
    for i in range(1, layers):
        model.add(LSTM(neurons, return_sequences=(i < layers - 1), activation='relu'))
        model.add(Dropout(dropout_rate))
    model.add(Dense(2))
    model.compile(optimizer='adam', loss='mse')
    return model

# Example LSTM model building
lstm_model = build_lstm_model((X_train_lstm.shape[1], X_train_lstm.shape[2]), layers=3, neurons=100)



# Assuming y_train['y'] is your target variable for the high price predictions
# Define the p, d, q range (for simplicity, we're using a small range here)
p = d = q = range(0, 3)
pdq_combinations = list(itertools.product(p, d, q))

best_aic = float("inf")
best_order = None
best_model = None

for combination in pdq_combinations:
    try:
        model = ARIMA(y_train['y'], order=combination).fit()
        if model.aic < best_aic:
            best_aic = model.aic
            best_order = combination
            best_model = model
    except:
        continue

print(f"Best ARIMA Order: {best_order}, AIC: {best_aic}")

from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor()
model.fit(X_train, y_train['y'])  # Adjust for your specific target variable
importances = model.feature_importances_
indices = np.argsort(importances)[::-1]
X_train_selected = X_train.iloc[:, indices[:10]]  # Select top 10 features

ensemble_forecast_high = (arima_forecast + prophet_forecast['yhat'].tail(len(test_data)) + lstm_forecast[:,0]) / 3
ensemble_forecast_low = (arima_forecast_low + prophet_forecast_low['yhat'].tail(len(test_data)) + lstm_forecast[:,1]) / 3
# Combine Prophet predictions with ARIMA and LSTM
ensemble_forecast_high = (arima_forecast + prophet_forecast['yhat'].tail(len(test_data)) + lstm_forecast[:, 0]) / 3
ensemble_forecast_low = (arima_forecast_low + prophet_forecast_low['yhat'].tail(len(test_data)) + lstm_forecast[:, 1]) / 3

# Adjust 'yhat' and array indexing based on your specific prediction setup

tscv = TimeSeriesSplit(n_splits=5)
for train_index, test_index in tscv.split(X):
    X_train_cv, X_test_cv = X.iloc[train_index], X.iloc[test_index]
    y_train_cv, y_test_cv = y.iloc[train_index], y.iloc[test_index]
    # Fit models on X_train_cv, y_train_cv and evaluate on X_test_cv, y_test_cv
