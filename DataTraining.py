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


# Function to perform seasonal decomposition using STL
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
    df['day_of_week'] = df['ds'].dt.dayofweek
    df['month'] = df['ds'].dt.month

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
df_csv = calculate_technical_indicators(df_csv)
df_csv = preprocess_data(df_csv)

df_csv_sorted = df_csv.sort_values(by='ds', ascending=True)
df_csv_standard = df_csv_sorted
df_csv_minmax = df_csv_sorted


features_to_scale = ['open', 'y', 'low', 'close', 'volumefrom', 'volumeto', 'SMA_20', 'EMA_12', 'EMA_26',
                     'RSI', 'BB_upper', 'BB_lower', 'ROC_1', 'ROC_7', 'ROC_30', 'VROC_1', 'VROC_7', 'VROC_30',
                     'PMO_12_26', 'OBV', 'PVT', 'RVI', 'CMF', 'ATR', 'Stochastic_Oscillator', 'MACD_Histogram',
                     'close_lag_1', 'close_lag_7', 'close_lag_30', 'close_to_volume_ratio', 'close_mean',
                     'close_std', 'close_detrended']

# Initialize the scaler
scaler = StandardScaler()

# Scale the selected features
df_csv_standard[features_to_scale] = scaler.fit_transform(df_csv[features_to_scale])
# Initialize the scaler
minmax_scaler = MinMaxScaler()
df_csv_minmax[features_to_scale] = minmax_scaler.fit_transform(df_csv[features_to_scale])

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
print("Testing minmax data shape:", test_data_minmax.shape)


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

# 1. ARIMA Model
arima_model = ARIMA(y_train['y'], order=(5, 1, 0))  # Example order, you may need to tune this
arima_model_fit = arima_model.fit()
arima_forecast = arima_model_fit.forecast(steps=len(test_data))

arima_model_standard = ARIMA(y_train_standard['y'], order=(5, 1, 0))  # Example order, you may need to tune this
arima_model_standard_fit = arima_model_standard.fit()
arima_standard_forecast = arima_model_standard_fit.forecast(steps=len(test_data_standard))

arima_model_minmax = ARIMA(y_train_minmax['y'], order=(5, 1, 0))  # Example order, you may need to tune this
arima_model_minmax_fit = arima_model_minmax.fit()
arima_minmax_forecast = arima_model_minmax_fit.forecast(steps=len(test_data_minmax))

# 2. Prophet Model
prophet_model = Prophet()
prophet_model.fit(train_data[['ds', 'y']])
future = prophet_model.make_future_dataframe(periods=len(test_data), freq='D')
prophet_forecast = prophet_model.predict(future)

prophet_standard_model = Prophet()
prophet_standard_model.fit(train_data_standard[['ds', 'y']])
future_standard = prophet_standard_model.make_future_dataframe(periods=len(test_data_standard), freq='D')
prophet_standard_forecast = prophet_standard_model.predict(future_standard)

prophet_minmax_model = Prophet()
prophet_minmax_model.fit(train_data_minmax[['ds', 'y']])
future_minmax = prophet_minmax_model.make_future_dataframe(periods=len(test_data_minmax), freq='D')
prophet_minmax_forecast = prophet_minmax_model.predict(future_minmax)

# 3. LSTM Model
X_train_lstm = np.array(X_train).reshape((X_train.shape[0], 1, X_train.shape[1]))
X_test_lstm = np.array(X_test).reshape((X_test.shape[0], 1, X_test.shape[1]))

lstm_model = Sequential()
lstm_model.add(LSTM(50, activation='relu', input_shape=(X_train_lstm.shape[1], X_train_lstm.shape[2])))
lstm_model.add(Dense(2))  # 2 output neurons for 'high' and 'low'
lstm_model.compile(optimizer='adam', loss='mse')
lstm_model.fit(X_train_lstm, y_train, epochs=50, batch_size=16, verbose=0)
lstm_forecast = lstm_model.predict(X_test_lstm)

X_train_standard_lstm = np.array(X_train_standard).reshape((X_train_standard.shape[0], 1, X_train_standard.shape[1]))
X_test_standard_lstm = np.array(X_test_standard).reshape((X_test_standard.shape[0], 1, X_test_standard.shape[1]))

lstm_standard_model = Sequential()
lstm_standard_model.add(LSTM(50, activation='relu', input_shape=(X_train_standard_lstm.shape[1], X_train_standard_lstm.shape[2])))
lstm_standard_model.add(Dense(2))  # 2 output neurons for 'high' and 'low'
lstm_standard_model.compile(optimizer='adam', loss='mse')
lstm_standard_model.fit(X_train_standard_lstm, y_train_standard, epochs=50, batch_size=16, verbose=0)
lstm_standard_forecast = lstm_standard_model.predict(X_test_standard_lstm)

X_train_minmax_lstm = np.array(X_train_minmax).reshape((X_train_minmax.shape[0], 1, X_train_minmax.shape[1]))
X_test_minmax_lstm = np.array(X_test_minmax).reshape((X_test_minmax.shape[0], 1, X_test_minmax.shape[1]))

lstm_minmax_model = Sequential()
lstm_minmax_model.add(LSTM(50, activation='relu', input_shape=(X_train_minmax_lstm.shape[1], X_train_minmax_lstm.shape[2])))
lstm_minmax_model.add(Dense(2))  # 2 output neurons for 'high' and 'low'
lstm_minmax_model.compile(optimizer='adam', loss='mse')
lstm_minmax_model.fit(X_train_minmax_lstm, y_train_minmax, epochs=50, batch_size=16, verbose=0)
lstm_minmax_forecast = lstm_minmax_model.predict(X_test_minmax_lstm)

# Evaluate ARIMA model
arima_rmse = mean_squared_error(y_test['y'], arima_forecast, squared=False)
arima_mae = mean_absolute_error(y_test['y'], arima_forecast)

arima_standard_rmse = mean_squared_error(y_test_standard['y'], arima_standard_forecast, squared=False)
arima_standard_mae = mean_absolute_error(y_test_standard['y'], arima_standard_forecast)

arima_minmax_rmse = mean_squared_error(y_test_minmax['y'], arima_minmax_forecast, squared=False)
arima_minmax_mae = mean_absolute_error(y_test_minmax['y'], arima_minmax_forecast)

print("ARIMA Model RMSE:", arima_rmse)
print("ARIMA Model MAE:", arima_mae)

print("ARIMA Standard Model RMSE:", arima_standard_rmse)
print("ARIMA Standard Model MAE:", arima_standard_mae)

print("ARIMA MinMax Model RMSE:", arima_minmax_rmse)
print("ARIMA MinMax Model MAE:", arima_minmax_mae)

# Evaluate LSTM model
lstm_rmse = mean_squared_error(y_test['y'], lstm_forecast, squared=False)
lstm_mae = mean_absolute_error(y_test['y'], lstm_forecast)

lstm_standard_rmse = mean_squared_error(y_test_standard['y'], lstm_standard_forecast, squared=False)
lstm_standard_mae = mean_absolute_error(y_test_standard['y'], lstm_standard_forecast)

lstm_minmax_rmse = mean_squared_error(y_test_minmax['y'], lstm_minmax_forecast, squared=False)
lstm_minmax_mae = mean_absolute_error(y_test_minmax['y'], lstm_minmax_forecast)

print("LSTM Model RMSE:", lstm_rmse)
print("LSTM Model MAE:", lstm_mae)

print("LSTM Standard Model RMSE:", lstm_standard_rmse)
print("LSTM Standard Model MAE:", lstm_standard_mae)

print("LSTM MinMax Model RMSE:", lstm_minmax_rmse)
print("LSTM MinMax Model MAE:", lstm_minmax_mae)

# Evaluate Prophet model
prophet_rmse = mean_squared_error(y_test['y'], prophet_forecast['yhat'].tail(len(y_test)), squared=False)
prophet_mae = mean_absolute_error(y_test['y'], prophet_forecast['yhat'].tail(len(y_test)))

prophet_standard_rmse = mean_squared_error(y_test_standard['y'], prophet_standard_forecast['yhat'].tail(len(y_test_standard)), squared=False)
prophet_standard_mae = mean_absolute_error(y_test_standard['y'], prophet_standard_forecast['yhat'].tail(len(y_test_standard)))

prophet_minmax_rmse = mean_squared_error(y_test_minmax['y'], prophet_minmax_forecast['yhat'].tail(len(y_test_minmax)), squared=False)
prophet_minmax_mae = mean_absolute_error(y_test_minmax['y'], prophet_minmax_forecast['yhat'].tail(len(y_test_minmax)))

print("Prophet Model RMSE:", prophet_rmse)
print("Prophet Model MAE:", prophet_mae)

print("Prophet Standard Model RMSE:", prophet_standard_rmse)
print("Prophet Standard Model MAE:", prophet_standard_mae)

print("Prophet MinMax Model RMSE:", prophet_minmax_rmse)
print("Prophet MinMax Model MAE:", prophet_minmax_mae)

# Plot actual vs. predicted values for ARIMA
plt.figure(figsize=(14, 7))
plt.plot(test_data.index, test_data['y'], label='Actual High', color='blue')
plt.plot(test_data.index, arima_forecast, label='ARIMA Predicted High', color='red')
plt.title('Actual vs. ARIMA Predicted High Prices')
plt.xlabel('Date')
plt.ylabel('High Price')
plt.legend()
plt.show()

# Plot actual vs. predicted values for Prophet
plt.figure(figsize=(14, 7))
plt.plot(test_data.index, test_data['y'], label='Actual High', color='blue')
plt.plot(test_data.index, prophet_forecast['yhat'], label='Prophet Predicted High', color='green')
plt.title('Actual vs. Prophet Predicted High Prices')
plt.xlabel('Date')
plt.ylabel('High Price')
plt.legend()
plt.show()

# Plot actual vs. predicted values for LSTM
plt.figure(figsize=(14, 7))
plt.plot(test_data.index, test_data['y'], label='Actual High', color='blue')
plt.plot(test_data.index, lstm_forecast[:,0], label='LSTM Predicted High', color='orange')
plt.title('Actual vs. LSTM Predicted High Prices')
plt.xlabel('Date')
plt.ylabel('High Price')
plt.legend()
plt.show()
