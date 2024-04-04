import requests
import pandas as pd
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.ensemble import RandomForestRegressor
from tensorflow.python.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.seasonal import STL
import talib


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


def filter_top_features(df, target='y', n_features=25):
    X = df.drop(['y', 'low', 'ds'], axis=1)
    y = df[target]

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)

    # Get feature importances and select top features
    feature_importances = pd.DataFrame(model.feature_importances_, index=X.columns, columns=['importance']).sort_values(
        'importance', ascending=False)
    top_features = feature_importances.head(n_features).index.tolist()

    # Include the target variables in the features to keep/
    features_to_keep = top_features + ['y', 'low', 'ds']
    return df[features_to_keep]


# Replace 'YOUR_API_KEY' with your actual API key from CryptoCompare
api_key = '5e634ddb33c885608fd2ae22be8b4e2af36c5675fad653eae97ad8a2c8811864'

# Base URL for the CryptoCompare API
base_url = 'https://sandbox-api.coinmarketcap.com/v1/cryptocurrency/listings/latest'

# Parameters for the API request
params = {
    'fsym': 'SOL',        # Symbol for Solana
    'tsym': 'USD',        # Convert prices to USD
    'limit': 30,          # Limit to the last 30 days of data
    'api_key': api_key     # Your API key
}


# Fetch historical data from the CryptoCompare API


response = requests.get(base_url, params=params)

if response.status_code == 200:
    data = response.json()['Data']['Data']
    df_api = pd.DataFrame(data)
    df_api.drop(['conversionType', 'conversionSymbol'], axis=1, inplace=True)
    df_api.rename(columns={
        'time': 'ds',
        'open': 'open',
        'high': 'y',
        'low': 'low',
        'close': 'close',
        'volumefrom': 'volumefrom',
        'volumeto': 'volumeto'
    }, inplace=True)
    df_api.drop(columns=['timeOpen', 'timeClose'], inplace=True)
    df_api['ds'] = pd.to_datetime(df_api['ds'])
    df_api['date'] = df_api['ds'].dt.date
    df_api.drop(columns=['ds'], inplace=True)
    df_api.rename(columns={'date': 'ds'}, inplace=True)
    df_api['ds'] = pd.to_datetime(df_api['ds'])
    df_api['timeLow'] = pd.to_datetime(df_api['timeLow'])
    df_api['timeHigh'] = pd.to_datetime(df_api['timeHigh'])

    # Extract relevant features
    df_api['hour_low'] = df_api['timeLow'].dt.hour
    df_api['minute_low'] = df_api['timeLow'].dt.minute
    df_api['second_low'] = df_api['timeLow'].dt.second

    df_api['hour_high'] = df_api['timeHigh'].dt.hour
    df_api['minute_high'] = df_api['timeHigh'].dt.minute
    df_api['second_high'] = df_api['timeHigh'].dt.second
    df_api.drop(columns=['timeLow', 'timeHigh'], inplace=True)

    # Extract relevant features
    df_api = calculate_technical_indicators(df_api)
    df_api = preprocess_data(df_api)
    df_api_sorted = df_api.sort_values(by='ds', ascending=True)
    df_api_sorted.drop(columns=['day_of_week'], inplace=True)
    features_to_scale = ['open', 'y', 'low', 'close', 'volumefrom', 'volumeto', 'SMA_20', 'EMA_12', 'EMA_26',
                         'RSI', 'BB_upper', 'BB_lower', 'ROC_1', 'ROC_7', 'ROC_30', 'VROC_1', 'VROC_7', 'VROC_30',
                         'PMO_12_26', 'OBV', 'PVT', 'RVI', 'CMF', 'ATR', 'Stochastic_Oscillator', 'MACD_Histogram',
                         'close_lag_1', 'close_lag_7', 'close_lag_30', 'close_to_volume_ratio', 'close_mean',
                         'close_std', 'close_detrended', 'year', 'month', 'day', 'day_of_week_cos', 'day_of_week_sin',
                         'hour_low', 'minute_low', 'second_low', 'hour_high', 'minute_high', 'second_high']
    minmax_scaler = MinMaxScaler()
    df_api_sorted[features_to_scale] = minmax_scaler.fit_transform(df_api_sorted[features_to_scale])
    df_api_sorted_filtered = filter_top_features(df_api_sorted)
    print(df_api_sorted_filtered)
    X_real_time = df_api_sorted_filtered.drop(['y', 'low', 'ds'], axis=1).values
    X_real_time_reshaped = X_real_time.reshape((X_real_time.shape[0], 1, X_real_time.shape[1]))
    lstm_model = load_model('../Trained_Models/best_lstm_model.h5')
    predictions = lstm_model.predict(X_real_time_reshaped)
    predictions_in_original_scale = minmax_scaler.inverse_transform(predictions)
    print(predictions_in_original_scale)
else:
    print('Error:', response.status_code)

