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

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)
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


def calculate_MACD(df, ema_short_max=12, ema_long_max=26, signal_max=9):
    # Determine the maximum available length of data
    available_length = len(df['close'].dropna())

    # Adjust the periods based on the available length
    ema_short = min(ema_short_max, available_length - 1)
    ema_long = min(ema_long_max, available_length - 1)
    signal_period = min(signal_max, available_length - 1)

    # Ensure that we have enough data to compute long EMA
    if available_length > ema_long:
        # Calculate the MACD
        exp1 = df['close'].ewm(span=ema_short, adjust=False).mean()
        exp2 = df['close'].ewm(span=ema_long, adjust=False).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=signal_period, adjust=False).mean()
        hist = macd - signal
    else:
        # Not enough data to compute MACD
        macd, signal, hist = [pd.Series([np.nan] * len(df)) for _ in range(3)]

    return macd, signal, hist


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


def main():

    csv_file_path = 'Data/crypto_last30days.csv'
    df_api = pd.read_csv(csv_file_path, delimiter=';', parse_dates=['timeOpen', 'timeClose', 'timeHigh', 'timeLow'])
    df_api.rename(columns={
            'timestamp': 'ds',
            'open': 'open',
            'high': 'y',
            'low': 'low',
            'close': 'close',
            'volume': 'volumefrom',
            'marketCap': 'volumeto'
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
    df_api['MACD'], df_api['MACD_signal'], df_api['MACD_Histogram'] = calculate_MACD(df_api)
    df_api = preprocess_data(df_api)
    df_api_sorted = df_api.sort_values(by='ds', ascending=False)
    df_api_sorted.drop(columns=['day_of_week'], inplace=True)

    # List of features used during training
    used_features = [
        'close', 'close_detrended', 'open', 'close_lag_1', 'volumeto', 'ATR', 'OBV',
        'VROC_30', 'VROC_1', 'volumefrom', 'VROC_7', 'hour_low', 'minute_high',
        'minute_low', 'close_to_volume_ratio', 'EMA_12', 'day', 'RSI',
        'close_lag_30', 'CMF', 'close_std', 'ROC_7', 'MACD_Histogram', 'ROC_30',
        'hour_high', 'y', 'low', 'ds'
    ]

    # Ensure all required features are present
    df_api_sorted = df_api_sorted[used_features]

    # Initialize and fit the scaler for the features
    feature_scaler = MinMaxScaler()
    df_scaled = df_api_sorted.copy()  # Copy df to keep original separate
    features_to_scale = df_api_sorted.columns.difference(['y', 'low', 'ds'])  # Exclude targets and date
    df_scaled[features_to_scale] = feature_scaler.fit_transform(df_api_sorted[features_to_scale])

    # Separate features and targets for prediction
    X = df_scaled.drop(['y', 'low', 'ds'], axis=1).values
    X = X.reshape((X.shape[0], 1, X.shape[1]))  # Reshape for LSTM

    # Load the trained LSTM model
    model_path = 'Trained_Models/best_lstm_model.h5'
    model = load_model(model_path)

    # Predict using the LSTM model
    predictions = model.predict(X)

    # Initialize and fit the scaler for the targets if they were scaled during training
    target_scaler = MinMaxScaler()
    df_targets = df_api_sorted[['y']]
    target_scaler.fit(df_targets)
    predictions_scaled_back = target_scaler.inverse_transform(predictions)

    # Print the original scale predictions
    print("Predictions in original scale:", predictions_scaled_back)


if __name__ == '__main__':
    main()
