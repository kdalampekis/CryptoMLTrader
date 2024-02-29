# utils/data_preprocessing.py
import pandas as pd
import numpy as np
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


pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

# Load historical data from the CSV file
csv_file_path = '/Users/kostasbekis/CryptoMLTrader/Data/Solana_20_2_2023-20_2_2024_historical_data_coinmarketcap.csv'
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
df_csv_sorted.drop(columns=['day_of_week'], inplace=True)
df_csv_sorted.to_csv('/Users/kostasbekis/CryptoMLTrader/Data/processed_data.csv', index=False)
