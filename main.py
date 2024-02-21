import requests
import pandas as pd

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

# Replace 'YOUR_API_KEY' with your actual API key from CryptoCompare
api_key = 'https://min-api.cryptocompare.com/data/v2/histoday'

# Base URL for the CryptoCompare API
base_url = 'https://min-api.cryptocompare.com/data/v2/histoday'

# Parameters for the API request
params = {
    'fsym': 'SOL',        # Symbol for Solana
    'tsym': 'USD',        # Convert prices to USD
    'limit': 30,          # Limit to the last 30 days of data
    'api_key': api_key     # Your API key
}

# Function to preprocess the data


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

# Fetch historical data from the CryptoCompare API


response = requests.get(base_url, params=params)

if response.status_code == 200:
    data = response.json()['Data']['Data']
    df_api = pd.DataFrame(data)
    df_api.drop(['conversionType', 'conversionSymbol'], axis=1, inplace=True)
    df_api['time'] = pd.to_datetime(df_api['time'], unit='s')
    df_api = preprocess_data(df_api)
    # Load historical data from the CSV file
    csv_file_path = 'Solana_20_2_2023-20_2_2024_historical_data_coinmarketcap.csv'
    df_csv = pd.read_csv(csv_file_path, delimiter=';', parse_dates=['timeOpen', 'timeClose', 'timeHigh', 'timeLow'])
    df_csv.rename(columns={
        'timestamp': 'time',
        'open': 'open',
        'high': 'high',
        'low': 'low',
        'close': 'close',
        'volume': 'volumefrom',
        'marketCap': 'volumeto'
    }, inplace=True)
    df_csv.drop(columns=['timeOpen', 'timeClose', 'timeHigh', 'timeLow'], inplace=True)
    df_csv['time'] = pd.to_datetime(df_csv['time'])
    df_csv['date'] = df_csv['time'].dt.date
    df_csv.drop(columns=['time'], inplace=True)
    df_csv.rename(columns={'date': 'time'}, inplace=True)
    df_csv['time'] = pd.to_datetime(df_csv['time'])
    df_csv = preprocess_data(df_csv)
    print(df_csv)

    combined_df = pd.concat([df_csv, df_api], ignore_index=True)

    # Drop duplicates if any (optional)
    combined_df = combined_df.drop_duplicates(subset='time')

    # Sort the dataframe by 'time'
    combined_df = combined_df.sort_values(by='time').reset_index(drop=True)

    print(combined_df)

    print(df_api)
else:
    print('Error:', response.status_code)
