from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.ensemble import RandomForestRegressor
import pandas as pd

df_csv_sorted = pd.read_csv('/Users/kostasbekis/CryptoMLTrader/Data/processed_data.csv')

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

# Calculate indices for splitting the data into training (80%), validation (10%), and testing (10%)
train_index = int(0.8 * len(df_csv_sorted))
val_index = int(0.9 * len(df_csv_sorted))  # 80% for training + 10% for validation

# Split the original dataset
train_data = df_csv_sorted.iloc[:train_index]
val_data = df_csv_sorted.iloc[train_index:val_index]
test_data = df_csv_sorted.iloc[val_index:]

# Split the standard scaled dataset
train_data_standard = df_csv_standard.iloc[:train_index]
val_data_standard = df_csv_standard.iloc[train_index:val_index]
test_data_standard = df_csv_standard.iloc[val_index:]

# Split the min-max scaled dataset
train_data_minmax = df_csv_minmax.iloc[:train_index]
val_data_minmax = df_csv_minmax.iloc[train_index:val_index]
test_data_minmax = df_csv_minmax.iloc[val_index:]

# Prepare feature sets (X) and target sets (y) for training, validation, and testing
# For the original dataset
X_train = train_data.drop(columns=["y", "low"])
X_val = val_data.drop(columns=["y", "low"])
X_test = test_data.drop(columns=["y", "low"])

y_train = train_data[["y", "low"]]
y_val = val_data[["y", "low"]]
y_test = test_data[["y", "low"]]

# For the standard scaled dataset
X_train_standard = train_data_standard.drop(columns=["y", "low"])
X_val_standard = val_data_standard.drop(columns=["y", "low"])
X_test_standard = test_data_standard.drop(columns=["y", "low"])

y_train_standard = train_data_standard[["y", "low"]]
y_val_standard = val_data_standard[["y", "low"]]
y_test_standard = test_data_standard[["y", "low"]]

# For the min-max scaled dataset
X_train_minmax = train_data_minmax.drop(columns=["y", "low"])
X_val_minmax = val_data_minmax.drop(columns=["y", "low"])
X_test_minmax = test_data_minmax.drop(columns=["y", "low"])

y_train_minmax = train_data_minmax[["y", "low"]]
y_val_minmax = val_data_minmax[["y", "low"]]
y_test_minmax = test_data_minmax[["y", "low"]]

# High and low targets for training, validation, and testing (Original dataset)
y_train_high = train_data[['y']]
y_val_high = val_data[['y']]
y_test_high = test_data[['y']]

y_train_low = train_data[['low']]
y_val_low = val_data[['low']]
y_test_low = test_data[['low']]

# High price as the target variable
y_train_minmax_high = train_data_minmax[['y']]  # For training
y_val_minmax_high = val_data_minmax[['y']]
y_test_minmax_high = test_data_minmax[['y']]    # For testing

# Low price as the target variable
y_train_minmax_low = train_data_minmax[['low']]  # For training
y_val_minmax_low = val_data_minmax[['low']]
y_test_minmax_low = test_data_minmax[['low']]    # For testing

# High price as the target variable
y_train_standard_high = train_data_standard[['y']]  # For training
y_val_standard_high = val_data_standard[['y']]
y_test_standard_high = test_data_standard[['y']]    # For testing

# Low price as the target variable
y_train_standard_low = train_data_standard[['low']]  # For training
y_val_standard_low = val_data_standard[['low']]
y_test_standard_low = test_data_standard[['low']]    # For testing


def perform_feature_selection(X_train, y_train, X_columns):
    """
    Performs feature selection using a RandomForestRegressor to determine feature importances.

    Args:
    - X_train: Training features DataFrame.
    - y_train: Training target Series.
    - X_columns: Column names of the training features for mapping back feature importances.

    Returns:
    - feature_importances_df: DataFrame containing feature names and their importance scores.
    """
    # Initialize and fit the RandomForestRegressor
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Get feature importances
    importances = model.feature_importances_

    # Map feature importances to the feature names
    feature_importances_df = pd.DataFrame({
        'Feature': X_columns,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False).reset_index(drop=True)

    return feature_importances_df


# Since y_train is a DataFrame with two targets, let's select one target for this example
# Here we're arbitrarily choosing 'y' which represents the 'high' price for feature selection
# Adjust according to your specific prediction target or run separately for each target
y_train_for_feature_selection = y_train['y'] if 'y' in y_train else y_train.iloc[:, 0]

# Perform feature selection
feature_importances_df = perform_feature_selection(X_train, y_train_for_feature_selection, X_train.columns)

# Print or save the feature importances
print(feature_importances_df)
# Optionally, save to CSV
# feature_importances_df.to_csv('feature_importances.csv', index=False)

def select_k_best_features(X_train, y_train, X_columns, k=20):
    """
    Selects the top k features based on univariate statistical tests.

    Args:
    - X_train: Training features DataFrame.
    - y_train: Training target Series.
    - X_columns: Column names of the training features for mapping selected features.
    - k: Number of top features to select.

    Returns:
    - selected_features_df: DataFrame containing the names of the selected features.
    """
    # Initialize and fit SelectKBest
    selector = SelectKBest(score_func=f_regression, k=k)
    selector.fit(X_train, y_train)

    # Get mask of selected features
    mask = selector.get_support(indices=True)

    # Map back to feature names
    selected_features_df = pd.DataFrame({
        'Feature': [X_columns[i] for i in mask]
    })

    return selected_features_df


# Adjust the target as necessary for your dataset
y_train_for_selectkbest = y_train['y'] if 'y' in y_train else y_train.iloc[:, 0]

# Perform feature selection with SelectKBest
selected_features_df = select_k_best_features(X_train, y_train_for_selectkbest, X_train.columns, k=20)

# Print or save the selected features from SelectKBest
print("Selected features by SelectKBest:")
print(selected_features_df)

# Optionally, you might want to save this to a CSV file
# selected_features_df.to_csv('selected_features_selectkbest.csv', index=False)

# Assuming perform_feature_selection function is already defined for RandomForestRegressor
# and has been run, as shown in the previous example
