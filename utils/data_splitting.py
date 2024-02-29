import pandas as pd


df_csv_sorted = pd.read_csv('/Users/kostasbekis/CryptoMLTrader/Data/processed_data.csv')
df_csv_standard = pd.read_csv('/Users/kostasbekis/CryptoMLTrader/Data/df_csv_standard.csv')
df_csv_minmax = pd.read_csv('/Users/kostasbekis/CryptoMLTrader/Data/df_csv_minmax.csv')
df_csv_sorted_filtered = pd.read_csv('/Users/kostasbekis/CryptoMLTrader/Data/df_csv_sorted_filtered.csv')
df_csv_standard_filtered = pd.read_csv('/Users/kostasbekis/CryptoMLTrader/Data/df_csv_standard_filtered.csv')
df_csv_minmax_filtered = pd.read_csv('/Users/kostasbekis/CryptoMLTrader/Data/df_csv_minmax_filtered.csv')

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


# Calculate indices for splitting the data into training (80%), validation (10%), and testing (10%)
train_index = int(0.8 * len(df_csv_sorted_filtered))
val_index = int(0.9 * len(df_csv_sorted_filtered))  # 80% for training + 10% for validation

# Split the original dataset
train_data_filtered = df_csv_sorted_filtered.iloc[:train_index]
val_data_filtered = df_csv_sorted_filtered.iloc[train_index:val_index]
test_data_filtered = df_csv_sorted_filtered.iloc[val_index:]

# Split the standard scaled dataset
train_data_standard_filtered = df_csv_standard_filtered.iloc[:train_index]
val_data_standard_filtered = df_csv_standard_filtered.iloc[train_index:val_index]
test_data_standard_filtered = df_csv_standard_filtered.iloc[val_index:]

# Split the min-max scaled dataset
train_data_minmax_filtered = df_csv_minmax_filtered.iloc[:train_index]
val_data_minmax_filtered = df_csv_minmax_filtered.iloc[train_index:val_index]
test_data_minmax_filtered = df_csv_minmax_filtered.iloc[val_index:]

# Prepare feature sets (X) and target sets (y) for training, validation, and testing
# For the original dataset
X_train_filtered = train_data_filtered.drop(columns=["y", "low"])
X_val_filtered = val_data_filtered.drop(columns=["y", "low"])
X_test_filtered = test_data_filtered.drop(columns=["y", "low"])

y_train_filtered = train_data_filtered[["y", "low"]]
y_val_filtered = val_data_filtered[["y", "low"]]
y_test_filtered = test_data_filtered[["y", "low"]]

# For the standard scaled dataset
X_train_standard_filtered = train_data_standard_filtered.drop(columns=["y", "low"])
X_val_standard_filtered = val_data_standard_filtered.drop(columns=["y", "low"])
X_test_standard_filtered = test_data_standard_filtered.drop(columns=["y", "low"])

y_train_standard_filtered = train_data_standard_filtered[["y", "low"]]
y_val_standard_filtered = val_data_standard_filtered[["y", "low"]]
y_test_standard_filtered = test_data_standard_filtered[["y", "low"]]

# For the min-max scaled dataset
X_train_minmax_filtered = train_data_minmax_filtered.drop(columns=["y", "low"])
X_val_minmax_filtered = val_data_minmax_filtered.drop(columns=["y", "low"])
X_test_minmax_filtered = test_data_minmax_filtered.drop(columns=["y", "low"])

y_train_minmax_filtered = train_data_minmax_filtered[["y", "low"]]
y_val_minmax_filtered = val_data_minmax_filtered[["y", "low"]]
y_test_minmax_filtered = test_data_minmax_filtered[["y", "low"]]

# High and low targets for training, validation, and testing (Original dataset)
y_train_high_filtered = train_data_filtered[['y']]
y_val_high_filtered = val_data_filtered[['y']]
y_test_high_filtered = test_data_filtered[['y']]

y_train_low_filtered = train_data_filtered[['low']]
y_val_low_filtered = val_data_filtered[['low']]
y_test_low_filtered = test_data_filtered[['low']]

# High price as the target variable
y_train_minmax_high_filtered = train_data_minmax_filtered[['y']]  # For training
y_val_minmax_high_filtered = val_data_minmax_filtered[['y']]
y_test_minmax_high_filtered = test_data_minmax_filtered[['y']]    # For testing

# Low price as the target variable
y_train_minmax_low_filtered = train_data_minmax_filtered[['low']]  # For training
y_val_minmax_low_filtered = val_data_minmax_filtered[['low']]
y_test_minmax_low_filtered = test_data_minmax_filtered[['low']]    # For testing

# High price as the target variable
y_train_standard_high_filtered = train_data_standard_filtered[['y']]  # For training
y_val_standard_high_filtered = val_data_standard_filtered[['y']]
y_test_standard_high_filtered = test_data_standard_filtered[['y']]    # For testing

# Low price as the target variable
y_train_standard_low_filtered = train_data_standard_filtered[['low']]  # For training
y_val_standard_low_filtered = val_data_standard_filtered[['low']]
y_test_standard_low_filtered = test_data_standard_filtered[['low']]    # For testing

print("Data splitted correctly")