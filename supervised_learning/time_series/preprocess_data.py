import matplotlib as mpl
import pandas as pd

# Define the path to the CSV file
csv_path = '../data/bitstampUSD_1-min_data_2012-01-01_to_2020-04-22.csv'

# Read the CSV data into a DataFrame
df = pd.read_csv(csv_path)

# Select every 8th row starting from the 9th row (skipping some data for
# downsampling)
df = df[8::60]

# Convert the 'Timestamp' column to a datetime object and filter data from
# year 2015 onwards
df['date'] = pd.to_datetime(df['Timestamp'], unit='s')
df = df[df['date'].dt.year > 2014]

# Extract the 'date' column for later use
date = df['date']

# Interpolate missing values using a polynomial of order 2
df = df.loc[:, ~df.columns.isin(['date', 'Timestamp'])].interpolate(
    method='polynomial', order=2)

# Calculate High-Low percentage change and Percentage change in Closing
# and Opening prices
df['HL_PCT'] = (df['High'] - df['Close']) / df['Close'] * 100
df['PCT_change'] = (df['Close'] - df['Open']) / df['Open'] * 100

# Select relevant columns for analysis
df = df[['Close', 'HL_PCT', 'PCT_change', 'Volume_(BTC)']]

# Convert the 'date' column to timestamp in seconds
timestamp_s = date.map(pd.Timestamp.timestamp)

# Create a dictionary mapping column names to indices
column_indices = {name: i for i, name in enumerate(df.columns)}

n = len(df)
train_df = df[0:int(n * 0.7)]
val_df = df[int(n * 0.7):int(n * 0.9)]
test_df = df[int(n * 0.9):]

# Calculate the number of features (columns) in the DataFrame
num_features = df.shape[1]

# Calculate mean and standard deviation for normalization
train_mean = train_df.mean()
train_std = train_df.std()

# Normalize the training, validation, and test DataFrames
train_df = (train_df - train_mean) / train_std
val_df = (val_df - train_mean) / train_std
test_df = (test_df - train_mean) / train_std

# Normalize the entire DataFrame using the training mean and standard deviation
df_std = (df - train_mean) / train_std
