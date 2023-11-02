#!/usr/bin/env python3

from datetime import date
import matplotlib.pyplot as plt
import pandas as pd
from_file = __import__('2-from_file').from_file

df = from_file('coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv', ',')

df = df.drop(columns=['Weighted_Price'])
df = df.rename(columns={"Timestamp": "Date"})
df['Date'] = pd.to_datetime(df['Date'], unit='s')
df = df.set_index('Date')

df['Close'] = df['Close'].fillna(method='ffill')
df['High'] = df['High'].fillna(df['Close'])
df['Low'] = df['Low'].fillna(df['Close'])
df['Open'] = df['Open'].fillna(df['Close'])
df['Volume_(BTC)'] = df['Volume_(BTC)'].fillna(0)
df['Volume_(Currency)'] = df['Volume_(Currency)'].fillna(0)

df_plot = pd.DataFrame()
df_plot['High'] = df['High'].resample('d').max()
df_plot['Low'] = df['Low'].resample('d').min()
df_plot['Open'] = df['Open'].resample('d').mean()
df_plot['Close'] = df['Close'].resample('d').mean()
df_plot['Volume_(BTC)'] = df['Volume_(BTC)'].resample('d').sum()
df_plot['Volume_(Currency)'] = df['Volume_(Currency)'].resample('d').sum()
# Plot the data from 2017 and beyond at daily intervals
df_plot.plot()
plt.savefig('figura.jpg')