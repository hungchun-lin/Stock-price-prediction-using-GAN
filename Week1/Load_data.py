import os
import numpy as np
import csv
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter

# with open('DATA.csv', 'r') as file:
#     reader = csv.reader(file)
#     names = next(reader)
#     print(names)
#     for row in reader:
#         print(row)

## import data
df = pd.read_csv('DATA.csv', parse_dates=['Date'])
print(df.head())
print(df.tail())
print(df.shape)
print(df.columns)

# Create the plot
## https://www.earthdatascience.org/courses/use-data-open-source-python/use-time-series-data-in-python/date-time-types-in-pandas-python/customize-dates-matplotlib-plots-python/
fig, ax = plt.subplots(figsize=(10,3))
ax.plot(df['Date'], df['APPLE Close'], label='Apple stock')
ax.set(xlabel="Date",
       ylabel="USD",
       title="Apple Stock Price")
date_form = DateFormatter("%Y")
ax.xaxis.set_major_formatter(date_form)
plt.show()

# # Calculate technical indicators
# def get_technical_indicators(dataset):
#     # Create 7 and 21 days Moving Average
#     dataset['ma7'] = dataset['APPLE Close'].rolling(window=7).mean()
#     dataset['ma21'] = dataset['APPLE Close'].rolling(window=21).mean()
#
#     # Create MACD
#     dataset['26ema'] = dataset['APPLE Close'].ewm(span=26).mean()
#     dataset['12ema'] = dataset['APPLE Close'].ewm(span=12).mean()
#     dataset['MACD'] = (dataset['12ema'] - dataset['26ema'])
#
#     # Create Bollinger Bands
#     dataset['20sd'] = pd.stats.moments.rolling_std(dataset['APPLE Close'], 20)
#     dataset['upper_band'] = dataset['ma21'] + (dataset['20sd'] * 2)
#     dataset['lower_band'] = dataset['ma21'] - (dataset['20sd'] * 2)
#
#     # Create Exponential moving average
#     dataset['ema'] = dataset['APPLE Close'].ewm(com=0.5).mean()
#
#     # Create Momentum
#     dataset['momentum'] = dataset['APPLE Close'] - 1
#
#     return dataset
# TI_df = get_technical_indicators(df[['APPLE Close']])
# TI_df.head()
