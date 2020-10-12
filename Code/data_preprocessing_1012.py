import time
import os
import pandas as pd
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import tensorflow

from numpy import *
from math import sqrt
from pandas import *
from datetime import datetime, timedelta

from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_squared_error
from pickle import dump

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.layers import BatchNormalization, Embedding, TimeDistributed, LeakyReLU
from tensorflow.keras.layers import LSTM
from tensorflow.keras.optimizers import Adam

from matplotlib import pyplot

dataset = pd.read_csv('Finaldata_with_Fourier.csv', parse_dates=['Date'])
print(dataset.columns)
# Check NA and fill them
dataset.isnull().sum()
dataset.iloc[:,1:] = pd.concat([dataset.iloc[:,1:].ffill(), dataset.iloc[:,1:].bfill()]).groupby(level=0).mean()

# Set the date to datetime data
datetime_series = pd.to_datetime(dataset['Date'])
datetime_index = pd.DatetimeIndex(datetime_series.values)
dataset = dataset.set_index(datetime_index)
dataset = dataset.sort_values(by = 'Date')
dataset = dataset.drop(columns = 'Date')


# Normalized the data
X_value = pd.DataFrame(dataset.iloc[:,:])
y_value = pd.DataFrame(dataset.iloc[:,0])

X_scaler = MinMaxScaler(feature_range = (-1,1))
y_scaler = MinMaxScaler(feature_range = (-1,1))
X_scaler.fit(X_value)
y_scaler.fit(y_value)
X_scale_dataset = X_scaler.fit_transform(X_value)
y_scale_dataset = y_scaler.fit_transform(y_value)

dump(X_scaler, open('X_scaler.pkl', 'wb'))
dump(y_scaler, open('y_scaler.pkl', 'wb'))

# Set input/output steps
#Parameters
n_steps_in = 30
n_features = X_value.shape[1]
n_steps_out = 7

data = dataset

# Get X/y dataset
def get_X_y(X_data, y_data):
    X = list()
    y = list()

    X_scale_value = X_scale_dataset
    y_scale_value = y_scale_dataset

    values = data
    for i in range(len(values) - (n_steps_in + n_steps_out)):
        X_value = X_scale_value[i: i + n_steps_in][:, :]
        y_value = y_scale_value[i + n_steps_in: i + (n_steps_in + n_steps_out)][:, 0]

        X.append(X_value)
        y.append(y_value)

    return np.array(X), np.array(y)

# Split train/test dataset
def split_train_test(X, y):
    train_size = round(len(X) * 0.7)
    test_size = len(X) - train_size

    X_train = X[0:train_size]
    y_train = y[0:train_size]

    X_test = X[train_size:]
    y_test = y[train_size:]
    return X_train, y_train, X_test, y_test

# Get data and check shape
#X, y = get_X_y(X_value.values, y_value.values)
X, y = get_X_y(X_scale_dataset, y_scale_dataset)
X_train, y_train, X_test, y_test = split_train_test(X, y)

y = y.reshape(y.shape[0],y.shape[1], 1)
#y_train = y_train.reshape(y_train.shape[0],y_train.shape[1], 1)
#y_test = y_test.reshape(y_test.shape[0],y_test.shape[1], 1)

dataset_train = dataset.iloc[:X_train.shape[0],:]
dataset_train.to_csv('dataset_train.csv')

print('dataset_train', dataset_train)

print('X shape: ', X.shape)
print('y shape: ', y.shape)
print('X_train shape: ', X_train.shape)
print('y_train shape: ', y_train.shape)
print('X_test shape: ', X_test.shape)
print('y_test shape: ', y_test.shape)

np.save("X_train.npy", X_train)
np.save("y_train.npy", y_train)
np.save("X_test.npy", X_test)
np.save("y_test.npy", y_test)
