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
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout #Sequential
from tensorflow.keras.layers import BatchNormalization, Embedding, TimeDistributed, LeakyReLU
from tensorflow.keras.layers import LSTM
from tensorflow.keras.optimizers import Adam

from matplotlib import pyplot
## - - - - - -- - - - - - - - - - - - - - - - -

## Read Data
dataset = read_csv('DATA_temp.csv')

## Fill Na with the previous and the next value
dataset.iloc[:,1:] = pd.concat([dataset.iloc[:,1:].ffill(), dataset.iloc[:,1:].bfill()]).groupby(level=0).mean()

## Set the date to datetime formate
datetime_series = pd.to_datetime(dataset['Date'])
datetime_index = pd.DatetimeIndex(datetime_series.values)
dataset = dataset.set_index(datetime_index)
dataset = dataset.sort_values(by = 'Date')
dataset = dataset.drop(columns = 'Date')

## Normalize the data
X_value = pd.DataFrame(dataset.iloc[:,1:])
y_value = pd.DataFrame(dataset.iloc[:,0])

X_scaler = MinMaxScaler(feature_range = (-1,1))
y_scaler = MinMaxScaler(feature_range = (-1,1))
X_scaler.fit(X_value)
y_scaler.fit(y_value)
X_scale_dataset = X_scaler.fit_transform(X_value)
y_scale_dataset = y_scaler.fit_transform(y_value)

## Set up the input/ output parameter
n_steps_in = 30
n_features = X_value.shape[1]
n_steps_out = 1

## Function of Get X and y dataset
data = dataset
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

## Function of split train and test dataset

def split_train_test(X, y):
    train_size = round(len(X) * 0.7)
    test_size = len(X) - train_size

    X_train = X[0:train_size]
    y_train = y[0:train_size]

    X_test = X[train_size:]
    y_test = y[train_size:]
    return X_train, y_train, X_test, y_test

## Get data to input the model and check the shape
#X, y = get_X_y(X_value.values, y_value.values)
X, y = get_X_y(X_scale_dataset, y_scale_dataset)
X_train, y_train, X_test, y_test = split_train_test(X, y)

print('X shape: ', X.shape)
print('y shape: ', y.shape)
print('X_train shape: ', X_train.shape)
print('y_train shape: ', y_train.shape)
print('X_test shape: ', X_test.shape)
print('y_test shape: ', y_test.shape)


## LSTM model and training
#Parameters
LR = 0.001
BATCH_SIZE = 128
N_EPOCH = 50

model = Sequential()
model.add(LSTM(units=64,input_shape=(n_steps_in, n_features), return_sequences=True)) #,return_sequences=True
#model.add(LeakyReLU())
model.add(LSTM(16))
#model.add(LeakyReLU())
model.add(Dense(n_steps_out))
model.compile(optimizer=Adam(lr = LR), loss='mse')

history = model.fit(X_train, y_train, epochs = N_EPOCH, batch_size = BATCH_SIZE, validation_data = (X_test, y_test),
                    verbose=2, shuffle=False)

yhat = model.predict(X_test, verbose=0)

#model.save('LSTM_model.h5')

# plot history
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
pyplot.show()

#print(yhat)
def rescale_dataset(X_train, y_train, X_test, y_test, yhat):
    ## Rescaled back
    rescaled_X_train = X_scaler.inverse_transform(np.vstack(X_train))
    rescaled_y_train = y_scaler.inverse_transform(y_train)

    rescaled_X_test = X_scaler.inverse_transform(np.vstack(X_test))
    rescaled_y_test = y_scaler.inverse_transform(y_test)

    ## Rescale the predicted value
    forecast_y = y_scaler.inverse_transform(yhat) #Should pare with test set

    return rescaled_y_test, forecast_y


def plot_result(rescaled_y_test, forecast_y):
    dataset_test = dataset.iloc[-(shape(forecast_y)[0]):,]
    y_test = pd.DataFrame(rescaled_y_test, columns = ["real_price"], index = dataset_test.index)
    y_predict = pd.DataFrame(forecast_y, columns = ["predicted_price"], index = dataset_test.index)

    plt.figure(figsize=(16, 8))
    plt.plot(y_test)
    plt.plot(y_predict, color = 'r')
    plt.xlabel("Date")
    plt.ylabel("Stock price")
    plt.legend(("test", "predicted"), loc="upper left", fontsize=16)
    plt.title("The result of Training", fontsize=20)
    plt.show()



rescaled_y_test, forecast_y = rescale_dataset(X_train, y_train, X_test, y_test, yhat)
plot_result(rescaled_y_test, forecast_y)