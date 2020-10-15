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
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.layers import BatchNormalization, Embedding, TimeDistributed, LeakyReLU
from tensorflow.keras.layers import LSTM
from tensorflow.keras.optimizers import Adam

from matplotlib import pyplot
from pickle import load

X_train = np.load("X_train.npy")
y_train = np.load("y_train.npy")
X_test = np.load("X_test.npy")
y_test = np.load("y_test.npy")

#Parameters
LR = 0.001
BATCH_SIZE = 64
N_EPOCH = 50

input_dim = X_train.shape[1]
feature_size = X_train.shape[2]


def basic_lstm(input_dim, feature_size):
    model = Sequential()
    model.add(LSTM(units=64, input_shape=(input_dim, feature_size)))
    model.add(Dense(units=7))
    model.compile(optimizer=Adam(lr = LR), loss='mse')
    history = model.fit(X_train, y_train, epochs=N_EPOCH, batch_size=BATCH_SIZE, validation_data=(X_test, y_test),
                        verbose=2, shuffle=False)

    pyplot.plot(history.history['loss'], label='train')
    pyplot.plot(history.history['val_loss'], label='validation')
    pyplot.legend()
    pyplot.show()

    return model


model = basic_lstm(input_dim, feature_size)
print(model.summary())

yhat = model.predict(X_test, verbose=0)
print(yhat)

rmse = sqrt(mean_squared_error(y_test, yhat))
#print('t+%d RMSE: %f' % (rmse))
print(rmse)

# %% --------------------------------------- Plot the result  -----------------------------------------------------------------

dataset_test = pd.read_csv('dataset_test.csv', index_col=0)

X_scaler = load(open('X_scaler.pkl', 'rb'))
y_scaler = load(open('y_scaler.pkl', 'rb'))

rescaled_y_test = y_scaler.inverse_transform(y_test)
forecast_y = y_scaler.inverse_transform(yhat)

rescale_rmse = sqrt(mean_squared_error(rescaled_y_test, forecast_y))
print('The RMSE is: ', rescale_rmse)


predict_result = pd.DataFrame()
for i in range((rescaled_y_test.shape)[0]):
    y_predict = pd.DataFrame(forecast_y[i], columns = ["predicted_price"], index = dataset_test.index[i:i+7])
    predict_result = pd.concat([predict_result, y_predict], axis=1, sort=False)

y_test_dataset = pd.DataFrame()
for i in range((rescaled_y_test.shape)[0]):
    y_test = pd.DataFrame(rescaled_y_test[i], columns = ["predicted_price"], index = dataset_test.index[i:i+7])
    y_test_dataset = pd.concat([y_test_dataset, y_test], axis=1, sort=False)

y_test_dataset['mean'] = y_test_dataset.mean(axis=1)
predict_result['mean'] = predict_result.mean(axis=1)


plt.figure(figsize=(16, 8))
plt.plot(y_test_dataset['mean'])
plt.plot(predict_result['mean'], color = 'r')
plt.xlabel("Date")
plt.ylabel("Stock price")
plt.legend(("Real price", "Predicted price"), loc="upper left", fontsize=16)
plt.title("The result of Training", fontsize=20)
plt.show()
