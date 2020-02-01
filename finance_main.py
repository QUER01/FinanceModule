# -*- coding: utf-8 -*-
"""
Created on Fri Nov 18 19:29:05 2016

@author: Julian Quernheim

interesting links: 
  http://gbeced.github.io/pyalgotrade/docs/v0.18/html/ 
  https://www.backtrader.com/docu/index.html
  https://ntguardian.wordpress.com/2016/09/19/introduction-stock-market-data-python-1/
"""

import quandl
import numpy as np
import pandas as pd
from urllib.request import urlopen
import os
import math
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(rc={'figure.figsize':(11, 4)})


from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dropout
from keras.layers import Dense
from keras.layers import LSTM
from keras.callbacks import EarlyStopping

from matplotlib import pyplot
#import pandas_profiling # pip install pandas-profiling
from FinanceModule.FinanceModule.quandlModule import Quandl
from pandas.tseries.offsets import DateOffset

# convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    """
    Frame a time series as a supervised learning dataset.
    Arguments:
    	data: Sequence of observations as a list or NumPy array.
    	n_in: Number of lag observations as input (X).
    	n_out: Number of observations as output (y).
    	dropnan: Boolean whether or not to drop rows with NaN values.
    Returns:
    	Pandas DataFrame of series framed for supervised learning.
    """
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
    # put it all together
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)

    print(agg.head())
    print('rows : ' + str(len(agg)))
    print('columns : ' + str(len(agg.T)))

    return agg

def profiling(df, title, outputFile = True , outputHTMLstr = False):
    profile = df.profile_report(title=title)
    if outputFile:
        return profile.to_file(output_file="output.html")
    if outputHTMLstr:
        return profile.to_html()

def rescaleData(data):
    values = data.values
    values = values.astype('float32')
    # normalize features
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(values)
    # frame as supervised learning
    return scaled , scaler

def splitTrainTest(data, n_test):
    '''

    :param data:
    :param percentage:
    :return:
    '''
    values = data.values
    train, test = values[0:-n_test], values[-n_test:]
    return train , test

def splitTrainTestXY(train, test):
    # split into input and outputs
    train_X, train_y = train[:, :-1], train[:, -1]
    test_X, test_y = test[:, :-1], test[:, -1]
    # reshape input to be 3D [samples, timesteps, features]
    train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
    test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
    print('Dataset shapes')
    print('Train_X: ' + str(train_X.shape))
    print('Train_y: ' + str(train_y.shape))
    print('Test_X: ' + str(test_X.shape))
    print('Test_y: ' + str(test_y.shape))

    return train_X, train_y, test_X, test_y

def designModel(train_X, return_sequence):
    # design network
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(train_X.shape[1], train_X.shape[2])))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.2))

    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.2))

    model.add(LSTM(units=50))
    model.add(Dropout(0.2))

    model.add(Dense(units=1))

    #model.add(LSTM(200, input_shape=(train_X.shape[1], train_X.shape[2]), return_sequences= return_sequence))
    #model.add(LSTM(1, return_sequences=False))
    #model.add(Dropout(0.2))
    #model.add(Dense(1))

    model.compile(loss='mean_squared_error', optimizer='adam')

    return model

def fitModel(train_X, train_y, test_X, test_y):
    '''

    :param train_X:
    :param train_y:
    :param test_X:
    :param test_y:
    :return:
    '''
    history = model.fit(train_X, train_y, epochs=200, batch_size=72, validation_data=(test_X, test_y), verbose=2,
                        callbacks=[EarlyStopping(monitor='val_loss', patience=10)], shuffle=False)

    # plot history
    pyplot.plot(history.history['loss'], label='train')
    pyplot.plot(history.history['val_loss'], label='test')
    pyplot.legend()
    pyplot.show()

    return history

def forecast_lstm(model, X, n_batch):
    # reshape input pattern to [samples, timesteps, features]
    X = X.reshape(1, 1, len(X))
    # make forecast
    forecast = model.predict(X, batch_size=n_batch)
    # convert to array
    return [x for x in forecast[0, :]]

def make_forecasts(model, n_batch, data, n_lag):
    forecasts = list()
    for i in range(len(data)):
        X, y = data[i, 0:n_lag], data[i, n_lag:]

        # make forecast
        forecast = forecast_lstm(model, X, n_batch)
        # store the forecast
        forecasts.append(forecast)
        #print('Forecast for time step : ' + str(i) + ':'  + str(forecast))
        #print(str(X))
    return forecasts

def evaluateModel(yhat, actual_x, actual_y, n_input , features, column):
    '''

    :param yhat:
    :param actual_x:
    :param actual_y:
    :param n_input:
    :param features:
    :return:
    '''

    actual_x = actual_x.reshape((actual_x.shape[0], actual_x.shape[2]))
    # invert scaling for forecast
    inv_yhat = np.concatenate((yhat, actual_x[:, n_input*features:]), axis=1)
    inv_yhat = scaler.inverse_transform(inv_yhat)
    inv_yhat = inv_yhat[:, 0]
    # invert scaling for actual
    actual_y = actual_y.reshape((len(actual_y), 1))
    inv_y = np.concatenate((actual_y, actual_x[:, n_input*features:]), axis=1)
    inv_y = scaler.inverse_transform(inv_y)
    inv_y = inv_y[:, 0]

    # plot prediction
    pyplot.plot(inv_yhat, label='prediction for ' + column[1])
    pyplot.plot(inv_y, label='actual ' + column[1])
    pyplot.legend()
    pyplot.show()

    # calculate RMSE
    rmse = math.sqrt(mean_squared_error(inv_y, inv_yhat))
    print('Test RMSE: %.3f' % rmse)


    return inv_y, inv_yhat , rmse

def plot_forecasts(series, forecasts, n_test):
	# plot the entire dataset in blue
	pyplot.plot(series.values)
	# plot the forecasts in red
	for i in range(len(forecasts)):
		off_s = len(series) - n_test + i - 1
		off_e = off_s + len(forecasts[i]) + 1
		xaxis = [x for x in range(off_s, off_e)]
		yaxis = [series.values[off_s]] + forecasts[i]
		pyplot.plot(xaxis, yaxis, color='red')
	# show the plot
	pyplot.show()


# --  MAIN  --
apiKey = "fKx3jVSpXasnsXKGnovb"
market = 'FSE'


# Download list of API codes
Quandl = Quandl(apiKey)
df_tickers = Quandl.getStockExchangeCodes()
# filter data API codes
l_tickers = df_tickers['code'].tolist()

# Download the dataset
#df_stockHistory = Quandl.getStockMarketData(market= market ,ListQuandleCodes=l_tickers)
#pd.DataFrame.to_csv(df_stockHistory, sep=';', path_or_buf='FinanceModule/data/fse_stocks')
df_stockHistory = pd.read_csv('FinanceModule/data/fse_stocks', sep=';')
df_stockHistory.Date = pd.to_datetime(df_stockHistory.Date)
df_stockHistory = df_stockHistory.set_index("Date")

# Filter the dataset and keep only Close values
#df_stockHistory = df_stockHistory.rename(columns={"Last": "EURONEXT/BOEI_Close"})
df_stockHistoryCleaned = df_stockHistory.filter(like='Close' , axis=1)
# Get only last 250*5 days
df_stockHistoryCleaned = df_stockHistoryCleaned.tail(250*5)
# keep only rows with at least 70% filled cells
#df_stockHistoryCleaned = df_stockHistoryCleaned.dropna(axis=0, thresh=(round(len(df_stockHistoryCleaned.T)*0.7)))
df_stockHistoryCleaned = df_stockHistoryCleaned.dropna(axis=1, thresh=(round(len(df_stockHistoryCleaned.T)*0.7)))


# Iterate over the sequence of column names
for column in df_stockHistoryCleaned:
    features = 1
    n_test = 100
    n_batch = 1
    n_input, n_output = 1, 1
    n_last_values = n_input + n_output -1

    # Select column contents by column name using [] operator
    df_stockHistoryCleanedColumn = pd.DataFrame(df_stockHistoryCleaned[column])
    df_proj = df_stockHistoryCleanedColumn
    print('Colunm Name : ', column)


    # -----------------------------
    #       Eval model
    # -----------------------------
    
    # Scale the dataset between 0 and 1
    #scaled, scaler = rescaleData(data = df_stockHistoryCleanedColumn)
    # reframe the dataset to an unsupervised learning dataset
    reframed = series_to_supervised(df_stockHistoryCleanedColumn, n_input, n_output)
    # split the dataset into train and test sets
    train, test = splitTrainTest(data =reframed, n_test=n_test)

    scaler = MinMaxScaler()
    scaler.fit(df_stockHistoryCleanedColumn)
    train = scaler.transform(train)
    test = scaler.transform(test)

    train_X, train_y, test_X, test_y = splitTrainTestXY(train=train , test=test)

    # Design the model
    model = designModel(train_X, return_sequence = True)
    # fit network
    history = fitModel(train_X, train_y, test_X, test_y)

    forecasts_for_validation = make_forecasts(model, n_batch=1, data=test, n_lag=n_input)
    forecasts_for_validation = scaler.inverse_transform(forecasts_for_validation)
    forecasts_for_validation = forecasts_for_validation.tolist()
    # plot forecasts
    plot_forecasts(df_stockHistoryCleanedColumn, forecasts_for_validation, n_test)


    # ----------------------------------------------
    # Model for future forecasts
    # ----------------------------------------------


    for i in range(0,5):

        print('Starting forecast : ' + str(i))

        # Scale the dataset between 0 and 1
        # scaled, scaler = rescaleData(data = df_stockHistoryCleanedColumn)
        # reframe the dataset to an unsupervised learning dataset
        reframed = series_to_supervised(df_stockHistoryCleanedColumn, n_input, n_output)

        # split the dataset into train and test sets
        train_X, train_y = reframed.values[:, :-1], reframed.values[:, -1]

        x_scaler = MinMaxScaler()
        y_scaler = MinMaxScaler()

        train_X_scaled = x_scaler.fit_transform(train_X)
        train_y_scaled = y_scaler.fit_transform(train_y.reshape(-1,1))


        # reshape input to be 3D [samples, timesteps, features]
        train_X_scaled_reshaped = train_X.reshape((train_X_scaled.shape[0], 1, train_X_scaled.shape[1]))

        # Design the model
        newModel = designModel(train_X_scaled_reshaped, return_sequence=False)
        # fit network
        history = newModel.fit(train_X_scaled_reshaped, train_y_scaled, epochs=20, batch_size=72, verbose=2,
                            callbacks=[EarlyStopping(monitor='val_loss', patience=10)], shuffle=False)


        #train_last_values = train[-n_last_values:]
        #train_last_values_inv = scaler.inverse_transform(train_last_values)

        #X= train_last_values[n_last_values-1, 1:n_last_values+1]

        #X = train_last_values[-1,0:train_last_values.shape[1]-1 ] # getting last row, gettin all -1 values from last row
        #X = train_last_values
        #X_inv = scaler.inverse_transform(X.reshape(-1,1))

        # make forecast
        data = train_X_scaled
        X= data[0, 0:n_input]
        X = X.reshape(1, 1, len(X))
        # make forecast
        forecast = newModel.predict(X, batch_size=n_batch)



        #forecast = make_forecasts(newModel, n_batch=1, data=train_X_scaled, n_lag=n_input)
        #forecast = forecast_lstm(model = newModel, X = train_X_scaled, n_batch=1)

        forecasts_for_future = y_scaler.inverse_transform(np.array(forecast).reshape(-1,1))

        add_dates = [df_stockHistoryCleanedColumn.index[-1] + DateOffset(days=x) for x in range(0, n_output + 1)]
        future_dates = pd.DataFrame(index=add_dates[1:], columns=df_stockHistoryCleanedColumn.columns)



        # create dataframe as new input

        df_predict = pd.DataFrame(forecasts_for_future[-n_output:], index=future_dates[-n_output:].index,
                                  columns=[column])
        df_stockHistoryCleanedColumn = df_stockHistoryCleanedColumn.append(df_predict, sort = False)

        # create dataframe for visualization
        df_predict = pd.DataFrame(forecasts_for_future[-n_output:], index=future_dates[-n_output:].index,
                                  columns=['Prediction'])
        df_proj = df_proj.append(df_predict, sort = False)




    plt.figure(figsize=(20, 5))
    plt.plot(df_proj.index, df_proj[column])
    plt.plot(df_proj.index, df_proj['Prediction'], color='r')
    plt.legend(loc='best', fontsize='xx-large')
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=16)
    plt.show()

