from keras.models import Model
from keras.layers import Dense, Dropout, LSTM, Input, Activation, concatenate
from keras import regularizers
from keras import optimizers
import numpy as np
np.random.seed(4)
from scipy.ndimage.interpolation import shift
import pandas as pd
from sklearn import preprocessing
from pulp import *
from lib.pypfopt_055.efficient_frontier import EfficientFrontier
from lib.pypfopt_055 import base_optimizer, risk_models
from lib.pypfopt_055 import expected_returns
from lib.pypfopt_055.discrete_allocation import DiscreteAllocation, get_latest_prices
import matplotlib.pyplot as plt

from termcolor import colored
from src.util_forecasting_metrics import *
from pandas.tseries.offsets import DateOffset
import gc


import keras
from keras.layers import Lambda, Input, Dense
from keras.models import Model
from keras.losses import mse, binary_crossentropy
from keras import backend as K
from keras.utils import plot_model
import numpy as np





def fun_column(matrix, i):
    return [row[i] for row in matrix]

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]




def transformDataset(input_path,input_sep, output_path, output_sep,metadata_input_path, metadata_sep, filter_sectors = None, n_tickers = 'empty', n_last_values = 250 ):
    # filter out tickers that have no values in the last n days
    df = pd.read_csv(input_path, sep=input_sep)
    df['RN'] = df.sort_values(['date'], ascending=[False]) \
                   .groupby(['ticker']) \
                   .cumcount() + 1
    df = df[df['RN'] <= n_last_values]

    # filter out tickers that have no values in the last n days from the metadata table
    df_metadata = pd.read_csv(metadata_input_path, sep=metadata_sep)
    metadata_filtered = df_metadata[df_metadata.sector.isin(filter_sectors)]
    metadata_filtered = metadata_filtered[metadata_filtered.ticker.isin(pd.unique(df['ticker']))]

    # get n_tickers within each sector group
    i = 0
    for sector in filter_sectors:
        if i == 0:
            tickers = metadata_filtered[metadata_filtered.sector == sector].head(n_tickers)

        else:
            tickers = pd.concat([tickers, metadata_filtered[metadata_filtered.sector == sector].head(n_tickers)])

        i = i + 1

    # filter out the n_tikcers within each sector group from main dataset
    df = df[df.ticker.isin(tickers['ticker'])]
    l_tickers = df.ticker.unique()

    # transform the dataset
    n = 1
    for i in l_tickers:

        print(i)
        if n == 1:

            df_stock = df[df.ticker == i]

            df_stock.date = pd.to_datetime(df_stock.date)
            df_stock = df_stock.set_index("date")

            df_stock = df_stock[['open', 'high', 'low', 'close', 'volume']]
            df_stock = df_stock.rename(
                columns={'open': i + '_Open', 'high': i + '_High', 'low': i + '_Low', 'close': i + '_Close',
                         'volume': i + '_Volume'})

        else:
            df_stock_new = df[df.ticker == i]

            df_stock_new.date = pd.to_datetime(df_stock_new.date)
            df_stock_new = df_stock_new.set_index("date")

            df_stock_new = df_stock_new[['open', 'high', 'low', 'close', 'volume']]
            df_stock_new = df_stock_new.rename(
                columns={'open': i + '_Open', 'high': i + '_High', 'low': i + '_Low', 'close': i + '_Close',
                         'volume': i + '_Volume'})

            df_stock = df_stock.merge(df_stock_new, how='outer', left_index=True, right_index=True)

        n = n + 1

    # get only last n_values and save it to disk
    df_n_last_vaues = df_stock.tail(n_last_values)
    df_n_last_vaues = df_n_last_vaues.dropna(axis='columns')

    df_n_last_vaues.to_csv(output_path, sep=output_sep)


def createDataset(df,predicted_value_index = 3,  history_points = 50):
    #open = 0
    #high = 1
    #low = 2
    #close =3
    #volume = 4

    # dataset
    data = df.values
    data_normaliser = preprocessing.MinMaxScaler()
    data_normalised = data_normaliser.fit_transform(data)

    # using the last {history_points} open close high low volume data points, predict the next open value
    ohlcv_histories_normalised = np.array(
        [data_normalised[i:i + history_points].copy() for i in range(len(data_normalised) - history_points)])
    next_day_open_values_normalised = np.array(
        [data_normalised[:, predicted_value_index][i + history_points].copy() for i in range(len(data_normalised) - history_points)])
    next_day_open_values_normalised = np.expand_dims(next_day_open_values_normalised, -1)

    next_day_open_values = np.array([data[:, predicted_value_index][i + history_points].copy() for i in range(len(data) - history_points)])
    next_day_open_values = np.expand_dims(next_day_open_values, -1)

    y_normaliser = preprocessing.MinMaxScaler()
    y_normaliser.fit(next_day_open_values)

    def calc_ema(values, time_period):
        # https://www.investopedia.com/ask/answers/122314/what-exponential-moving-average-ema-formula-and-how-ema-calculated.asp
        sma = np.mean(values[:, predicted_value_index])
        ema_values = [sma]
        k = 2 / (1 + time_period)
        for i in range(len(his) - time_period, len(his)):
            close = his[i][3]
            ema_values.append(close * k + ema_values[-1] * (1 - k))
        return ema_values[-1]

    technical_indicators = []
    for his in ohlcv_histories_normalised:
        # note since we are using his[3] we are taking the SMA of the closing price
        sma = np.mean(his[:, predicted_value_index])
        macd = calc_ema(his, 12) - calc_ema(his, 26)
        returns = his[:, 3] / shift(his[:,3], 1, cval=np.NaN)
        returns = returns[-1]
        #technical_indicators.append(np.array([sma]))
        technical_indicators.append(np.array([sma,macd,returns]))

    technical_indicators = np.array(technical_indicators)

    tech_ind_scaler = preprocessing.MinMaxScaler()
    technical_indicators_normalised = tech_ind_scaler.fit_transform(technical_indicators)

    assert ohlcv_histories_normalised.shape[0] == next_day_open_values_normalised.shape[0] == \
           technical_indicators_normalised.shape[0]

    return next_day_open_values_normalised, next_day_open_values, ohlcv_histories_normalised, technical_indicators, data_normaliser, y_normaliser

def splitTrainTest(values, n, verbose = 0):
    train, test = values[:n], values[n:]
    if verbose != 0:
        print('Shape of training dataset: ' + str(train.shape))
        print('Shape of testing dataset: ' + str(test.shape))

    return train, test

def defineModel(ohlcv_histories_normalised,technical_indicators, verbose= 0 ):
    # define two sets of inputs
    lstm_input = Input(shape=(ohlcv_histories_normalised.shape[1], ohlcv_histories_normalised.shape[2]), name='lstm_input')
    dense_input = Input(shape=(technical_indicators.shape[1],), name='tech_input')

    # the first branch operates on the first input
    x = LSTM(50, name='lstm_0')(lstm_input)
    x = Dropout(0.2, name='lstm_dropout_0')(x)
    lstm_branch = Model(inputs=lstm_input, outputs=x)

    # the second branch opreates on the second input
    y = Dense(20, name='tech_dense_0')(dense_input)
    y = Activation("relu", name='tech_relu_0')(y)
    y = Dropout(0.2, name='tech_dropout_0')(y)
    technical_indicators_branch = Model(inputs=dense_input, outputs=y)

    # combine the output of the two branches
    combined = concatenate([lstm_branch.output, technical_indicators_branch.output], name='concatenate')

    z = Dense(64, activation="sigmoid", name='dense_pooling')(combined)
    z = Dense(1, activation="linear", name='dense_out')(z)

    # our model will accept the inputs of the two branches and
    # then output a single value
    model = Model(inputs=[lstm_branch.input, technical_indicators_branch.input], outputs=z)
    adam = optimizers.Adam(lr=0.0005)
    model.compile(optimizer=adam, loss='mse')
    if verbose != 0:
        model.summary()
    return model


def print_stock(name):
    print(name)
    return print('hello', name)


def stock_forceasting(i,
                      column,
                      df_filtered,
                      timeseries_evaluation,
                      timeseries_forecasting,
                      l_tickers_unique,
                      n_days,
                      n_forecast,
                      test_split,
                      verbose,
                      d,
                      plot_results,
                      fontsize,
                      batch_size,
                      epochs):

    print('-' * 10 + 'Iteration: ' + str(i) + '/' + str(len(l_tickers_unique)) + '  ticker name:' + column)


    if len(df_filtered.columns) != 5:
        print(colored('-' * 15 + 'Not all columns available', 'red'))


    else:
        # data imputation
        df_filtered = df_filtered.T.fillna(df_filtered.mean(axis=1)).T
        df_filtered = df_filtered.fillna(method='ffill')
        df_filtered = df_filtered.tail(n_days)

        if timeseries_evaluation:
            print('-' * 15 + ' PART I: Timeseries Evaluation for : ' + column)
            print('-' * 20 + 'Transform the timeseries into an supervised learning problem')


            next_day_open_values_normalised, next_day_open_values, ohlcv_histories_normalised, technical_indicators, data_normaliser, y_normaliser = createDataset(
                df_filtered, predicted_value_index = 3)

            # split data into train and test datasets
            print('-' * 20 + 'Split data into train and test datasets')
            n = int(ohlcv_histories_normalised.shape[0] * test_split)
            ohlcv_train, ohlcv_test = splitTrainTest(values=ohlcv_histories_normalised, n=n)
            tech_ind_train, tech_ind_test = splitTrainTest(values=technical_indicators, n=n)
            y_train, y_test = splitTrainTest(values=next_day_open_values_normalised, n=n)
            unscaled_y_train, unscaled_y_test = splitTrainTest(values=next_day_open_values, n=n)

            # model architecture
            print('-' * 20 + 'Design the model')
            model = defineModel(ohlcv_histories_normalised=ohlcv_histories_normalised,
                                technical_indicators=technical_indicators, verbose=verbose)
            # fit model
            print('-' * 20 + 'Fit the model')
            model.fit(x=[ohlcv_train, tech_ind_train], y=y_train, batch_size=batch_size, epochs=epochs, shuffle=True,
                      validation_split=0.1, verbose=verbose)
            model.save('img/backtest_' + str(d) + '_' + column + '_eval_model.h5')

            # evaluate model
            print('-' * 20 + 'Evaluate the model')
            y_test_predicted = model.predict([ohlcv_test, tech_ind_test])
            y_test_predicted = y_normaliser.inverse_transform(y_test_predicted)
            y_predicted = model.predict([ohlcv_histories_normalised, technical_indicators])
            y_predicted = y_normaliser.inverse_transform(y_predicted)
            assert unscaled_y_test.shape == y_test_predicted.shape
            assert unscaled_y_test.shape == y_test_predicted.shape

            metrics = evaluate_all(unscaled_y_test, y_test_predicted)
            #metrics = evaluate_all(next_day_open_values, y_predicted)

            #real_mse = np.mean(np.square(unscaled_y_test - y_test_predicted))
            #scaled_mse = real_mse / (np.max(unscaled_y_test) - np.min(unscaled_y_test)) * 100
            #print('-' * 25 + 'Scaled MSE: ' + str(scaled_mse))
            #scaled_mse_arr.append([d, column, scaled_mse])


            metric = 'mape'
            df_metric = pd.DataFrame.from_dict(metrics)
            df_metric_transposed = df_metric.transpose()
            df_metric.to_csv('data/intermediary/df_metric_' + str(d) + '_' + column + '.csv', sep=';')

            if plot_results:
                # plot the results
                print('-' * 20 + 'Plot the results')
                plt.figure()
                plt.rcParams["figure.figsize"] = (10, 7)
                plt.box(False)
                fig, (ax1, ax2) = plt.subplots(2, 1)
                fig.tight_layout(pad=5.0)


                # fig.suptitle('Horizontally stacked subplots')
                start = 0
                end = -1
                real = ax1.plot(next_day_open_values[start:end], label='real')
                pred = ax1.plot(y_predicted[start:end], label='predicted')

                # set title
                fig.suptitle('Stock price [{stock}] over time. [{metric_name} = {metric_value}]'.format(stock=column,metric_name =metric, metric_value = str(round(df_metric[metric].iloc[0],2))),  fontsize=fontsize)

                # removing all borders
                ax1.spines['top'].set_visible(False)
                ax1.spines['right'].set_visible(False)
                ax1.spines['left'].set_visible(False)
                ax1.spines['bottom'].set_visible(False)


                real = ax2.plot(unscaled_y_test[start:end], label='real')
                pred = ax2.plot(y_test_predicted[start:end], label='predicted')

                ax2.set_xlabel('Days')
                #ax2.set_ylabel('€')

                ax2.spines['top'].set_visible(False)
                ax2.spines['right'].set_visible(False)
                ax2.spines['left'].set_visible(False)
                ax2.spines['bottom'].set_visible(False)

                ax1.legend(['Real', 'Predicted'], frameon=False, fontsize=fontsize)
                plt.savefig('img/backtest_' + str(d) + '_' + column + '_evaluation_'+str(batch_size) +'_' +str(epochs) +'.png', dpi = 300)

        ## forecast
        print('-' * 15 + 'PART III: ' + str(n_forecast) + '-Step-Forward Prediction ')
        for j in range(0, n_forecast):
            print('-' * 17 + 'Starting forecast ' + str(j)  )
            print('-' * 20 + 'Transform the timeseries into an supervised learning problem')
            predicted_values_array = []
            for predicted_value_index in range(0,5):
                print("Predicting {d} ".format(d = predicted_value_index))
                next_day_open_values_normalised, next_day_open_values, ohlcv_histories_normalised, technical_indicators, data_normaliser, y_normaliser = createDataset(
                    df=df_filtered, predicted_value_index= predicted_value_index)

                #if j == 0:
                # initialize the dataset
                print('-' * 20 + 'Initialize certain objects')
                  # model architecture
                print('-' * 25 + 'Design the model')

                model = defineModel(ohlcv_histories_normalised=ohlcv_histories_normalised,
                                    technical_indicators=technical_indicators, verbose=verbose)
                # fit model
                print('-' * 25 + 'Fit the model')
                model.fit(x=[ohlcv_histories_normalised, technical_indicators],
                          y=next_day_open_values_normalised,
                          batch_size=batch_size, epochs=epochs, shuffle=True, validation_split=0.1, verbose=verbose)

                model.save('img/backtest_' + str(d) + '_' + column + '_'+str(batch_size) +'_' +str(epochs) +'_forecasting_model.h5')

                # evaluate model
                print('-' * 20 + 'Predict with the model')
                y_predicted = model.predict([ohlcv_histories_normalised, technical_indicators])
                y_predicted = y_normaliser.inverse_transform(y_predicted)

                if predicted_value_index == 3:
                    # this is the close value
                    y_predicted_close = y_predicted

                print('-' * 20 + 'Creating the result dataset')
                n_output = 1
                # identifying the predicted output
                newValue = y_predicted[-n_output:, 0].flat[0]
                # identifying the date index
                add_dates = [df_filtered.index[-1] + DateOffset(days=x) for x in range(1, n_output + 1)]

                predicted_values_array.append(newValue)


            """
            df_predict_ohlcv = pd.DataFrame(data=np.array([df_filtered.iloc[-1, 0]
            
                                                              , df_filtered.iloc[-1, 1]
                                                              , df_filtered.iloc[-1, 2]
                                                              , newValue
                                                              , df_filtered.iloc[-1, 4]]).reshape(1, 5),
                                            index=add_dates[0:n_output], columns=df_filtered.columns)
            """

            df_predict_ohlcv = pd.DataFrame(data=np.array(predicted_values_array).reshape(1,5), index=add_dates[0:n_output], columns=df_filtered.columns)
            df_filtered = df_filtered.append(df_predict_ohlcv, sort=False)



        # initialize the result dataset
        # We need to initialize these values here because they depend on the firsts computations
        if 'df_result' not in locals():
            print('-' * 20 + 'Iteration: ' + str(i) + '   Initialize the result dataset')
            global df_result
            df_result = pd.DataFrame(index=df_filtered.index)

        print('-' * 15 + ' Creating the result dataset')
        df_predicted = pd.DataFrame(data=y_predicted_close, index=df_filtered.tail(len(y_predicted)).index,
                                    columns=[column + '/prediction'])

        # add ohlcv columns to the dataset
        df_result = df_result.join(df_filtered)
        # add model prediction to the dataset
        df_result = df_result.join(df_predicted)

        # save to disk
        print('-' * 10 + ' Save results to disk Backtest number: ' + str(d) + ' as: data/intermediary/df_result_' + str(d) + '_' + column + '.csv')
        df_result.to_csv('data/intermediary/df_result_' + str(d) + '_' + column + '_'+str(batch_size) +'_' +str(epochs) +'.csv', sep=';')
        #plot_model(model, to_file='img/model_lstm_{name}.png'.format(name = column), show_shapes=True, show_layer_names=True)

        if plot_results:

            from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
            from mpl_toolkits.axes_grid1.inset_locator import mark_inset


            # previously we filled the ohlv values with their last value and the c (close) value with the prediction.
            # Now we temporarily remove those values again.
            # Alternativly we could have used the input dataset, but it now has been enriched with the forecast results already.
            last_n_days = 5
            df_plot = df_filtered.head(len(df_filtered)-n_forecast).tail(last_n_days)
            df_plot_predicted = df_predicted.tail(last_n_days+n_forecast)

            print('-' * 15 + ' Plot the results of the ' + str(n_forecast) + '-Step-Forward Prediction ')
            plt.figure(figsize=(22, 10))
            plt.box(False)
            plt.plot(df_plot.index,  df_plot[df_plot.columns[0]])
            plt.plot(df_plot.index, df_plot[df_plot.columns[1]])
            plt.plot(df_plot.index, df_plot[df_plot.columns[2]])
            plt.plot(df_plot.index, df_plot[df_plot.columns[3]])
            plt.plot(df_plot_predicted.index, df_plot_predicted)

            #plt.plot(df_plot_predicted.index, df_plot_predicted[df_plot_predicted.columns[0]])

            # plt.plot(df_plot.index, df_plot['Prediction_Future'], color='r')
            # plt.plot(df_proj.index, df_proj['Prediction'], color='y')
            plt.legend(
                [df_plot.columns[0], df_plot.columns[1], df_plot.columns[2], df_plot.columns[3],
                 df_predicted.columns[0]
                 ], frameon=False)
            plt.xticks(fontsize=18)
            plt.yticks(fontsize=16)
            #plt.show()

            plt.savefig('img/backtest_' + str(d) + '_' + column + '_' +str(batch_size) +'_' +str(epochs) +'.png' ,dpi = 300)

            df_plot = df_filtered.head(len(df_filtered) - n_forecast)
            df_plot_predicted =df_predicted

            print('-' * 15 + ' Plot the results of the ' + str(n_forecast) + '-Step-Forward Prediction ')
            plt.figure(figsize=(22, 10))
            plt.box(False)
            plt.plot(df_plot.index, df_plot[df_plot.columns[0]])
            plt.plot(df_plot.index, df_plot[df_plot.columns[1]])
            plt.plot(df_plot.index, df_plot[df_plot.columns[2]])
            plt.plot(df_plot.index, df_plot[df_plot.columns[3]])
            plt.plot(df_plot_predicted.index, df_plot_predicted)

            # plt.plot(df_plot_predicted.index, df_plot_predicted[df_plot_predicted.columns[0]])

            # plt.plot(df_plot.index, df_plot['Prediction_Future'], color='r')
            # plt.plot(df_proj.index, df_proj['Prediction'], color='y')
            plt.legend(
                [df_plot.columns[0], df_plot.columns[1], df_plot.columns[2], df_plot.columns[3],
                 df_predicted.columns[0]
                 ], frameon=False)
            plt.xticks(fontsize=18)
            plt.yticks(fontsize=16)
            # plt.show()

            plt.savefig('img/backtest_' + str(d) + '_' + column + '_' + str(batch_size) + '_' + str(epochs) + '_full.png',  dpi = 300)















            print('-' * 15 + ' Plot the origibal timeseries')
            plt.figure(figsize=(22, 10))
            plt.box(False)
            plt.plot(df_plot.index, df_plot[df_plot.columns[3]])

            # plt.plot(df_plot.index, df_plot['Prediction_Future'], color='r')
            # plt.plot(df_proj.index, df_proj['Prediction'], color='y')
            plt.legend(
                [ df_plot.columns[3]], frameon=False)
            plt.xticks(fontsize=18)
            plt.yticks(fontsize=16)
            # plt.show()

            plt.savefig(
                'img/backtest_' + str(d) + '_' + column + '_' + str(batch_size) + '_' + str(epochs) + '_close.png',
                dpi=300)


        # clean up
        del df_predict_ohlcv
        del add_dates
        del newValue
        del y_predicted
        del model
        del next_day_open_values_normalised, next_day_open_values, ohlcv_histories_normalised, technical_indicators, data_normaliser, y_normaliser
        del df_filtered

        # collect and remove variables from garbage colector and thereby free up memory
        gc.collect()




def defineAutoencoder(num_stock, encoding_dim = 5, verbose=0):

    # connect all layers
    input = Input(shape=(num_stock,))

    encoded = Dense(encoding_dim, kernel_regularizer=regularizers.l2(0.00001),name ='Encoder_Input')(input)

    decoded = Dense(num_stock, kernel_regularizer=regularizers.l2(0.00001), name ='Decoder_Input')(encoded)
    decoded = Activation("linear", name='Decoder_Activation_function')(decoded)

    # construct and compile AE model
    autoencoder = Model(inputs=input, outputs=decoded)
    adam = optimizers.Adam(lr=0.0005)
    autoencoder.compile(optimizer=adam, loss='mean_squared_error')
    if verbose!= 0:
        autoencoder.summary()

    return autoencoder

 # Use those parameters to sample new points from the latent space:
# reparameterization trick
# instead of sampling from Q(z|X), sample epsilon = N(0,I)
# z = z_mean + sqrt(var) * epsilon
def sampling(args):
    """Reparameterization trick by sampling from an isotropic unit Gaussian.
    # Arguments
        args (tensor): mean and log of variance of Q(z|X)
    # Returns
        z (tensor): sampled latent vector
    """

    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    # by default, random_normal has mean = 0 and std = 1.0
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon


def defineVariationalAutoencoder(original_dim, intermediate_dim, latent_dim, verbose=0):
    input_shape = (original_dim,)

    # Map inputs to the latent distribution parameters:
    # VAE model = encoder + decoder
    # build encoder model
    inputs = Input(shape=input_shape, name='encoder_input')
    x = Dense(intermediate_dim, activation='relu')(inputs)
    z_mean = Dense(latent_dim, name='z_mean')(x)
    z_log_var = Dense(latent_dim, name='z_log_var')(x)

    # use reparameterization trick to push the sampling out as input
    # note that "output_shape" isn't necessary with the TensorFlow backend
    z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])

    # Instantiate the encoder model:
    encoder = Model(inputs, z, name='encoder')

    # Build the decoder model:
    latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
    x = Dense(intermediate_dim, activation='relu')(latent_inputs)
    outputs = Dense(original_dim, activation='sigmoid')(x)

    # Instantiate the decoder model:
    decoder = Model(latent_inputs, outputs, name='decoder')

    # Instantiate the VAE model:
    outputs = decoder(encoder(inputs))
    vae = Model(inputs, outputs, name='vae_mlp')

    # As in the Keras tutorial, we define a custom loss function:
    def vae_loss(x, x_decoded_mean):
        xent_loss = binary_crossentropy(x, x_decoded_mean)
        kl_loss = - 0.5 * K.mean(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
        return xent_loss + kl_loss

    # We compile the model:
    vae.compile(optimizer='rmsprop', loss=vae_loss)

    if verbose!= 0:
        vae.summary()

    return vae, decoder,encoder



def predictAutoencoder(autoencoder, data):
    # train autoencoder
    autoencoder.fit(data, data, shuffle=True, epochs=500, batch_size=50)
    # test/reconstruct market information matrix
    reconstruct = autoencoder.predict(data)

    return reconstruct


def getAverageReturns(df, index, days=None):
    '''

    :param df:
    :param index:
    :param days:
    :return:
    '''

    if days == None:
        average = np.average(df.iloc[:, index])*100
    else:
        average =  np.average(df.iloc[-days:, index])*100

    return average


def getAverageReturnsDF(stock_names , df_pct_change, df_result_close,df_original, forecasting_days, backtest_iteration):

    stocks_ranked = []

    stock_index = 0
    for stock_name in stock_names:
        stocks_ranked.append([   backtest_iteration
                                 , df_pct_change.iloc[:, stock_index].name
                                 , getAverageReturns(df=df_pct_change, index=stock_index)
                                 , getAverageReturns(df=df_pct_change, index=stock_index, days=10)
                                 , getAverageReturns(df=df_pct_change, index=stock_index, days=50)
                                 , getAverageReturns(df=df_pct_change, index=stock_index, days=100)
                                 , df_result_close.iloc[-forecasting_days - 1:, stock_index].head(1).iloc[0]
                                 ,df_original[df_pct_change.iloc[:, stock_index].name + '_Close'].tail(forecasting_days * backtest_iteration - forecasting_days + 1).iloc[0]])
        stock_index = stock_index + 1


    columns = ['backtest_iteration','stock_name', 'avg_returns', 'avg_returns_last10_days',
               'avg_returns_last50_days', 'avg_returns_last100_days', 'current_price','value_after_x_days']

    df = pd.DataFrame(stocks_ranked, columns=columns)
    df['delta'] = df['value_after_x_days'] - df['current_price']

    df = df.set_index('stock_name')
    return df




def getReconstructionErrorsDF(df_pct_change, reconstructed_data):
    array = []
    stocks_ranked = []
    num_columns = reconstructed_data.shape[1]
    for i in range(0, num_columns):
        diff = np.linalg.norm((df_pct_change.iloc[:, i] - reconstructed_data[:, i]))  # 2 norm difference
        array.append(float(diff))

    ranking = np.array(array).argsort()
    r = 1
    for stock_index in ranking:
        stocks_ranked.append([ r
                              ,stock_index
                              ,df_pct_change.iloc[:, stock_index].name
                              ,array[stock_index]
                              ])
        r = r + 1

    columns = ['ranking','stock_index', 'stock_name' ,'recreation_error']
    df = pd.DataFrame(stocks_ranked, columns=columns)
    df = df.set_index('stock_name')
    return df

def getLatentFeaturesSimilariryDF(df_pct_change, latent_features, sorted = True):
    stocks_latent_feature = []
    array = []
    num_columns = latent_features.shape[0]
    #for i in range(0, num_columns - 1):
    for i in range(0, num_columns ):
        l2norm = np.linalg.norm(latent_features[i, :])  # 2 norm difference
        array.append(float(l2norm))



    stock_index = 0
    for similarity_score in array:
        stocks_latent_feature.append([stock_index
                                 ,similarity_score
                                 , df_pct_change.iloc[:, stock_index].name
                                 , ])
        stock_index = stock_index + 1

    columns = ['stock_index',
               'similarity_score'
               ,'stock_name']
    df = pd.DataFrame(stocks_latent_feature, columns=columns)
    df = df.set_index('stock_name')

    if sorted == True:
        df = df.sort_values(by=['similarity_score'], ascending=True)
    return df




def portfolio_selection(d,df_portfolio , ranking_colum ,  n_stocks_per_bin, budget ,n_bins,group_by = True):
    n_stocks_total = n_stocks_per_bin * n_bins
    if group_by == True:
        df_portfolio['rn'] = df_portfolio.sort_values([ranking_colum], ascending=[False]) \
                                         .groupby(['similarity_score_quartile']) \
                                         .cumcount() + 1

        df_portfolio_selected_stocks = df_portfolio[df_portfolio['rn'] <= n_stocks_per_bin]
        print(df_portfolio_selected_stocks.__len__())
    else:
        df_portfolio['rn'] = df_portfolio[ranking_colum].rank(method='max', ascending=False)
        df_portfolio_selected_stocks = df_portfolio[df_portfolio['rn'] <= n_stocks_total]
        print(df_portfolio_selected_stocks.__len__())

    var_names = ['x' + str(i) for i in range(n_stocks_total)]
    x_int = [pulp.LpVariable(i, lowBound=0, cat='Integer') for i in var_names]

    my_lp_problem = pulp.LpProblem("Portfolio_Selection_LP_Problem", pulp.LpMaximize)
    # Objective function
    my_lp_problem += lpSum([x_int[i] * df_portfolio_selected_stocks['current_price'].iloc[i] for i in range(n_stocks_total)]) <= budget
    # Constraints
    my_lp_problem += lpSum(
        [x_int[i] * df_portfolio_selected_stocks['current_price'].iloc[i] for i in range(n_stocks_total)])

    for i in range(n_stocks_total):
        my_lp_problem += x_int[i] * df_portfolio_selected_stocks['current_price'].iloc[
            i] <= (budget / n_stocks_total) * 0.5

    my_lp_problem.solve()
    pulp.LpStatus[my_lp_problem.status]


    bought_volume_arr = []
    for variable in my_lp_problem.variables():
        print("{} = {}".format(variable.name, variable.varValue))
        bought_volume_arr.append(variable.varValue)

    df_portfolio_selected_stocks['bought_volume'] = bought_volume_arr


    df_portfolio_selected_stocks['pnl'] = df_portfolio_selected_stocks['delta'] * df_portfolio_selected_stocks['bought_volume']
    print('Profit for iteration ' + str(d) + ': '  + str(df_portfolio_selected_stocks['pnl'].sum()))

    profits = [df_portfolio_selected_stocks['pnl'].sum()]

    return profits , df_portfolio_selected_stocks



def calcMarkowitzPortfolio(df, budget, S,target = None, type = 'max_sharpe', frequency=252, cov_type=None):

    '''
    :param df:  dataframe of returns
    :param budget: budegt to be allocated to the portfolio
    :param S: Covariance matrix
    :param frequency: annual trading days
    :return: returns discrete allocation of stocks, discrete leftover and cleaned weights of stocks in the portfolio
    '''

    # Calculate expected returns and sample covariance
    mu = expected_returns.mean_historical_return(df)
    S = S * frequency
    # Optimise for maximal Sharpe ratio
    if cov_type == 'adjusted':
        S_unadjusted = risk_models.sample_cov(df)
        ef = EfficientFrontier(expected_returns=mu, cov_matrix = S, cov_matrix_unadjusted = S_unadjusted, cov_type = cov_type)
    if cov_type == 'ledoit_wolf':
        cs = risk_models.CovarianceShrinkage(df, frequency=252)
        S = cs.ledoit_wolf(shrinkage_target='constant_variance')
        ef = EfficientFrontier(expected_returns=mu, cov_matrix=S)
    else:
        ef = EfficientFrontier(expected_returns=mu, cov_matrix = S)


    if type == 'efficient_return':
        weights = ef.efficient_return(target_return=target)
    if type == 'max_sharpe':
        weights = ef.max_sharpe()
    if type == 'efficient_risk':
        weights = ef.efficient_risk(target_risk=target)

    cleaned_weights = ef.clean_weights()
    results = ef.portfolio_performance(verbose=True)

    latest_prices = get_latest_prices(df)
    da = DiscreteAllocation(weights, latest_prices, total_portfolio_value=budget)
    try:
        discrete_allocation, discrete_leftover = da.lp_portfolio()
    except:
        discrete_allocation, discrete_leftover = da.greedy_portfolio()

    return discrete_allocation, discrete_leftover, weights, cleaned_weights , mu, S, results





def calc_delta_matrix(self):
    numeric_df = self._get_numeric_data()
    cols = self.shape[0]
    idx = numeric_df.index
    mat = numeric_df.values

    K = cols
    delta_mat = np.empty((K, K), dtype=float)
    mask = np.isfinite(mat)
    for i, ac in enumerate(mat):
        for j, bc in enumerate(mat):
            if i > j:
                continue

            valid = mask[i] & mask[j]
            if i == j:
                c = 0
            elif not valid.all():
                c = ac[valid] - bc[valid]

            else:
                c = ac - bc
            delta_mat[i, j] = c
            delta_mat[j, i] = c

    df_delta_mat = pd.DataFrame(data=delta_mat, columns=idx, index=idx)

    return df_delta_mat



def column(matrix, i):
    return [row[i] for row in matrix]


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]




def generate_outlier_series(median=630, err=12, outlier_err=100, size=80, outlier_size=10):
    errs = err * np.random.rand(size) * np.random.choice((-1, 1), size)
    data = median + errs

    lower_errs = outlier_err * np.random.rand(outlier_size)
    lower_outliers = median - err - lower_errs

    upper_errs = outlier_err * np.random.rand(outlier_size)
    upper_outliers = median + err + upper_errs

    data = np.concatenate((data, lower_outliers, upper_outliers))
    np.random.shuffle(data)

    return data


# Test case 1
test_df = pd.DataFrame(data={"a":[ 1,2,3,4]})
test_df_change = test_df.pct_change(1)

'''
#Test Case 2
test_df = pd.DataFrame(data={"a":[-1,-2,-3,-4]})
test_df_change = test_df.pct_change(1)

# Teast Case 3
test_df = pd.DataFrame(data={"a":[1,0, 1,2,3,4]})
test_df_change = test_df.pct_change(1)
'''

def convert_relative_changes_to_absolute_values(relative_values, initial_value):
    reconstructed_series = (1 + relative_values).cumprod() * initial_value
    reconstructed_series.iloc[0] =initial_value

    return reconstructed_series


reconstructed_series =convert_relative_changes_to_absolute_values(relative_values=test_df_change, initial_value=test_df.iloc[0,0])


def append_to_portfolio_results(array, d, portfolio_type, discrete_allocation, results):
    array.append(
        {
            'backtest_iteration': d,
            'portfolio_type': portfolio_type,
            'expected_annual_return': results[0],
            'annual_volatility': results[1],
            'sharpe_ratio': results[2],
            'discrete_allocation': discrete_allocation
        }
    )
    return array


def plot_backtest_results(df, column, colors, title):
    '''
    plt.rcParams["figure.figsize"] = [16, 9]
    df = df.pivot_table(column, index='backtest_iteration', columns='portfolio_type')
    df = df.sort_values(by=['backtest_iteration'], ascending=False)
    ax = df.plot.bar(rot=0,ylabel= column )
    plt.savefig('img/backtest_results_{i}.png'.format(i=str(column)))
    '''

    # set width of bar
    barWidth = 0.25

    df = df.pivot_table(column, index='backtest_iteration', columns='portfolio_type')
    df = df.sort_values(by=['backtest_iteration'], ascending=False)

    plt.figure(figsize=(12, 7))
    plt.box(False)
    # set height of bar
    bars1 = df.iloc[:, 0].values

    # Set position of bar on X axis
    r1 = np.arange(len(bars1))
    plt.bar(r1, bars1, color=colors[0], width=barWidth, edgecolor='white', label=df.columns[0])

    try:
        bars2 = df.iloc[:, 1].values
        r2 = [x + barWidth for x in r1]
        plt.bar(r2, bars2, color=colors[1], width=barWidth, edgecolor='white', label=df.columns[1])
    except: print(' ')
    try:
        bars3 = df.iloc[:, 2].values
        r3 = [x + barWidth for x in r2]
        plt.bar(r3, bars3, color=colors[2], width=barWidth, edgecolor='white', label=df.columns[2])

    except:
        print(' ')
    try:
        bars4 = df.iloc[:, 3].values
        r4 = [x + barWidth for x in r3]
        plt.bar(r4, bars4, color=colors[3], width=barWidth, edgecolor='white', label=df.columns[3])

    except:
        print(' ')

    # Make the plot

    # Add xticks on the middle of the group bars
    plt.xlabel('backtest iterations', fontweight='bold')
    plt.xticks([r + barWidth for r in range(len(bars1))], df.index)

    # Create legend & Show graphic
    plt.legend(frameon=False)
    plt.title(title)
    plt.savefig('img/backtest_results_{i}.png'.format(i=str(column)), dpi=300)
    plt.close()

    def portfolio_selection(d, number_of_stocks, n_forecast, callbacks_list, epochs, verbose, batch_size, plot_results):
        import numpy as np
        import matplotlib.pyplot as plt

        from src.util import defineVariationalAutoencoder \
            , getReconstructionErrorsDF \
            , Model \
            , defineAutoencoder \
            , getLatentFeaturesSimilariryDF \
            , calc_delta_matrix \
            , calcMarkowitzPortfolio, \
            append_to_portfolio_results

        import pandas as pd
        from sklearn import preprocessing

        portfolio_results = []
        markowitz_allocation = []
        df_results_markowitz_allocation = pd.DataFrame()
        df_results_portfolio = pd.DataFrame()
        new_columns = []
        portfolio_results_temp = []

        budget = 100000
        hidden_layers_latent = 20
        target_annual_return = 0.50

        # get full dataset
        df_original = pd.read_csv('data/historical_stock_prices_original.csv', sep=';')
        df_original.index = pd.to_datetime(df_original.date)

        # get backtest iteration dataset with forecasted values
        df_result = pd.read_csv('data/df_result_' + str(d) + '.csv', sep=';', index_col='Unnamed: 0',
                                parse_dates=True)
        ## Deep Portfolio
        print('-' * 15 + 'PART IV: Autoencoder Deep Portfolio Optimization')
        print('-' * 20 + 'Create dataset')
        df_result_close = df_result.filter(like='Close', axis=1)
        df_original_close_full = df_original.filter(like='Close', axis=1)

        new_columns = []
        [new_columns.append(c.split('_')[0]) for c in df_result_close.columns]
        df_result_close.columns = new_columns

        new_columns = []
        [new_columns.append(c.split('_')[0]) for c in df_original_close_full.columns]
        df_original_close_full.columns = new_columns
        df_original_close = df_original_close_full.iloc[:, :number_of_stocks]

        print('-' * 20 + 'Data Cleaning: Check if all values are positive')
        try:
            assert len(df_result_close[df_result_close >= 0].dropna(axis=1).columns) == len(df_result_close.columns)
        except Exception as exception:
            # Output unexpected Exceptions.
            print('Dataframe contains negative and zero numbers. Replacing them with 0')
            df_result_close = df_result_close[df_result_close >= 0].dropna(axis=1)

        try:
            assert len(df_original_close[df_original_close >= 0].dropna(axis=1).columns) == len(
                df_original_close.columns)
        except Exception as exception:
            # Output unexpected Exceptions.
            print('Dataframe contains negative and zero numbers. Replacing them with 0')
            df_original_close = df_original_close[df_original_close >= 0].dropna(axis=1)

        print('-' * 20 + 'Transform dataset')
        df_pct_change = df_result_close.pct_change(1).astype(float)
        df_pct_change = df_pct_change.replace([np.inf, -np.inf], np.nan)
        df_pct_change = df_pct_change.fillna(method='ffill')
        # the percentage change function will make the first two rows equal to nan
        df_pct_change = df_pct_change.tail(len(df_pct_change) - 2)

        df_pct_change_original = df_original_close.pct_change(1).astype(float)
        df_pct_change_original = df_pct_change_original.replace([np.inf, -np.inf], np.nan)
        df_pct_change_original = df_pct_change_original.fillna(method='ffill')
        # the percentage change function will make the first two rows equal to nan
        df_pct_change_original = df_pct_change_original.tail(len(df_pct_change_original) - 2)

        df_pct_change_original_full = df_original_close_full.pct_change(1).astype(float)
        df_pct_change_original_full = df_pct_change_original_full.replace([np.inf, -np.inf], np.nan)
        df_pct_change_original_full = df_pct_change_original_full.fillna(method='ffill')
        # the percentage change function will make the first two rows equal to nan
        df_pct_change_original_full = df_pct_change_original_full.tail(len(df_pct_change_original_full) - 2)

        # -------------------------------------------------------
        #           Step2: Variational Autoencoder Model
        # -------------------------------------------------------
        print('-' * 25 + 'Apply MinMax Scaler')
        df_scaler = preprocessing.MinMaxScaler()
        df_pct_change_normalised = df_scaler.fit_transform(df_pct_change)

        print('-' * 25 + 'Define variables')
        x = np.array(df_pct_change_normalised)
        input_dim = x.shape[1]
        timesteps = x.shape[0]

        print('-' * 25 + 'Define Variational Autoencoder Model')
        var_autoencoder, var_decoder, var_encoder = defineVariationalAutoencoder(original_dim=input_dim,
                                                                                 intermediate_dim=300,
                                                                                 latent_dim=1)

        # plot_model(var_encoder, to_file='img/model_var_autoencoder_encoder.png', show_shapes=True,
        #           show_layer_names=True)
        # plot_model(var_decoder, to_file='img/model_var_autoencoder_decoder.png', show_shapes=True,
        #           show_layer_names=True)

        print('-' * 25 + 'Fit variational autoencoder model')
        var_autoencoder.fit(x, x, callbacks=callbacks_list, batch_size=64, epochs=epochs, verbose=verbose)
        reconstruct = var_autoencoder.predict(x, batch_size=batch_size)

        print('-' * 25 + 'Inverse transform dataset with MinMax Scaler')
        reconstruct_real = df_scaler.inverse_transform(reconstruct)
        df_var_autoencoder_reconstruct_real = pd.DataFrame(data=reconstruct_real, columns=df_pct_change.columns)

        print('-' * 25 + 'Calculate L2 norm as reconstruction loss metric')
        df_recreation_error = getReconstructionErrorsDF(df_pct_change=df_pct_change
                                                        , reconstructed_data=reconstruct_real)
        df_var_autoencoder_reconstruct_real_cov = df_var_autoencoder_reconstruct_real.cov()

        # -------------------------------------------------------
        #           Step2: Similarity Model
        # -------------------------------------------------------

        print('-' * 20 + 'Step 2 : Returns vs. latent feature similarity')
        print('-' * 25 + 'Transpose dataset')

        # change if original dataset should be used instead o cleaned version
        df_latent_feature_input = df_pct_change
        df_pct_change_transposed = df_latent_feature_input.transpose()

        print('-' * 25 + 'Transform dataset with MinMax Scaler')
        df_scaler = preprocessing.MinMaxScaler()
        df_pct_change_transposed_normalised = df_scaler.fit_transform(df_pct_change_transposed)

        # define autoencoder
        print('-' * 25 + 'Define autoencoder model')
        num_stock = len(df_pct_change_transposed.columns)
        autoencoderTransposed = defineAutoencoder(num_stock=num_stock, encoding_dim=hidden_layers_latent,
                                                  verbose=verbose)

        # train autoencoder
        print('-' * 25 + 'Train autoencoder model')
        autoencoderTransposed.fit(df_pct_change_transposed_normalised, df_pct_change_transposed_normalised,
                                  shuffle=False, epochs=epochs, batch_size=batch_size, verbose=verbose)

        # Get the latent feature vector
        print('-' * 25 + 'Get the latent feature vector')
        autoencoderTransposedLatent = Model(inputs=autoencoderTransposed.input,
                                            outputs=autoencoderTransposed.get_layer('Encoder_Input').output)
        # plot_model(autoencoderTransposedLatent, to_file='img/model_autoencoder_2.png', show_shapes=True,
        #           show_layer_names=True)

        # predict autoencoder model
        print('-' * 25 + 'Predict autoencoder model')
        latent_features = autoencoderTransposedLatent.predict(df_pct_change_transposed_normalised)

        print('-' * 25 + 'Calculate L2 norm as similarity metric')
        df_similarity = getLatentFeaturesSimilariryDF(df_pct_change=df_latent_feature_input
                                                      , latent_features=latent_features
                                                      , sorted=False)

        df_latent_feature = pd.DataFrame(latent_features.T, columns=df_latent_feature_input.columns)
        # df_similarity_delta = calc_delta_matrix(df_latent_feature.transpose())
        # the smaller the value, the closer the stocks are related
        df_similarity_delta = calc_delta_matrix(df_similarity['similarity_score'].transpose())
        df_similarity_cov = df_latent_feature.cov()
        df_similarity_corr = df_latent_feature.corr()

        # normalize between 0 and 1
        min = np.min(df_similarity_cov)
        max = np.max(df_similarity_cov)
        df_similarity_cov_normalized = (df_similarity_cov - min) / (max - min)

        # calculate covariance and correlation matrix of stocks
        df_pct_change_cov = df_pct_change.cov()
        df_pct_change_corr = df_pct_change.corr()

        # plots for 1. Selection of least volatile stocks using autoencoder latent feature value
        if plot_results:
            stable_stocks = False
            unstable_stocks = True
            plot_original_values = False
            plot_delta_values = True
            number_of_stable_unstable_stocks = 20

            df_stable_stocks = df_recreation_error.sort_values(by=['recreation_error'], ascending=True).head(
                number_of_stable_unstable_stocks)
            df_stable_stocks['recreation_error_class'] = 'top ' + str(number_of_stable_unstable_stocks)
            l_stable_stocks = np.array(df_stable_stocks.head(number_of_stable_unstable_stocks).index)

            df_unstable_stocks = df_recreation_error.sort_values(by=['recreation_error'], ascending=False).head(
                number_of_stable_unstable_stocks)
            df_unstable_stocks['recreation_error_class'] = 'bottom ' + str(number_of_stable_unstable_stocks)
            print(df_unstable_stocks.head(5))
            l_unstable_stocks = np.array(df_unstable_stocks.head(number_of_stable_unstable_stocks).index)

            '''
            plt.figure(figsize=(11, 6))
            plt.box(False)
            for stock in df_stable_stocks.index:
                plt.plot(df_pct_change.head(500).index, df_pct_change[stock].head(500), label=stock)

            plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False)
            plt.title('Top ' + str(number_of_stable_unstable_stocks) + ' most stable stocks based on recreation error')
            plt.xlabel("Dates")
            plt.ylabel("Returns")
            plt.show()



            # plot unstable stocks
            plt.figure(figsize=(11, 6))
            plt.box(False)
            for stock in df_unstable_stocks.index:
                plt.plot(df_pct_change.head(500).index, df_pct_change[stock].head(500), label=stock)

            plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False)
            plt.title('Top ' + str(number_of_stable_unstable_stocks) + ' most unstable stocks based on recreation error')
            plt.xlabel("Dates")
            plt.ylabel("Returns")
            plt.show()
            '''

            if stable_stocks:
                print('Plotting stable stocks')
                list = l_stable_stocks
                title = 'Original versus autoencoded stock price for low recreation error (stable stocks)'

            if unstable_stocks:
                print('Plotting unstable stocks')
                list = l_unstable_stocks
                title = 'Original versus autoencoded stock price for high recreation error (unstable stocks)'

            '''
            plt.figure()
            plt.rcParams["figure.figsize"] = (8, 14)
            plt.title(title, y=1.08)
            plt.box(False)
            fig, ax = plt.subplots(len(list), 1)

            i = 0
            for stock in list:
                which_stock = df_result_close.columns.get_loc(stock)
                which_stock_name = df_result_close.columns[which_stock,]

                ## plot for comparison
                if plot_original_values:

                    stock_autoencoder_1 = convert_relative_changes_to_absolute_values(
                        relative_values=df_reconstruct_real[stock], initial_value=df_result_close.iloc[
                            2, which_stock])  # the initial value is the second one as the first one is nan because of the delta calculation

                    print('Plotting original values')
                    ax[i].plot(df_result_close.iloc[2:, which_stock])
                    ax[i].plot(df_result_close.index[2:], stock_autoencoder_1[:])

                if plot_delta_values:
                    print('Plotting delta values')
                    ax[i].plot(df_pct_change[stock])
                    ax[i].plot(df_pct_change.index[:], df_reconstruct_real[stock])

                ax[i].legend(['Original ' + str(which_stock_name), 'Autoencoded ' + str(which_stock_name)],
                             frameon=False)

                # set title
                # plt.set_title('Original stock price [{}] versus autoencoded stock price '.format(column), fontsize=fontsize)
                ax[i].spines['top'].set_visible(False)
                ax[i].spines['right'].set_visible(False)
                ax[i].spines['left'].set_visible(False)
                ax[i].spines['bottom'].set_visible(False)
                ax[i].axes.get_xaxis().set_visible(False)

                i = i + 1

            plt.xticks(fontsize=fontsize)
            plt.yticks(fontsize=fontsize)
            plt.show()
            '''

            # Plots for 3. Calculating stock risk for portfolio diversification
            least_similar_stocks = False
            most_similar_stocks = True

            example_stock_names = df_pct_change.columns  # 'AMZN'
            for example_stock_name in example_stock_names[0:30]:
                example_stock_name = 'GOOG'  # 'GOOGL' #'MSFT'#'AAPL'#'AMZN' #
                top_n = 10

                df_pct_change_corr_most_example = df_pct_change_corr[[example_stock_name]].sort_values(
                    by=[example_stock_name], ascending=False).head(top_n)
                df_pct_change_corr_least_example = df_pct_change_corr[[example_stock_name]].sort_values(
                    by=[example_stock_name], ascending=False).tail(top_n)

                df_similarity_most_example = df_similarity_corr[[example_stock_name]].sort_values(
                    by=[example_stock_name],
                    ascending=False).head(top_n)
                df_similarity_least_example = df_similarity_corr[[example_stock_name]].sort_values(
                    by=[example_stock_name],
                    ascending=False).tail(top_n)

                least_stock_cv = df_pct_change_corr_least_example.head(1).index.values[0]
                most_stock_cv = df_pct_change_corr_most_example.iloc[[1]].index.values[0]

                least_stock_ae = df_similarity_least_example.head(1).index.values[0]
                most_stock_ae = df_similarity_most_example.iloc[[1]].index.values[0]

                # Plot original series for comparison
                df_plot = df_result_close.tail(50)
                plt.figure()
                plt.rcParams["figure.figsize"] = (30, 12)
                plt.box(False)
                fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2)
                # fig.tight_layout(pad=10.0)
                fig.suptitle(
                    'Baseline stock: ' + example_stock_name + ' compared to least (left) and most (right) related stocks',
                    y=1)

                ax1.plot(df_plot.index, df_plot[example_stock_name], label=example_stock_name)
                ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False)
                # removing all borders
                ax1.spines['top'].set_visible(False)
                ax1.spines['right'].set_visible(False)
                ax1.spines['left'].set_visible(False)
                ax1.spines['bottom'].set_visible(False)

                ax3.plot(df_plot.index, df_plot[least_stock_cv], label=least_stock_cv + '(covariance)')
                ax3.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False)
                # removing all borders
                ax3.spines['top'].set_visible(False)
                ax3.spines['right'].set_visible(False)
                ax3.spines['left'].set_visible(False)
                ax3.spines['bottom'].set_visible(False)

                ax5.plot(df_plot.index, df_plot[least_stock_ae], label=least_stock_ae + '(latent feature)')
                ax5.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False)
                # removing all borders
                ax5.spines['top'].set_visible(False)
                ax5.spines['right'].set_visible(False)
                ax5.spines['left'].set_visible(False)
                ax5.spines['bottom'].set_visible(False)

                ax2.plot(df_plot.index, df_plot[example_stock_name], label=example_stock_name)
                ax2.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False)
                # removing all borders
                ax2.spines['top'].set_visible(False)
                ax2.spines['right'].set_visible(False)
                ax2.spines['left'].set_visible(False)
                ax2.spines['bottom'].set_visible(False)

                ax4.plot(df_plot.index, df_plot[most_stock_cv],
                         label=most_stock_cv + '(covariance)')
                ax4.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False)
                # removing all borders
                ax4.spines['top'].set_visible(False)
                ax4.spines['right'].set_visible(False)
                ax4.spines['left'].set_visible(False)
                ax4.spines['bottom'].set_visible(False)

                ax6.plot(df_plot.index, df_plot[most_stock_ae],
                         label=most_stock_ae + '(latent feature)')
                ax6.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False)
                # removing all borders
                ax6.spines['top'].set_visible(False)
                ax6.spines['right'].set_visible(False)
                ax6.spines['left'].set_visible(False)
                ax6.spines['bottom'].set_visible(False)

                plt.xlabel("Dates")
                plt.ylabel("Stock Value")
                plt.savefig(
                    'img/{v_d}_similarity_comparision_stock_{v_stock}.png'.format(v_d=d, v_stock=example_stock_name))

            '''
              # Plots for 3. compare original timeseries with latent features
              plt.figure()
              plt.rcParams["figure.figsize"] = (18, 10)
              plt.box(False)
              fig, ((ax1),(ax2)) = plt.subplots(2, 1)
              # fig.tight_layout(pad=10.0)
              fig.suptitle(
                  'Orignal stock (top): ' + example_stock_name + ' compared to least (left) and most (right) related stocks',
                  y=1)

              ax1.plot(df_pct_change.index, df_pct_change[example_stock_name], label=example_stock_name)
              ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False)
              # removing all borders
              ax1.spines['top'].set_visible(False)
              ax1.spines['right'].set_visible(False)
              ax1.spines['left'].set_visible(False)
              ax1.spines['bottom'].set_visible(False)

              ax2.plot(df_latent_feature.index, df_latent_feature[example_stock_name], label=example_stock_name)
              ax2.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False)
              # removing all borders
              ax2.spines['top'].set_visible(False)
              ax2.spines['right'].set_visible(False)
              ax2.spines['left'].set_visible(False)
              ax2.spines['bottom'].set_visible(False)

              plt.xlabel("Dates")
              plt.ylabel("Stock Value")
              '''

        # -------------------------------------------------------
        #           Step3: Markowitz Model
        # -------------------------------------------------------

        print('-' * 20 + 'Step 3: Create dataset')
        df_result_close = df_result_close[df_pct_change.columns]

        print('-' * 20 + 'Step 3: Markowitz model without forecast values and without preselection')
        discrete_allocation, discrete_leftover, weights, cleaned_weights, mu, S, results = calcMarkowitzPortfolio(
            df=df_original_close_full.head(len(df_original_close_full) - n_forecast * d)
            , budget=budget
            , S=df_pct_change_original_full.cov()
            , type='max_sharpe'
            , target=target_annual_return)

        df_markowitz_allocation_without_forecast_without_preseletcion_full = pd.DataFrame(
            discrete_allocation.items(),
            columns=['stock_name', 'bought_volume_without_forecast_without_preselection_full'])
        df_markowitz_allocation_without_forecast_without_preseletcion_full = df_markowitz_allocation_without_forecast_without_preseletcion_full.set_index(
            'stock_name')

        append_to_portfolio_results(array=portfolio_results_temp,
                                    d=d,
                                    portfolio_type='markowitz_portfolio_baseline_full',
                                    discrete_allocation=discrete_allocation,
                                    results=results)

        print('-' * 20 + 'Step 3: Markowitz model without forecast values and without preselection')
        discrete_allocation, discrete_leftover, weights, cleaned_weights, mu, S, results = calcMarkowitzPortfolio(
            df=df_original_close.head(len(df_original_close) - n_forecast * d)
            , budget=budget
            , S=df_pct_change_original.cov()
            , type='max_sharpe'
            , target=target_annual_return)

        df_markowitz_allocation_without_forecast_without_preseletcion = pd.DataFrame(
            discrete_allocation.items(),
            columns=['stock_name', 'bought_volume_without_forecast_without_preselection'])
        df_markowitz_allocation_without_forecast_without_preseletcion = df_markowitz_allocation_without_forecast_without_preseletcion.set_index(
            'stock_name')

        append_to_portfolio_results(array=portfolio_results_temp,
                                    d=d,
                                    portfolio_type='markowitz_portfolio_baseline',
                                    discrete_allocation=discrete_allocation,
                                    results=results)

        '''
        df_temp = df_markowitz_allocation_without_forecast_without_preseletcion_full.join(df_markowitz_allocation_without_forecast_without_preseletcion, lsuffix='l')
        df_temp['delta'] = df_temp.iloc[:,0] - df_temp.iloc[:,1]

        print('Identical stocks')
        print(df_temp['delta'][df_temp['delta'] == 0].count())

        print('number of stocks selected in smaller dataset not in full dataset')
        print(df_temp['delta'][df_temp['delta'] < 0].count())

        print('number of stocks selected in full dataset not in small dataset')
        print(df_temp['delta'][df_temp['delta'] > 0].count())
        '''

        print('-' * 20 + 'Step 3: Markowitz model without forecast values')
        discrete_allocation, discrete_leftover, weights, cleaned_weights, mu, S, results = calcMarkowitzPortfolio(
            df=df_result_close.head(len(df_result_close) - n_forecast)
            , budget=budget
            , S=df_pct_change_cov
            , type='max_sharpe'
            , target=target_annual_return)
        df_markowitz_allocation_without_forecast = pd.DataFrame(discrete_allocation.items(),
                                                                columns=['stock_name',
                                                                         'bought_volume_without_forecast'])
        df_markowitz_allocation_without_forecast = df_markowitz_allocation_without_forecast.set_index(
            'stock_name')

        append_to_portfolio_results(array=portfolio_results_temp,
                                    d=d,
                                    portfolio_type='markowitz_portfolio_without_forecast',
                                    discrete_allocation=discrete_allocation,
                                    results=results)

        print('-' * 20 + 'Step 3: Markowitz model with forecast')
        discrete_allocation, discrete_leftover, weights, cleaned_weights, mu, S, results = calcMarkowitzPortfolio(
            df=df_result_close
            , budget=budget
            , S=df_pct_change_cov
            , type='max_sharpe'
            , target=target_annual_return)
        df_markowitz_allocation_with_forecast = pd.DataFrame(discrete_allocation.items(),
                                                             columns=['stock_name',
                                                                      'bought_volume_with_forecast'])
        df_markowitz_allocation_with_forecast = df_markowitz_allocation_with_forecast.set_index(
            'stock_name')

        append_to_portfolio_results(array=portfolio_results_temp,
                                    d=d,
                                    portfolio_type='markowitz_portfolio_with_forecast',
                                    discrete_allocation=discrete_allocation,
                                    results=results)

        # cla = CLA(expected_returns=mu, cov_matrix=S, weight_bounds=(0, 1))
        # Plotting.plot_efficient_frontier(cla, points=100, show_assets=True)

        print('-' * 20 + 'Step 3: Markowitz model with cleaned covariance matrix')
        discrete_allocation, discrete_leftover, weights, cleaned_weights, mu, S, results = calcMarkowitzPortfolio(
            df=df_result_close
            , budget=budget
            , S=df_var_autoencoder_reconstruct_real_cov
            , type='max_sharpe'
            , target=target_annual_return
            , cov_type='adjusted')
        df_markowitz_allocation_var_autoencoder = pd.DataFrame(discrete_allocation.items(),
                                                               columns=['stock_name',
                                                                        'bought_volume_with_forecast_cleaned'])
        df_markowitz_allocation_var_autoencoder = df_markowitz_allocation_var_autoencoder.set_index(
            'stock_name')

        append_to_portfolio_results(array=portfolio_results_temp,
                                    d=d,
                                    portfolio_type='markowitz_portfolio_with_forecast_and adjusted covariance_matrix',
                                    discrete_allocation=discrete_allocation,
                                    results=results)

        print('-' * 20 + 'Step 3: Markowitz model with latent features')
        gamma = 0.1

        ''''
        1. df_similarity_cov_normalized[1,2] > df_similarity_cov_normalized[1,3] --> stock 1 and 2 is more similar than stock 1 and 3
        2. S= df_pct_change_cov * df_similarity_cov_normalized and df_pct_change_cov[1,2]=df_pct_change_cov[1,3] --> S[1,2] > S[1,3]
        3. min(S) --> S[1,3] will be considered during optimization more likely then S[1,2] -->  penalize similar stocks more than non-similar stocks

        '''

        try:
            discrete_allocation, discrete_leftover, weights, cleaned_weights, mu, S, results = calcMarkowitzPortfolio(
                df=df_result_close
                , budget=budget
                , S=df_pct_change_cov * df_similarity_cov_normalized
                , type='max_sharpe'
                , target=target_annual_return
                , cov_type='adjusted')
        except RuntimeWarning:
            print(
                'RuntimeWarning: invalid value encountered in sqrt sigma = np.sqrt(np.dot(weights, np.dot(cov_matrix, weights.T)))')
            discrete_allocation, discrete_leftover, weights, cleaned_weights, mu, S, results = calcMarkowitzPortfolio(
                df=df_result_close
                , budget=budget
                , S=df_pct_change_cov
                , type='max_sharpe'
                , target=target_annual_return
                , cov_type='adjusted')

        df_markowitz_allocation_latent_feature = pd.DataFrame(discrete_allocation.items(),
                                                              columns=['stock_name',
                                                                       'bought_volume_with_forecast_latent'])
        df_markowitz_allocation_latent_feature = df_markowitz_allocation_latent_feature.set_index(
            'stock_name')

        append_to_portfolio_results(array=portfolio_results_temp,
                                    d=d,
                                    portfolio_type='markowitz_portfolio_with_forecast_and_latent_features',
                                    discrete_allocation=discrete_allocation,
                                    results=results)

        # Joining portfolio optimization results to one dataframe
        df_markowitz_allocation = df_markowitz_allocation_without_forecast_without_preseletcion_full.join(
            df_markowitz_allocation_without_forecast_without_preseletcion
            , how='outer'
            , lsuffix=''
            , rsuffix='')
        df_markowitz_allocation = df_markowitz_allocation.join(df_markowitz_allocation_with_forecast
                                                               , how='outer'
                                                               , lsuffix=''
                                                               , rsuffix='')

        df_markowitz_allocation = df_markowitz_allocation.join(df_markowitz_allocation_without_forecast
                                                               , how='outer'
                                                               , lsuffix=''
                                                               , rsuffix='')
        df_markowitz_allocation = df_markowitz_allocation.join(df_markowitz_allocation_var_autoencoder
                                                               , how='outer'
                                                               , lsuffix=''
                                                               , rsuffix='')

        df_markowitz_allocation = df_markowitz_allocation.join(df_markowitz_allocation_latent_feature
                                                               , how='outer'
                                                               , lsuffix=''
                                                               , rsuffix='')

        df_result_close_buy_price = df_original_close.tail(n_forecast * d).head(1).transpose()
        df_result_close_buy_price = df_result_close_buy_price.rename(
            columns={df_result_close_buy_price.columns[0]: "buy_price"})

        df_result_close_predicted_price = df_original_close.tail(1).transpose()
        df_result_close_predicted_price = df_result_close_predicted_price.rename(
            columns={df_result_close_predicted_price.columns[0]: "predicted_price"})

        df_result_close_sell_price = df_original_close.head(len(df_original_close) - n_forecast * (d - 1))
        df_result_close_sell_price = df_result_close_sell_price.tail(1).transpose()
        df_result_close_sell_price = df_result_close_sell_price.rename(
            columns={df_result_close_sell_price.columns[0]: "sell_price"})

        df_markowitz_allocation = df_markowitz_allocation.join(df_result_close_buy_price, how='left')
        df_markowitz_allocation = df_markowitz_allocation.join(df_result_close_predicted_price, how='left')
        df_markowitz_allocation = df_markowitz_allocation.join(df_result_close_sell_price, how='left')
        df_markowitz_allocation['backtest_id'] = d

        df_markowitz_allocation['delta'] = df_markowitz_allocation['sell_price'] - df_markowitz_allocation[
            'buy_price']

        df_markowitz_allocation['profit_without_forecast_without_preselection_full'] = df_markowitz_allocation[
                                                                                           'delta'] * \
                                                                                       df_markowitz_allocation[
                                                                                           'bought_volume_without_forecast_without_preselection_full']

        df_markowitz_allocation['profit_without_forecast_without_preselection'] = df_markowitz_allocation[
                                                                                      'delta'] * \
                                                                                  df_markowitz_allocation[
                                                                                      'bought_volume_without_forecast_without_preselection']
        df_markowitz_allocation['profit_without_forecast'] = df_markowitz_allocation['delta'] * \
                                                             df_markowitz_allocation[
                                                                 'bought_volume_without_forecast']
        df_markowitz_allocation['profit_with_forecast'] = df_markowitz_allocation['delta'] * \
                                                          df_markowitz_allocation[
                                                              'bought_volume_with_forecast']
        df_markowitz_allocation['profit_with_forecast_cleaned'] = df_markowitz_allocation['delta'] * \
                                                                  df_markowitz_allocation[
                                                                      'bought_volume_with_forecast_cleaned']
        df_markowitz_allocation['profit_with_forecast_latent'] = df_markowitz_allocation['delta'] * \
                                                                 df_markowitz_allocation[
                                                                     'bought_volume_with_forecast_latent']

        df_results_markowitz_allocation = df_results_markowitz_allocation.append(df_markowitz_allocation)

        df_results_portfolio_temp = pd.DataFrame.from_dict(portfolio_results_temp)

        df_results_portfolio_temp['profit'] = [
            np.sum(df_markowitz_allocation['profit_without_forecast_without_preselection_full']),
            np.sum(df_markowitz_allocation['profit_without_forecast_without_preselection']),
            np.sum(df_markowitz_allocation['profit_without_forecast'])
            , np.sum(df_markowitz_allocation['profit_with_forecast'])
            , np.sum(df_markowitz_allocation['profit_with_forecast_cleaned'])
            , np.sum(df_markowitz_allocation['profit_with_forecast_latent'])
        ]

        # append to final dataset
        df_results_portfolio = df_results_portfolio.append(df_results_portfolio_temp)
        df_results_portfolio.to_csv('data/df_backtest_{}v_dportfolio.csv'.format(v_d=d), sep=';',
                                    columns=df_results_portfolio_temp.columns)

