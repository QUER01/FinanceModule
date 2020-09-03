
if __name__ == "__main__":  # confirms that the code is under main function

    import time
    import matplotlib
    import pydot
    import os
    #from pandas_profiling import ProfileReport
    os.environ["PATH"] += os.pathsep + 'lib/Graphviz2.38/bin/'
    import matplotlib.pyplot as plt
    from keras.models import Model
    import numpy as np

    from keras.callbacks import TensorBoard
    from keras.utils.vis_utils import  plot_model


    from FinanceModule.util import *

    from FinanceModule.quandlModule import Quandl
    import copy
    import pandas as pd
    from sklearn import preprocessing
    from datetime import datetime
    from multiprocessing import Pool, TimeoutError, Process
    import multiprocessing
    import sys

    #from pypfopt import Plotting, CLA
    #plt.xkcd()





    print('-' * 50)
    print('PART I: Timeseries Cleaning')
    print('-' * 50)

    # general parameters
    fontsize = 12
    parallel_processes = multiprocessing.cpu_count() - 1

    # indicate folder to save, plus other options
    date = datetime.now().strftime('%Y-%m-%d_%H_%M')
    tensorboard = TensorBoard(log_dir='./logs/run_' + date)
    # save it in your callback list, where you can include other callbacks
    callbacks_list = [tensorboard]

    # script parameters
    test_setting = False
    plot_results = True
    stock_selection = False
    '''
     Note Multiprocessing cannot run when in the main session a keras backend was created before creating the worker pool. 
     E.g. Time Series Evaluation cannot run in the same script as time series forecasting. 
    '''
    timeseries_evaluation = False
    timeseries_forecasting =True
    portfolio_optimization = False


    verbose = 0

    # 0 Data Preparation
    history_points = 150
    test_split = 0.9
    n_forecast = 10
    n_tickers = 6000
    n_days = 250 * 4
    trading_days = 252
    sectors = ['FINANCE', 'CONSUMER SERVICES', 'TECHNOLOGY',
               'CAPITAL GOODS', 'BASIC INDUSTRIES', 'HEALTH CARE',
               'CONSUMER DURABLES', 'ENERGY', 'TRANSPORTATION', 'CONSUMER NON-DURABLES']


    # 1. Selection of least volatile stocks using autoencoders
    hidden_layers = 5
    batch_size = 500
    epochs = 500
    stock_selection_number = 500

    # 2. Forecasting using recurrent neural networks
    backtest_days = 200
    scaled_mse_arr = []

    if test_setting:
        d = 5
        number_of_stocks = 960


    """
    transformDataset( input_path='data/historical_stock_prices.csv', input_sep=','
                     , metadata_input_path = 'data/historical_stocks.csv', metadata_sep = ','
                     ,output_path='data/historical_stock_prices_original.csv', output_sep=';'
                     ,filter_sectors = sectors
                     ,n_tickers = n_tickers, n_last_values = n_days )
    
    """

    print('-' * 5 + 'Loading the dataset from disk')
    df_original = pd.read_csv('data/historical_stock_prices_original.csv', sep=';', index_col='date')


    df_original.index = pd.to_datetime(df_original.index)



    # 1. Selection of least volatile stocks using autoencoders
    if stock_selection:

        ## Deep Portfolio
        print('-' * 15 + 'PART IV: Autoencoder Deep Portfolio Optimization')
        print('-' * 20 + 'Create dataset')
        df_result_close = df_original.filter(like='Close', axis=1)

        # TODO REMOVE ME
        #if test_setting:
        #    df_result_close = df_result_close.iloc[:, 0:number_of_stocks]

        new_columns = []
        [new_columns.append(c.split('_')[0]) for c in df_result_close.columns]
        df_result_close.columns = new_columns
        df_result_close = df_result_close.dropna(axis=1, how='any', thresh=0.90 * len(df_original))

        print('-' * 20 + 'Transform dataset')
        df = df_result_close

        df_pct_change = df_result_close.pct_change(1).astype(float)
        df_pct_change = df_pct_change.replace([np.inf, -np.inf], np.nan)
        df_pct_change = df_pct_change.fillna(method='ffill')
        # the percentage change function will make the first two rows equal to nan
        df_pct_change = df_pct_change.tail(len(df_pct_change) - 2)

        # remove columns where there is no change over a longer time period
        df_pct_change = df_pct_change[df_pct_change.columns[((df_pct_change == 0).mean() <= 0.05)]]

        # -------------------------------------------------------
        #           Step1: Recreation Error
        # -------------------------------------------------------
        print('-' * 20 + 'Step 1 : Returns vs. recreation error (recreation_error)')
        print('-' * 25 + 'Transform dataset with MinMax Scaler')
        df_scaler = preprocessing.MinMaxScaler()
        df_pct_change_normalised = df_scaler.fit_transform(df_pct_change)

        # define autoencoder
        print('-' * 25 + 'Define autoencoder model')
        num_stock = len(df_pct_change.columns)
        autoencoder = defineAutoencoder(num_stock=num_stock, encoding_dim=hidden_layers, verbose=verbose)
        #plot_model(autoencoder, to_file='img/model_autoencoder_1.png', show_shapes=True,
        #           show_layer_names=True)

        # train autoencoder
        print('-' * 25 + 'Train autoencoder model')
        autoencoder.fit(df_pct_change_normalised, df_pct_change_normalised, shuffle=False, epochs=epochs,
                        batch_size=batch_size,
                        verbose=verbose)

        # predict autoencoder
        print('-' * 25 + 'Predict autoencoder model')
        reconstruct = autoencoder.predict(df_pct_change_normalised)

        # Inverse transform dataset with MinMax Scaler
        print('-' * 25 + 'Inverse transform dataset with MinMax Scaler')
        reconstruct_real = df_scaler.inverse_transform(reconstruct)
        df_reconstruct_real = pd.DataFrame(data=reconstruct_real, columns=df_pct_change.columns)

        print('-' * 25 + 'Calculate L2 norm as reconstruction loss metric')
        df_recreation_error = getReconstructionErrorsDF(df_pct_change=df_pct_change
                                                        , reconstructed_data=reconstruct_real)



        filtered_stocks = df_recreation_error.head(stock_selection_number).index
        df_result_close_filtered = df_result_close[filtered_stocks]
        df_result_close_filtered.to_csv('data/df_result_close_filtered.csv', sep=';')




    df_result_close_filtered = pd.read_csv('data/df_result_close_filtered.csv', sep=';', index_col ='date')


    # Get tickers as a list
    print('-' * 5 + 'Getting list of unique tickers')
    l_tickers_new = df_result_close_filtered.columns.str.split('_')
    l_tickers_unique = np.unique(fun_column(l_tickers_new, 0))
    l_tickers_unique_chunks = list(chunks(l_tickers_unique, parallel_processes))


    # 2. Forecasting using recurrent neural networks
    if timeseries_forecasting:
        #for d in range(5, 8)[::-1]:
        for d in range(int(backtest_days/n_forecast)+1)[::-1]:

            if d != 0 and d > 11:
                print('-' * 5 + 'Backtest Iteration ' + str(d))
                df = df_original.head(len(df_original) - n_forecast * d)
                print('-' * 5 + 'Starting for loop over all tickers')
                j = 0
                for j_val in l_tickers_unique_chunks:
                    print('opening new pool: ' + str(j) + '/' + str(len(l_tickers_unique_chunks)))
                    pool = Pool(processes=parallel_processes)  # start 12 worker processes
                    i = 0
                    for val in j_val:
                        column = val
                        #print(column)
                        df_filtered = df.filter(regex='^' + column + '', axis=1)
                        #stock_forceasting(i, column, df_filtered, timeseries_evaluation, timeseries_forecasting)
                        pool.apply_async(stock_forceasting,
                                         args=(i, column, df_filtered, timeseries_evaluation, timeseries_forecasting, l_tickers_unique, n_days, n_forecast, test_split, verbose, d, plot_results,fontsize))
                        i = i + 1
                    print('closing new pool')
                    pool.close()
                    pool.join()
                    j = j + 1

                for i in range(0, len(l_tickers_unique)):
                    column = l_tickers_unique[i]
                    try:
                        if timeseries_forecasting:
                            df_result_ticker = pd.read_csv('data/intermediary/df_result_' + str(d) + '_' + column + '.csv', sep=';',
                                                           index_col='Unnamed: 0')
                        if timeseries_evaluation:
                            df_scaled_mse_ticker = pd.read_csv('data/intermediary/df_scaled_mse_' + str(d) + '_' + column + '.csv',
                                                               sep=';')

                        if i == 0:
                            if timeseries_forecasting:
                                df_result = pd.DataFrame(index=df_result_ticker.index)

                            if timeseries_evaluation:
                                df_scaled_mse = pd.DataFrame()

                        if timeseries_forecasting:
                            df_result = df_result.join(df_result_ticker)
                        if timeseries_evaluation:
                            df_scaled_mse = pd.concat([df_scaled_mse, df_scaled_mse_ticker])

                    except:
                        print('file not available')

                if timeseries_forecasting:
                    df_result.to_csv('data/df_result_' + str(d) + '.csv', sep=';')
                if timeseries_evaluation:
                    df_scaled_mse.to_csv('data/df_scaled_mse_' + str(d) + '.csv', sep=';')

    # End of for loops
    # 3. Calculating stock risk for portfolio diversification
    # 4.Portfolio optimization using linear programming
    if portfolio_optimization:

        portfolio_results = []
        markowitz_allocation = []
        df_results_markowitz_allocation = pd.DataFrame()
        df_results_portfolio = pd.DataFrame()
        new_columns = []


        # TODO REMOVE ME
        profits_option_1 = []
        tickers_option_1 = []

        profits_option_2 = []
        tickers_option_2 = []

        profits_option_3 = []
        tickers_option_3 = []

        profits_option_4 = []
        tickers_option_4 = []
        # --



        avg_return_column = 'avg_returns_last50_days'
        avg_return_days = 50
        forecasting_days = 10


        n_stocks_per_bin = 2
        budget = 100000
        n_bins = 10

        hidden_layers = 20
        target_annual_return = 0.50


        for d in range(int(backtest_days / n_forecast) + 1)[::-1]:
        #for d in range(3)[::-1]:
            if d != 0:

                print('-' * 5 + 'Backtest Iteration ' + str(d))
                portfolio_results_temp = []

                # get full dataset
                #if 'df_original' not in locals():
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

                # TODO REMOVE ME !!!
                df_original_close = df_original_close_full.iloc[:, : 1000]
                # !!! REMOVE ME END!!!



                if test_setting:
                    df_result_close = df_result_close.iloc[:, 0:number_of_stocks]

                new_columns = []
                [new_columns.append(c.split('_')[0]) for c in df_result_close.columns ]
                df_result_close.columns = new_columns

                new_columns = []
                [new_columns.append(c.split('_')[0]) for c in df_original_close.columns ]
                df_original_close.columns = new_columns

                print('-' * 20 + 'Data Cleaning: Check if all values are positive')
                try:
                    assert len(df_result_close[df_result_close >= 0].dropna(axis=1).columns) == len(df_result_close.columns)
                except Exception as exception:
                    # Output unexpected Exceptions.
                    print('Dataframe contains negative and zero numbers. Replacing them with 0')
                    df_result_close = df_result_close[df_result_close >= 0].dropna(axis=1)

                try:
                    assert len(df_original_close[df_original_close >= 0].dropna(axis=1).columns) == len(df_original_close.columns)
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

                # remove columns where there is no change over a longer time period
                #df_pct_change = df_pct_change[df_pct_change.columns[((df_pct_change == 0).mean() <= 0.05)]]




                # -------------------------------------------------------
                #           Step2: Similarity Model
                # -------------------------------------------------------

                '''
                print('-' * 20 + 'Step 2 : Returns vs. latent feature similarity')
                print('-' * 25 + 'Transpose dataset')
                df_pct_change_transposed = df_pct_change.transpose()

                print('-' * 25 + 'Transform dataset with MinMax Scaler')
                df_scaler = preprocessing.MinMaxScaler()
                df_pct_change_transposed_normalised = df_scaler.fit_transform(df_pct_change_transposed)

                # define autoencoder
                print('-' * 25 + 'Define autoencoder model')
                num_stock = len(df_pct_change_transposed.columns)
                autoencoderTransposed = defineAutoencoder(num_stock=num_stock, encoding_dim=hidden_layers, verbose=verbose)


                # train autoencoder
                print('-' * 25 + 'Train autoencoder model')
                autoencoderTransposed.fit(df_pct_change_transposed_normalised, df_pct_change_transposed_normalised,
                                          shuffle=False, epochs=epochs, batch_size=batch_size, verbose=verbose)

                # Get the latent feature vector
                print('-' * 25 + 'Get the latent feature vector')
                autoencoderTransposedLatent = Model(inputs=autoencoderTransposed.input,
                                                    outputs=autoencoderTransposed.get_layer('Encoder_Input').output)
                #plot_model(autoencoderTransposedLatent, to_file='img/model_autoencoder_2.png', show_shapes=True,
                #           show_layer_names=True)

                # predict autoencoder model
                print('-' * 25 + 'Predict autoencoder model')
                latent_features = autoencoderTransposedLatent.predict(df_pct_change_transposed_normalised)

                print('-' * 25 + 'Calculate L2 norm as similarity metric')
                df_similarity = getLatentFeaturesSimilariryDF(df_pct_change=df_pct_change
                                                            , latent_features=latent_features )

                df_latent_feature = pd.DataFrame(latent_features.T, columns=df_pct_change.columns)
                df_similarity_cov = df_latent_feature.cov()
                df_similarity_cor = df_latent_feature.corr()
                '''


                # -------------------------------------------------------
                #           Step2: Variational Autoencoder Model
                # -------------------------------------------------------

                # NEW
                print('-' * 25 + 'Apply MinMax Scaler')
                df_scaler = preprocessing.MinMaxScaler()
                df_pct_change_normalised = df_scaler.fit_transform(df_pct_change)

                print('-' * 25 + 'Define variables')
                x = np.array(df_pct_change_normalised)
                input_dim = x.shape[1]
                timesteps = x.shape[0]

                print('-' * 25 + 'Define Variational Autoencoder Model')
                var_autoencoder, var_decoder, var_encoder = defineVariationalAutoencoder(original_dim = input_dim,
                                     intermediate_dim = 300,
                                     latent_dim= 2)

                #plot_model(var_encoder, to_file='img/model_var_autoencoder_encoder.png', show_shapes=True,
                #           show_layer_names=True)
                #plot_model(var_decoder, to_file='img/model_var_autoencoder_decoder.png', show_shapes=True,
                #           show_layer_names=True)

                print('-' * 25 + 'Fit variational autoencoder model')
                var_autoencoder.fit(x, x, callbacks=callbacks_list,  batch_size=64, epochs=epochs, verbose=verbose)
                reconstruct = var_autoencoder.predict(x, batch_size=batch_size)

                print('-' * 25 + 'Inverse transform dataset with MinMax Scaler')
                reconstruct_real = df_scaler.inverse_transform(reconstruct)
                df_var_autoencoder_reconstruct_real = pd.DataFrame(data=reconstruct_real, columns=df_pct_change.columns)

                print('-' * 25 + 'Calculate L2 norm as reconstruction loss metric')
                df_recreation_error = getReconstructionErrorsDF(df_pct_change=df_pct_change
                                                                , reconstructed_data=reconstruct_real)
                df_var_autoencoder_reconstruct_real_cov = df_var_autoencoder_reconstruct_real.cov()

                # NEW END
                # TODO REMOVE ME
                '''
                df_returns = getAverageReturnsDF(stock_names=df_pct_change.columns
                                                 , df_pct_change=df_pct_change
                                                 , df_result_close=df_result_close
                                                 , df_original=df_original
                                                 , forecasting_days=forecasting_days
                                                 , backtest_iteration=d)
                '''

                # calculate covariance matrix of stocks
                df_pct_change_corr = df_pct_change.corr()
                df_pct_change_cov = df_pct_change.cov()

                '''
                # plots for 1. Selection of least volatile stocks using autoencoder latent feature value
                if plot_results:
                    stable_stocks = False
                    unstable_stocks = True
                    plot_original_values = False
                    plot_delta_values = True
                    number_of_stable_unstable_stocks = 20


                    df_stable_stocks = df_recreation_error.sort_values(by=['recreation_error'], ascending=True).head(number_of_stable_unstable_stocks)
                    df_stable_stocks['recreation_error_class'] =  'top ' + str(number_of_stable_unstable_stocks)
                    l_stable_stocks = np.array(df_stable_stocks.head(number_of_stable_unstable_stocks).index)

                    plt.figure(figsize=(11, 6))
                    plt.box(False)
                    for stock in df_stable_stocks.index:
                        plt.plot(df_pct_change.head(500).index, df_pct_change[stock].head(500), label=stock)

                    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False)
                    plt.title('Top ' + str(number_of_stable_unstable_stocks) + ' most stable stocks based on recreation error')
                    plt.xlabel("Dates")
                    plt.ylabel("Returns")
                    plt.show()



                    df_unstable_stocks = df_recreation_error.sort_values(by=['recreation_error'], ascending=False).head(number_of_stable_unstable_stocks)
                    df_unstable_stocks['recreation_error_class'] = 'bottom ' + str(number_of_stable_unstable_stocks)
                    print(df_unstable_stocks.head(5))
                    l_unstable_stocks = np.array(df_unstable_stocks.head(number_of_stable_unstable_stocks).index)

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

                    if stable_stocks:
                        print('Plotting stable stocks')
                        list = l_stable_stocks
                        title = 'Original versus autoencoded stock price for low recreation error (stable stocks)'

                    if unstable_stocks:
                        print('Plotting unstable stocks')
                        list = l_unstable_stocks
                        title = 'Original versus autoencoded stock price for high recreation error (unstable stocks)'


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

                    # Plots for 3. Calculating stock risk for portfolio diversification
                    least_similar_stocks = False
                    most_similar_stocks = True

                    example_stock_names = df_pct_change.columns  # 'AMZN'
                    for example_stock_name in example_stock_names[0:30]:

                        example_stock_name =  'GOOGL' #'MSFT'#'AAPL'#'AMZN' #
                        top_n = 10

                        df_pct_change_corr_most_example = df_pct_change_corr[[example_stock_name]].sort_values(
                            by=[example_stock_name], ascending=False).head(top_n)
                        df_pct_change_corr_least_example = df_pct_change_corr[[example_stock_name]].sort_values(
                            by=[example_stock_name], ascending=False).tail(top_n)

                        df_similarity_most_example = df_similarity_cor[[example_stock_name]].sort_values(
                            by=[example_stock_name],
                            ascending=False).head(top_n)
                        df_similarity_least_example = df_similarity_cor[[example_stock_name]].sort_values(
                            by=[example_stock_name],
                            ascending=False).tail(top_n)



                        least_stock_cv = df_pct_change_corr_least_example.head(1).index.values[0]
                        most_stock_cv = df_pct_change_corr_most_example.iloc[[1]].index.values[0]

                        least_stock_ae = df_similarity_least_example.head(1).index.values[0]
                        most_stock_ae = df_similarity_most_example.iloc[[1]].index.values[0]




                        # Plot original series for comparison
                        plt.figure()
                        plt.rcParams["figure.figsize"] = (18, 10)
                        plt.box(False)
                        fig, ((ax1, ax2),(ax3, ax4 ), (ax5,ax6)) = plt.subplots(3, 2)
                        #fig.tight_layout(pad=10.0)
                        fig.suptitle('Baseline stock: ' + example_stock_name + ' compared to least (left) and most (right) related stocks', y=1)

                        ax1.plot(df_result_close.index, df_result_close[example_stock_name], label=example_stock_name)
                        ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False)
                        # removing all borders
                        ax1.spines['top'].set_visible(False)
                        ax1.spines['right'].set_visible(False)
                        ax1.spines['left'].set_visible(False)
                        ax1.spines['bottom'].set_visible(False)

                        ax3.plot(df_result_close.index, df_result_close[least_stock_cv], label=least_stock_cv + '(covariance)')
                        ax3.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False)
                        # removing all borders
                        ax3.spines['top'].set_visible(False)
                        ax3.spines['right'].set_visible(False)
                        ax3.spines['left'].set_visible(False)
                        ax3.spines['bottom'].set_visible(False)

                        ax5.plot(df_result_close.index, df_result_close[least_stock_ae], label=least_stock_ae + '(latent feature)')
                        ax5.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False)
                        # removing all borders
                        ax5.spines['top'].set_visible(False)
                        ax5.spines['right'].set_visible(False)
                        ax5.spines['left'].set_visible(False)
                        ax5.spines['bottom'].set_visible(False)

                        ax2.plot(df_result_close.index, df_result_close[example_stock_name], label=example_stock_name)
                        ax2.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False)
                        # removing all borders
                        ax2.spines['top'].set_visible(False)
                        ax2.spines['right'].set_visible(False)
                        ax2.spines['left'].set_visible(False)
                        ax2.spines['bottom'].set_visible(False)

                        ax4.plot(df_result_close.index, df_result_close[most_stock_cv],
                                 label=most_stock_cv + '(covariance)')
                        ax4.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False)
                        # removing all borders
                        ax4.spines['top'].set_visible(False)
                        ax4.spines['right'].set_visible(False)
                        ax4.spines['left'].set_visible(False)
                        ax4.spines['bottom'].set_visible(False)

                        ax6.plot(df_result_close.index, df_result_close[most_stock_ae],
                                 label=most_stock_ae + '(latent feature)')
                        ax6.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False)
                        # removing all borders
                        ax6.spines['top'].set_visible(False)
                        ax6.spines['right'].set_visible(False)
                        ax6.spines['left'].set_visible(False)
                        ax6.spines['bottom'].set_visible(False)

                        plt.xlabel("Dates")
                        plt.ylabel("Stock Value")
                        plt.show()



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
                        plt.show()
                   

            '''

                # -------------------------------------------------------
                #           Step3: Markowitz Model
                # -------------------------------------------------------

                print('-' * 20 + 'Step 3: Create dataset')
                df_result_close = df_result_close[df_pct_change.columns]

                print('-' * 20 + 'Step 3: Markowitz model without forecast values and without preselection')
                discrete_allocation, discrete_leftover, weights, cleaned_weights, mu, S, results = calcMarkowitzPortfolio(
                     df=df_original_close.head(len(df_original_close)-n_forecast  )
                    , budget=budget
                    , S=df_pct_change_original.cov()
                    , type='max_sharpe'
                    , target=target_annual_return)
                df_markowitz_allocation_without_forecast_without_preseletcion = pd.DataFrame(discrete_allocation.items(),
                                                            columns=['stock_name', 'bought_volume_without_forecast_without_preselection'])
                df_markowitz_allocation_without_forecast_without_preseletcion = df_markowitz_allocation_without_forecast_without_preseletcion.set_index('stock_name')

                append_to_portfolio_results(array=portfolio_results_temp,
                                            d=d,
                                            portfolio_type='markowitz_portfolio_without_forecast_without_preselection',
                                            discrete_allocation=discrete_allocation,
                                            results=results)


                print('-' * 20 + 'Step 3: Markowitz model without forecast values')
                discrete_allocation, discrete_leftover, weights, cleaned_weights, mu, S, results = calcMarkowitzPortfolio(
                     df=df_result_close.head(len(df_result_close)-n_forecast  )
                    , budget=budget
                    , S=df_pct_change_cov
                    , type='max_sharpe'
                    , target=target_annual_return)
                df_markowitz_allocation_without_forecast = pd.DataFrame(discrete_allocation.items(),
                                                            columns=['stock_name', 'bought_volume_without_forecast'])
                df_markowitz_allocation_without_forecast = df_markowitz_allocation_without_forecast.set_index('stock_name')

                append_to_portfolio_results(array=portfolio_results_temp,
                                            d=d,
                                            portfolio_type='markowitz_portfolio_without_forecast',
                                            discrete_allocation=discrete_allocation,
                                            results=results)


                print('-' * 20 + 'Step 3: Markowitz model with forecast')
                discrete_allocation, discrete_leftover, weights, cleaned_weights, mu, S, results = calcMarkowitzPortfolio(df=df_result_close
                                                                                                          ,budget=budget
                                                                                                          ,S = df_pct_change_cov
                                                                                                          ,type = 'max_sharpe'
                                                                                                          ,target = target_annual_return)
                df_markowitz_allocation_with_forecast = pd.DataFrame(discrete_allocation.items(),  columns=['stock_name','bought_volume_with_forecast'])
                df_markowitz_allocation_with_forecast = df_markowitz_allocation_with_forecast.set_index('stock_name')

                append_to_portfolio_results(array = portfolio_results_temp,
                                             d =d,
                                             portfolio_type= 'markowitz_portfolio_with_forecast',
                                             discrete_allocation=discrete_allocation,
                                             results=results)


                #cla = CLA(expected_returns=mu, cov_matrix=S, weight_bounds=(0, 1))
                #Plotting.plot_efficient_frontier(cla, points=100, show_assets=True)

                print('-' * 20 + 'Step 3: Markowitz model with cleaned covariance matrix')
                discrete_allocation, discrete_leftover, weights, cleaned_weights,mu, S, results = calcMarkowitzPortfolio(df=df_result_close
                                                                                                          ,budget=budget
                                                                                                          ,S = df_var_autoencoder_reconstruct_real_cov
                                                                                                          ,type = 'max_sharpe'
                                                                                                          ,target = target_annual_return
                                                                                                          ,cov_type = 'adjusted')
                df_markowitz_allocation_var_autoencoder = pd.DataFrame(discrete_allocation.items(),  columns=['stock_name','bought_volume_with_forecast_cleaned'])
                df_markowitz_allocation_var_autoencoder = df_markowitz_allocation_var_autoencoder.set_index('stock_name')

                append_to_portfolio_results(array=portfolio_results_temp,
                                            d=d,
                                            portfolio_type='markowitz_portfolio_with_forecast_and adjusted covariance_matrix',
                                            discrete_allocation=discrete_allocation,
                                            results=results)


                '''
                print('-' * 20 + 'Step 3: Markowitz model with latent features')
                discrete_allocation, discrete_leftover, weights, cleaned_weights,mu, S, results = calcMarkowitzPortfolio(df=df_result_close
                                                                                                          ,budget=budget
                                                                                                          ,S = df_similarity_cov
                                                                                                          ,type = 'max_sharpe'
                                                                                                          ,target = target_annual_return
                                                                                                          ,cov_type = 'adjusted-disabled')
                df_markowitz_allocation_latent_feature = pd.DataFrame(discrete_allocation.items(),  columns=['stock_name','bought_volume_with_forecast_latent'])
                df_markowitz_allocation_latent_feature = df_markowitz_allocation_latent_feature.set_index('stock_name')

                append_to_portfolio_results(array=portfolio_results_temp,
                                            d=d,
                                            portfolio_type='markowitz_portfolio_with_forecast_and_latent_features',
                                            discrete_allocation=discrete_allocation,
                                            results=results)



                #cla = CLA(expected_returns=mu, cov_matrix=S, weight_bounds=(0, 1))
                #Plotting.plot_efficient_frontier(cla, points=100, show_assets=True)
                '''
                df_markowitz_allocation = df_markowitz_allocation_without_forecast_without_preseletcion.join(df_markowitz_allocation_with_forecast
                                                                            , how='outer'
                                                                            , lsuffix =''
                                                                            , rsuffix='')
                df_markowitz_allocation = df_markowitz_allocation.join(df_markowitz_allocation_without_forecast
                                                                            , how='outer'
                                                                            , lsuffix =''
                                                                            , rsuffix='')
                df_markowitz_allocation = df_markowitz_allocation.join(df_markowitz_allocation_var_autoencoder
                                                                            , how='outer'
                                                                            , lsuffix =''
                                                                            , rsuffix='')

                '''
                df_markowitz_allocation = df_markowitz_allocation.join(df_markowitz_allocation_latent_feature
                                                       ,how='outer'
                                                       ,lsuffix=''
                                                       ,rsuffix='')
                '''
                df_result_close_buy_price = df_original_close.tail(n_forecast).head(1).transpose()
                df_result_close_buy_price = df_result_close_buy_price.rename(columns={df_result_close_buy_price.columns[0]: "buy_price"})

                df_result_close_predicted_price = df_original_close.tail(1).transpose()
                df_result_close_predicted_price = df_result_close_predicted_price.rename(columns={df_result_close_predicted_price.columns[0]: "predicted_price"})

                df_result_close_sell_price = df_original_close.head(len(df_original_close) - n_forecast * (d-1))
                df_result_close_sell_price = df_result_close_sell_price.tail(1).transpose()
                df_result_close_sell_price = df_result_close_sell_price.rename(columns={df_result_close_sell_price.columns[0]: "sell_price"})


                df_markowitz_allocation = df_markowitz_allocation.join(df_result_close_buy_price, how ='left')
                df_markowitz_allocation = df_markowitz_allocation.join(df_result_close_predicted_price, how='left')
                df_markowitz_allocation = df_markowitz_allocation.join(df_result_close_sell_price, how='left')
                df_markowitz_allocation['backtest_id'] = d

                df_markowitz_allocation['delta'] = df_markowitz_allocation['sell_price'] - df_markowitz_allocation['buy_price']

                df_markowitz_allocation['profit_without_forecast_without_preselection'] = df_markowitz_allocation['delta'] * \
                                                                     df_markowitz_allocation[
                                                                         'bought_volume_without_forecast_without_preselection']
                df_markowitz_allocation['profit_without_forecast'] = df_markowitz_allocation['delta'] * df_markowitz_allocation['bought_volume_without_forecast']
                df_markowitz_allocation['profit_with_forecast'] = df_markowitz_allocation['delta'] * df_markowitz_allocation[
                    'bought_volume_with_forecast']
                df_markowitz_allocation['profit_with_forecast_cleaned'] = df_markowitz_allocation['delta'] * df_markowitz_allocation[
                    'bought_volume_with_forecast_cleaned']
                #df_markowitz_allocation['profit_with_forecast_latent'] = df_markowitz_allocation['delta'] * df_markowitz_allocation[
                #    'bought_volume_with_forecast_latent']

                df_results_markowitz_allocation = df_results_markowitz_allocation.append(df_markowitz_allocation)

                df_results_portfolio_temp = pd.DataFrame.from_dict(portfolio_results_temp)

                df_results_portfolio_temp['profit'] = [np.sum(df_markowitz_allocation['profit_without_forecast_without_preselection']),
                                                       np.sum(df_markowitz_allocation['profit_without_forecast'])
                                                       , np.sum(df_markowitz_allocation['profit_with_forecast'])
                                                       ,np.sum(df_markowitz_allocation['profit_with_forecast_cleaned'])
                                                       #,np.sum(df_markowitz_allocation['profit_with_forecast_latent'])
                                                     ]






                df_results_portfolio = df_results_portfolio.append(df_results_portfolio_temp)
                # End of for loop

            # Plot
            groups = df_results_portfolio.groupby('portfolio_type')

            fig, ax = plt.subplots(figsize=(10, 6))
            plt.box(False)
            for name, group in groups:
                ax.plot(group.backtest_iteration, group.profit, ms=12,
                        label=str(name), alpha=0.5)

            plt.title('Profits per backtest')
            plt.xlabel("Backtest iteration")
            plt.ylabel("Profits generated in the last {v_n_forecast} days".format(v_n_forecast=n_forecast))
            ax.legend(loc='best',  frameon=False)
            plt.savefig('img/backtest_profits.png')
            plt.show()










'''
                print('-' * 20 + 'Step 4: Create Portfolio ')
                print(
                    '-' * 25 + 'Join the datasets from the similarity score and the reconstruction error with some metadata')
                df_metadata = pd.read_csv('data/historical_stocks.csv', sep=',')
                df_metadata = df_metadata.rename(columns={'ticker': 'stock_name'})
                df_metadata = df_metadata.set_index('stock_name')
    
                df_scaled_mse = pd.read_csv('data/df_scaled_mse_' + str(d)+'.csv', sep=';')
                df_scaled_mse = df_scaled_mse.set_index('ticker')
    
                df_portfolio = df_returns\
                                .join(df_recreation_error[['recreation_error']], how='left')\
                                .join(df_similarity[['similarity_score']], how='left') \
                                .join(df_scaled_mse[['scaled mse']], how='left')\
                                .join(df_metadata, how='left')
    
    
                # remove very high values
                df_portfolio = df_portfolio[df_portfolio['recreation_error'] < df_portfolio['recreation_error'].quantile(0.99)]
                df_portfolio = df_portfolio[
                    df_portfolio['similarity_score'] < df_portfolio['similarity_score'].quantile(0.99)]
                df_portfolio = df_portfolio[
                    df_portfolio[avg_return_column] < df_portfolio[avg_return_column].quantile(0.95)]
    
    
                # calculate bins
                df_portfolio['similarity_score_quartile'] = pd.qcut(df_portfolio.similarity_score, n_bins, precision=0, labels=False)
    
                # calculate return*recreation error
                df_scaler_recreation_error = preprocessing.MinMaxScaler()
                df_portfolio['recreation_error_scaled_inverse'] = 1 - df_scaler_recreation_error.fit_transform(
                    df_portfolio[['recreation_error']].values)
                df_portfolio['avg_return*recreation_error'] = df_portfolio[avg_return_column] * df_portfolio[
                    'recreation_error_scaled_inverse']
    
                #calculate inverse scaled scaled mse
                df_scaler_sclaed_mse = preprocessing.MinMaxScaler()
                df_portfolio['scaled_mse_scaled_inverse'] = 1 - df_scaler_sclaed_mse.fit_transform(
                    df_portfolio[['scaled mse']].values)
                df_portfolio['recreation_error*scaled_mse_scaled_inverse'] = df_portfolio['scaled_mse_scaled_inverse'] * df_portfolio[
                    'recreation_error_scaled_inverse']
    
    
    
                if plot_results:
                    # plot the results
                    print('-' * 25 + 'Plot the results')
                    df_plot = df_portfolio[df_portfolio[avg_return_column] > 0]
                    df_plot = df_plot[df_plot['similarity_score'] < df_plot['similarity_score'].quantile(0.99)]
                    groups = df_plot.groupby('sector')
    
                    # Plot
                    fig, ax = plt.subplots(figsize=(10, 6))
                    plt.box(False)
                    for name, group in groups:
                        ax.plot(group.similarity_score, group[avg_return_column] * 100, marker='o', linestyle='', ms=12,
                                label=name, alpha=0.5)
    
                    plt.title('Average retuns vs. similarity metric (latent feature values)')
                    plt.xlabel("Similarity Score")
                    plt.ylabel(avg_return_column)
                    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False)
                    plt.show()
    
                    print('-' * 25 + 'Plot results')
                    df_plot = df_portfolio[df_portfolio[avg_return_column] > 0]
                    df_plot = df_plot[df_plot['recreation_error'] < df_plot['recreation_error'].quantile(0.99)]
                    groups = df_plot.groupby('sector')
    
                    # Plot
                    fig, ax = plt.subplots(figsize=(10, 6))
                    plt.box(False)
                    for name, group in groups:
                        ax.plot(group.recreation_error, group[avg_return_column] * 100, marker='o', linestyle='', ms=12, label=name, alpha=0.5)
    
                    plt.title('Average return vs. recreation error')
                    plt.xlabel("Recreation error")
                    plt.ylabel("average returns last 10 days in %")
                    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False)
                    plt.show()
    
                    df_plot = df_portfolio[df_portfolio[avg_return_column] > 0]
    
                    df_plot = df_plot[df_plot[avg_return_column] < df_plot[avg_return_column].quantile(0.9)]
                    df_plot = df_plot[df_plot['recreation_error'] < df_plot['recreation_error'].quantile(0.9)]
                    groups = df_plot.groupby('similarity_score_quartile')
    
                    # Plot
                    fig, ax = plt.subplots(figsize=(10, 6))
                    plt.box(False)
                    for name, group in groups:
                        ax.plot(group.recreation_error, group[avg_return_column] * 100, marker='o', linestyle='', ms=12, label= 'Group '+ str(name), alpha=0.5)
    
                    plt.title('Average return vs. recreation error by similarity quartiles (colors)')
                    plt.xlabel("Recreation error")
                    plt.ylabel("Average returns last 10 days in %")
                    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False)
                    plt.show()
    
    
    
                # merge the results
                df_portfolio_selected_stocks_option_0 = df_markowitz_allocation.join(df_portfolio, how='left')
                df_portfolio_selected_stocks_option_0 = df_portfolio_selected_stocks_option_0[df_portfolio_selected_stocks_option_0['bought_volume'] != 0]
                df_portfolio_selected_stocks_option_0['pnl'] = df_portfolio_selected_stocks_option_0['delta'] * df_portfolio_selected_stocks_option_0['bought_volume']
                markowitz_profit = [df_portfolio_selected_stocks_option_0['pnl'].sum()]
    
                profits_option_1, df_portfolio_selected_stocks_option_1 = portfolio_selection(d=d
                                                                                              , df_portfolio=df_portfolio
                                                                                              , ranking_colum='recreation_error'
                                                                                              , group_by=True
                                                                                              , n_stocks_per_bin=n_stocks_per_bin
                                                                                              , n_bins=n_bins
                                                                                              , budget=budget)
    
                profits_option_2, df_portfolio_selected_stocks_option_2 = portfolio_selection(d=d
                                                                                              , df_portfolio=df_portfolio
                                                                                              , ranking_colum='recreation_error'
                                                                                              , group_by=False
                                                                                              , n_stocks_per_bin=n_stocks_per_bin
                                                                                              , n_bins=n_bins
                                                                                              , budget=budget)
    
                profits_option_3, df_portfolio_selected_stocks_option_3 = portfolio_selection(  d=d
                                                                                              , df_portfolio=df_portfolio
                                                                                              , ranking_colum='recreation_error*scaled_mse_scaled_inverse'
                                                                                              , group_by=True
                                                                                              , n_stocks_per_bin=n_stocks_per_bin
                                                                                              , n_bins=n_bins
                                                                                              , budget=budget)
    
                profits_option_4, df_portfolio_selected_stocks_option_4 = portfolio_selection(d=d
                                                                                              , df_portfolio=df_portfolio
                                                                                              , group_by=False
                                                                                              , ranking_colum='recreation_error*scaled_mse_scaled_inverse'
                                                                                              , n_stocks_per_bin=n_stocks_per_bin
                                                                                              , n_bins=n_bins
                                                                                              , budget=budget)
    
                print('-' * 25 + 'Merging the portfolio optimization results and compare them')
                df_portfolio_selected_stocks_option_0['options'] = 'Markowitz'
                df_portfolio_selected_stocks_option_1['options'] = 'recreation error with grouping'
                df_portfolio_selected_stocks_option_2['options'] = 'recreation error without grouping'
                df_portfolio_selected_stocks_option_3[
                    'options'] = 'Scaled MSE * recreation error with grouping'
                df_portfolio_selected_stocks_option_4[
                    'options'] = 'Scaled MSE * recreation error without grouping'
    
    
                df_portfolio_selected_stocks_option_0['total_profit'] = int(markowitz_profit[0])
                df_portfolio_selected_stocks_option_1['total_profit'] = int(profits_option_1[0])
                df_portfolio_selected_stocks_option_2['total_profit'] = int(profits_option_2[0])
                df_portfolio_selected_stocks_option_3['total_profit'] = int(profits_option_3[0])
                df_portfolio_selected_stocks_option_4['total_profit'] = int(profits_option_4[0])
    
                df_portfolio_selection_results = df_portfolio_selected_stocks_option_0.append(
                    df_portfolio_selected_stocks_option_1) \
                    .append(df_portfolio_selected_stocks_option_2) \
                    .append(df_portfolio_selected_stocks_option_3) \
                    .append(df_portfolio_selected_stocks_option_4)
    
            if 'df_portfolio_selection_results_final' not in locals():
                df_portfolio_selection_results_final = df_portfolio_selection_results
            else:
                df_portfolio_selection_results_final = df_portfolio_selection_results_final.append(
                    df_portfolio_selection_results)
    
        print(df_portfolio_selection_results_final.to_string())
    
        if plot_results:
            print('-' * 10 + 'Plot results')
            df_plot = df_portfolio_selection_results_final.groupby(['backtest_iteration', 'options'], as_index=False)[
                'total_profit'].max()
            groups = df_plot.groupby('options')
    
            # Plot
            fig, ax = plt.subplots(figsize=(12, 6))
            i = 0
            for name, group in groups:
                print(name, i)
    
                if i == 0:
                    print(name)
                    ax.plot(group.backtest_iteration, group['total_profit'], marker='o', linestyle='', ms=12, label=name)
                if i == 1:
                    print(name)
                    ax.plot(group.backtest_iteration, group['total_profit'], marker='s', linestyle='', ms=12, label=name)
                if i == 2:
                    print(name)
                    ax.plot(group.backtest_iteration, group['total_profit'], marker='x', linestyle='', ms=12, label=name)
                if i == 3:
                    print(name)
                    ax.plot(group.backtest_iteration, group['total_profit'], marker='v', linestyle='', ms=12, label=name)
                i = i + 1
    
            plt.title('Backtest Profits')
            plt.xlabel("Back Test Iteration")
            plt.ylabel("Total Profit in {} days".format(n_forecast))
            # Shrink current axis's height by 10% on the bottom
            box = ax.get_position()
            ax.set_position([box.x0, box.y0 + box.height * 0.3,
                             box.width, box.height * 0.8])
    
            # Put a legend below current axis
            ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
                      fancybox=True, shadow=True, ncol=1)
    
            plt.savefig('img/backtest_results.png')
            plt.show()
    
    
    '''