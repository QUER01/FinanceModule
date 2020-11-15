
if __name__ == "__main__":  # confirms that the code is under main function
    import matplotlib.pyplot as plt
    from keras.models import Model
    import numpy as np
    import json
    from keras.callbacks import TensorBoard
    from FinanceModule.util import *
    import copy
    import pandas as pd
    from sklearn import preprocessing
    from datetime import datetime
    from multiprocessing import Pool
    import multiprocessing
    from pandas.plotting import register_matplotlib_converters
    register_matplotlib_converters()

    warnings.filterwarnings("error")
    os.environ["PATH"] += os.pathsep + 'lib/Graphviz2.38/bin/'
    print('-' * 50)
    print('PART I: Timeseries Cleaning')
    print('-' * 50)

    # general parameters
    fontsize = 12
    parallel_processes = multiprocessing.cpu_count() - 1

    # indicate folder to save, plus other options
    date = datetime.now().strftime('%Y-%m-%d_%H_%M')
    tensorboard = TensorBoard(log_dir='./logs/run_' + date
                              ,histogram_freq=0
                              ,write_graph=True
                              ,write_images=False
                              ,embeddings_freq=0
                              ,embeddings_metadata=None)
    # save it in your callback list, where you can include other callbacks
    callbacks_list = [tensorboard]

    # script parameters
    test_setting = False
    plot_results = True
    stock_selection = False
    verbose = 0

    '''
     Note Multiprocessing cannot run when in the main session a keras backend was created before creating the worker pool. 
     E.g. Time Series Evaluation cannot run in the same script as time series forecasting. 
    '''
    timeseries_evaluation = False
    timeseries_forecasting =False
    portfolio_optimization = True
    portfolio_analyis = True


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
    stock_selection_number = 1000

    # 2. Forecasting using recurrent neural networks
    backtest_days = 200

    # 3. Portfolio Optimization parameters
    budget = 100000
    hidden_layers_latent = 20
    target_annual_return = 0.50

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

    l_tickers_unique = ['GOOGL',"SSRM","CPE"]
    l_tickers_unique_chunks = list(chunks(l_tickers_unique, parallel_processes))


    # 2. Forecasting using recurrent neural networks
    if timeseries_forecasting:
        #for d in range(5, 8)[::-1]:
        for d in range(int(backtest_days/n_forecast)+1)[::-1]:

            if d != 0 :
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
                        if not os.path.isfile('data/intermediary/df_result_' + str(d) + '_' + column + '.csv'):
                            for batch_size in [20,25,30,35,40,45,50,55,60,65]:
                                print(batch_size)
                                for epochs in [30,  50, 100, 150, 200,400,500,600]:
                                    pool.apply_async(stock_forceasting,
                                                     args=(i,
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
                                                           epochs
                                                           ))
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



    def portfolio_selection(d, stock_selection_number, n_forecast, callbacks_list, epochs, verbose, batch_size, plot_results):
        import numpy as np
        import matplotlib.pyplot as plt

        from FinanceModule.util import defineVariationalAutoencoder \
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
        df_original_close = df_original_close_full.iloc[:, :stock_selection_number]

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
        df_results_portfolio.to_csv('data/df_backtest_{v_d}portfolio.csv'.format(v_d=d), sep=';',
                                    columns=df_results_portfolio_temp.columns)


if portfolio_optimization:
    portfolio_results = []
    markowitz_allocation = []
    df_results_markowitz_allocation = pd.DataFrame()
    df_results_portfolio = pd.DataFrame()
    new_columns = []

    backtest_range = range(int(backtest_days / n_forecast) + 1)[::-1]
    l_backtest_range_chunks = list(chunks(backtest_range, parallel_processes))

    for l_backtest_range_chunk in l_backtest_range_chunks:
        print(l_backtest_range_chunk)
        pool = Pool(processes=parallel_processes)  # start 12 worker processes

        for d in l_backtest_range_chunk:
            if d != 0:
                print('-' * 5 + 'Backtest Iteration ' + str(d))
                try:
                    pool.apply_async(portfolio_selection,
                                     args=(d, stock_selection_number, n_forecast, callbacks_list, epochs, verbose, batch_size,
                                           plot_results)
                                     )
                except:
                    print('file does not exists')

        print('closing new pool')
        pool.close()
        pool.join()

  for d in backtest_range:
        print(d)
        if d == 20:
            df_temp = pd.read_csv('data/df_backtest_{v_d}portfolio.csv'.format(v_d=d), sep=';')
        else:
            df_temp2 = pd.read_csv('data/df_backtest_{v_d}portfolio.csv'.format(v_d=d), sep=';')
            df_temp = df_temp.append(df_temp2)

    df_results_portfolio = df_temp.reset_index()
    df_results_portfolio.to_csv('data/df_backtest_portfolio.csv', sep=';')

    '''
    df_results_portfolio.to_csv('data/df_backtest_portfolio.csv', sep=';',
                                columns=df_results_portfolio_temp.columns)
    '''



if portfolio_analyis:

    # load full dataset
    df_original = pd.read_csv('data/historical_stock_prices_original.csv', sep=';')
    df_original.index = pd.to_datetime(df_original.date)
    df_original_close_full = df_original.filter(like='Close', axis=1)

    new_columns = []
    [new_columns.append(c.split('_')[0]) for c in df_original_close_full.columns]
    df_original_close_full.columns = new_columns
    df_original_close = df_original_close_full.iloc[:, :stock_selection_number]

    # load portfolio results
    df_results_portfolio_used = pd.read_csv('data/df_backtest_portfolio.csv', sep=';')
    #df_results_portfolio_used = df_results_portfolio_used[df_results_portfolio_temp.columns]

    df_results_portfolio_used = df_results_portfolio_used[df_results_portfolio_used['portfolio_type'] != 'markowitz_portfolio_without_forecast_without_preselection' ]
    df_results_portfolio_used = df_results_portfolio_used[df_results_portfolio_used['portfolio_type'] != 'markowitz_portfolio_with_forecast_and adjusted covariance_matrix']

    # renaming model types
    df_results_portfolio_used.loc[df_results_portfolio_used['portfolio_type'] == 'markowitz_portfolio_baseline_full', 'portfolio_type'] = "markowitz_portfolio_full_dataset"
    df_results_portfolio_used.loc[df_results_portfolio_used['portfolio_type'] == 'markowitz_portfolio_baseline', 'portfolio_type'] = "markowitz_portfolio_filtered_dataset"

    df_results_portfolio_used = df_results_portfolio_used[df_results_portfolio_used['portfolio_type'].isin(['markowitz_portfolio_full_dataset'
                                                                                                            ,'markowitz_portfolio_filtered_dataset'
                                                                                                           # ,'markowitz_portfolio_with_forecast'
                                                                                                           #,'markowitz_portfolio_with_forecast_and_latent_features'
                                                                                                             ])]


    df_results_portfolio_used = df_results_portfolio_used.reset_index()

    volatility_10 = []
    volatility_252 = []
    for backtest in df_results_portfolio_used['backtest_iteration'].unique():
        print('Calculating portfolio annual volatility for backtest {v_backtest}'.format(v_backtest =str(backtest)))
        df_original_backtest = df_original_close_full.head(len(df_original_close_full) - n_forecast * (backtest - 1))
        for portfolio_type in df_results_portfolio_used['portfolio_type'].unique():
            selected_stocks = \
            df_results_portfolio_used[(df_results_portfolio_used['backtest_iteration'] == backtest)
                                      & (df_results_portfolio_used['portfolio_type'] == portfolio_type)][
                'discrete_allocation'].values

            selected_stocks_json = json.loads(str(selected_stocks[0]).replace("\'", "\""))
            stocks = []
            weights = []
            for stock in selected_stocks_json:
                stocks.append(stock)
                weights.append(selected_stocks_json[stock])


            weights = weights / np.sum(weights)
            try:
                data = df_original_backtest[stocks]
                #print(data.shape)
            except:
                print("Some of these stocks are not in the dataset: {b}_{s}".format(b=str(backtest), s=str(stocks)))

            log_returns = data.pct_change()
            portfolio_vol_252 = np.sqrt(np.dot(weights.T, np.dot(log_returns.tail(252 ).cov() * 252, weights)))
            portfolio_vol_10 = np.sqrt(np.dot(weights.T, np.dot(log_returns.tail(10 ).cov() * 10, weights)))
            print(portfolio_vol_10)
            volatility_252.append(portfolio_vol_252)
            volatility_10.append(portfolio_vol_10)

    df_volatility = pd.DataFrame(volatility_252, columns=['portfolio_volatility_252'])
    df_volatility['portfolio_volatility_10'] = volatility_10
    df_results_portfolio_used_with_volatility = df_results_portfolio_used.join(df_volatility, how='left')
    df_results_portfolio_used_with_volatility['cumsum_profit'] = df_results_portfolio_used_with_volatility.groupby('portfolio_type')['profit'].cumsum()
    df_results_portfolio_used_with_volatility['10_expected_return'] = df_results_portfolio_used_with_volatility['profit']/budget
    df_results_portfolio_used_with_volatility['sharpe_ratio_10'] = df_results_portfolio_used_with_volatility['10_expected_return']\
                                                                   /df_results_portfolio_used_with_volatility['portfolio_volatility_10']

    # Plot
    colors = ['#2195ca' ,'#c1c1c1','#f9c77d','#a3660b']# blue, grey, yellow
    plot_backtest_results(df_results_portfolio_used_with_volatility, column='profit',  colors =  colors, title = 'Out-of sample - profit')
    plot_backtest_results(df_results_portfolio_used_with_volatility, column='cumsum_profit',  colors =  colors, title = 'Out-of sample - cummulative profit')
    plot_backtest_results(df_results_portfolio_used_with_volatility, column='portfolio_volatility_10',  colors =  colors, title = 'Out-of sample - volatility ')
    plot_backtest_results(df_results_portfolio_used_with_volatility, column='10_expected_return',  colors =  colors, title = 'Out-of sample - returns ')
    plot_backtest_results(df_results_portfolio_used_with_volatility, column='sharpe_ratio_10',  colors =  colors, title = 'Out-of sample - sharpe ratio' )

    plot_backtest_results(df_results_portfolio_used_with_volatility, column='expected_annual_return',  colors =  colors, title = 'In sample - expected annual return')
    plot_backtest_results(df_results_portfolio_used_with_volatility, column='sharpe_ratio',  colors =  colors , title = 'In sample - sharpe ratio')
    plot_backtest_results(df_results_portfolio_used_with_volatility, column='annual_volatility',  colors =  colors, title = 'In sample - annual volatility')

    plt.close('all')

    df = df_results_portfolio_used_with_volatility
    df_stock_allocation = pd.read_json(json.dumps(json.loads(df['discrete_allocation'].iloc[0,].replace("\'", "\""))), orient='index')
    df_stock_allocation.columns = ["{d}_{type}".format(d=df_results_portfolio_used_with_volatility['backtest_iteration'].iloc[0], type=df_results_portfolio_used_with_volatility['portfolio_type'].iloc[0])]

    for index, df_stocks in df[1:len(df)].iterrows():
        stocks = df_stocks['discrete_allocation']
        backtest = str(df_stocks['backtest_iteration'])
        type = str(df_stocks['portfolio_type'])
        df_temp = pd.read_json(json.dumps(json.loads(stocks.replace("\'", "\""))), orient='index')
        df_temp.columns = ["{d}_{type}".format(d = backtest, type =type) ]
        df_stock_allocation = df_stock_allocation.join(df_temp, how='outer')


    #df_stock_allocation = df_stock_allocation.fillna(0)
    #df_stock_allocation = df_stock_allocation.drop('ADXS')
    #df_stock_allocation = df_stock_allocation.drop('ICON')

    stock = df_stock_allocation.index
    backtest = df_stock_allocation.columns
    value = np.array(df_stock_allocation.transpose())

    plt.figure()
    plt.rcParams["figure.figsize"] = (40, 20)
    fig, ax = plt.subplots()
    im = ax.imshow(value)

    # We want to show all ticks...
    ax.set_xticks(np.arange(len(stock)))
    ax.set_yticks(np.arange(len(backtest)))
    # ... and label them with the respective list entries
    ax.set_xticklabels(stock)
    ax.set_yticklabels(backtest)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")
    cbar = fig.colorbar(im)


    ax.set_title("Selected Stocks in each backtest iteration")
    plt.savefig('img/selected_stocks_heatmap_{}.png'.format(type), dpi=300)

