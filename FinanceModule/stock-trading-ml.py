
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

    verbose = 0


    timeseries_transformation = False
    stock_selection = False
    timeseries_evaluation = False
    timeseries_forecasting = True
    portfolio_optimization = False
    portfolio_analysis = False

    while timeseries_forecasting == timeseries_evaluation and timeseries_evaluation == True:
        print('''
                Note Multiprocessing cannot run when in the main session a keras backend was created before creating the worker pool. 
                 E.g. Time Series Evaluation cannot run in the same script as time series forecasting. 
                ''')
        break

    # 0 Data Preparation
    history_points = 150
    test_split = 0.9
    n_forecast = 5
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
    backtest_days = 100
    forecast_all_columns = False # only forecast close value

    # 3. Portfolio Optimization parameters
    budget = 200000
    hidden_layers_latent = 20
    target_annual_return = 0.50
    run_id = 'etf_20200217'

    try:
        os.mkdir('data/'  +run_id)
        os.mkdir('data/' + run_id + '/intermediary')
        os.mkdir('img/' + run_id )
    except OSError:
        print("Creation of the directory %s failed" % run_id)
    else:
        print("Successfully created the directory %s " % run_id)

    if timeseries_transformation:
        transformDataset( input_path='/home/workstation/synology_docker/python_stock_data/stock_data/2021-02-11-07-39-31_historic_data.csv'
                          , input_sep=';'
                          ,input_symbol='symbol'
                         , metadata_input_path = 'data/historical_stocks.csv', metadata_sep = ','
                         ,output_path='data/{v_run_id}/historical_stock_prices_original.csv'.format(v_run_id=run_id), output_sep=';'
                        # ,filter_sectors = sectors
                         ,n_tickers = n_tickers, n_last_values = stock_selection_number )
    

    print('-' * 5 + 'Loading the dataset from disk')
    df_original = pd.read_csv('data/{v_run_id}/historical_stock_prices_original.csv'.format(v_run_id = run_id), sep=';', index_col='date')

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
        df_result_close_filtered.to_csv('data/{v_run_id}/df_result_close_filtered.csv'.format(v_run_id = run_id), sep=';')

    df_result_close_filtered = pd.read_csv('data/{v_run_id}/df_result_close_filtered.csv'.format(v_run_id = run_id), sep=';', index_col ='date')


    # Get tickers as a list
    print('-' * 5 + 'Getting list of unique tickers')
    l_tickers_new = df_result_close_filtered.columns.str.split('_')
    l_tickers_unique = np.unique(fun_column(l_tickers_new, 0))

    #l_tickers_unique = ['GOOGL',"SSRM","CPE"]
    l_tickers_unique_chunks = list(chunks(l_tickers_unique, parallel_processes))


    # 2. Forecasting using recurrent neural networks
    if timeseries_forecasting:
        #for d in range(5, 8)[::-1]:
        for d in range(int(backtest_days/n_forecast)+1)[::-1]:

            if d != 0 and d > 4:
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
                        if not os.path.isfile('data/{v_run_id}/intermediary/df_result_{v_d}_{v_column}.csv'.format(v_run_id = run_id, v_d = str(d),v_column = column )):
                            for batch_size in [40]:
                                print(batch_size)
                                for epochs in [400]:
                                    pool.apply_async(stock_forceasting,
                                                     args=(i,
                                                           column,
                                                           run_id,
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
                                                           epochs,
                                                           forecast_all_columns
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
                            df_result_ticker = pd.read_csv('data/{v_run_id}/intermediary/df_result_{v_d}_{v_column}.csv'.format(v_run_id = run_id, v_d = str(d),v_column = column ), sep=';',
                                                           index_col='Unnamed: 0')
                        if timeseries_evaluation:
                            df_scaled_mse_ticker = pd.read_csv('data/{v_run_id}/intermediary/df_scaled_mse_{v_d}_{v_column}.csv'.format(v_run_id = run_id, v_d = str(d),v_column = column ),
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
                    df_result.to_csv('data/{v_run_id}/df_result_{v_d}.csv'.format(v_run_id = run_id, v_d = str(d)), sep=';')
                if timeseries_evaluation:
                    df_scaled_mse.to_csv('data/{v_run_id}/df_scaled_mse_{v_d}.csv'.format(v_run_id = run_id, v_d = str(d)), sep=';')

    # End of for loops
    # 3. Calculating stock risk for portfolio diversification
    # 4.Portfolio optimization using linear programming



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
                                           plot_results,run_id)
                                     )
                except:
                    print('file does not exists')

        print('closing new pool')
        pool.close()
        pool.join()

    for d in backtest_range:
        if d != 0:
            print(d)
            if d == len(backtest_range)-1:
                df_temp = pd.read_csv('data/{v_run_id}/df_backtest_{v_d}_portfolio.csv'.format(v_d=str(d), v_run_id = run_id), sep=';')
            else:
                df_temp2 = pd.read_csv('data/{v_run_id}/df_backtest_{v_d}_portfolio.csv'.format(v_d=str(d), v_run_id = run_id), sep=';')
                df_temp = df_temp.append(df_temp2)

        df_results_portfolio = df_temp.reset_index()
        df_results_portfolio.to_csv('data/{v_run_id}/df_backtest_portfolio.csv'.format( v_run_id = run_id), sep=';')

    '''
    df_results_portfolio.to_csv('data/df_backtest_portfolio.csv', sep=';',
                                columns=df_results_portfolio_temp.columns)
    '''



if portfolio_analysis:

    # load full dataset
    df_original = pd.read_csv('data/{v_run_id}/historical_stock_prices_original.csv'.format( v_run_id = run_id), sep=';')
    df_original.index = pd.to_datetime(df_original.date)
    df_original_close_full = df_original.filter(like='Close', axis=1)

    new_columns = []
    [new_columns.append(c.split('_')[0]) for c in df_original_close_full.columns]
    df_original_close_full.columns = new_columns
    df_original_close = df_original_close_full.iloc[:, :stock_selection_number]

    # load portfolio results
    df_results_portfolio_used = pd.read_csv('data/{v_run_id}/df_backtest_portfolio.csv'.format( v_run_id = run_id), sep=';')
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
    plot_backtest_results(df_results_portfolio_used_with_volatility, column='profit',  colors =  colors, title = 'Out-of sample - profit', run_id=run_id)
    plot_backtest_results(df_results_portfolio_used_with_volatility, column='cumsum_profit',  colors =  colors, title = 'Out-of sample - cummulative profit', run_id=run_id)
    plot_backtest_results(df_results_portfolio_used_with_volatility, column='portfolio_volatility_10',  colors =  colors, title = 'Out-of sample - volatility', run_id=run_id)
    plot_backtest_results(df_results_portfolio_used_with_volatility, column='10_expected_return',  colors =  colors, title = 'Out-of sample - returns', run_id=run_id)
    plot_backtest_results(df_results_portfolio_used_with_volatility, column='sharpe_ratio_10',  colors =  colors, title = 'Out-of sample - sharpe ratio', run_id=run_id )

    plot_backtest_results(df_results_portfolio_used_with_volatility, column='expected_annual_return',  colors =  colors, title = 'In sample - expected annual return', run_id=run_id)
    plot_backtest_results(df_results_portfolio_used_with_volatility, column='sharpe_ratio',  colors =  colors , title = 'In sample - sharpe ratio', run_id=run_id)
    plot_backtest_results(df_results_portfolio_used_with_volatility, column='annual_volatility',  colors =  colors, title = 'In sample - annual volatility', run_id=run_id)

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
    plt.savefig('img/{v_run_id}/selected_stocks_heatmap_{v_type}.png'.format(v_type =type, v_run_id=run_id), dpi=300)

    df = pd.DataFrame({'lab': ['A', 'B', 'C'], 'val': [10, 30, 20]})
    df_stock_allocation['index'] = df_stock_allocation.index
    ax = df_stock_allocation.plot.barh(x='index', y=df_stock_allocation.columns[0])
    fig = ax.get_figure()
    fig.savefig('img/{v_run_id}/discrete allocation.png'.format(v_type =type, v_run_id=run_id), dpi=300)





