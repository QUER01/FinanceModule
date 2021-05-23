
if __name__ == "__main__":  # confirms that the code is under main function
    import matplotlib.pyplot as plt
    import os
    import numpy as np
    import json
    from keras.callbacks import TensorBoard
    from src.util import *
    import copy
    import pandas as pd
    from sklearn import preprocessing
    from datetime import datetime
    from multiprocessing import Pool
    import multiprocessing
    from pandas.plotting import register_matplotlib_converters

    register_matplotlib_converters()
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
    timeseries_forecasting = False
    portfolio_optimization = False
    portfolio_analysis = True

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
    stock_selection_number = 200

    # 2. Forecasting using recurrent neural networks
    backtest_days = 100
    forecast_all_columns = False # only forecast close value

    # 3. Portfolio Optimization parameters
    budget = 200000
    hidden_layers_latent = 20
    target_annual_return = 0.50
    run_ids = ['stock_20101101','etf_20200217','etf_20210514']
    run_id =run_ids[2]
    try:
        os.mkdir('data/'  +run_id)
        os.mkdir('data/' + run_id + '/intermediary')
        os.mkdir('img/' + run_id )
    except OSError:
        print("Creation of the directory %s failed" % run_id)
    else:
        print("Successfully created the directory %s " % run_id)

    metadata_path = 'data/stock_data/2021-02-11-23-00-05_metadata.csv'
    df_metadata = pd.read_csv(metadata_path, sep=';', index_col='ticker')
    if timeseries_transformation:
        transformDataset( input_path='data/stock_data/2021-02-11-07-39-31_historic_data.csv'
                          , input_sep=';'
                          ,input_symbol='symbol'
                         , metadata_input_path = metadata_path, metadata_sep = ','
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
        df_pct_change = df_pct_change[df_pct_change.columns[((df_pct_change == 0).mean() <= 0.5)]]

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

            if d != 0: #and d > 4:
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

    def create_dataset(run_id):
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

        df_results_portfolio_used = df_results_portfolio_used[df_results_portfolio_used['portfolio_type'].isin(['markowitz_portfolio_baseline_full',
                                                                                                                'markowitz_portfolio_baseline'
                                                                                                                ,'markowitz_portfolio_with_forecast'
                                                                                                               #,'markowitz_portfolio_with_forecast_and adjusted covariance_matrix'
                                                                                                                ,'markowitz_portfolio_with_forecast_and_latent_features'
                                                                                                                 ])]


        df_results_portfolio_used = df_results_portfolio_used.reset_index()

        volatility_10 = []
        volatility_252 = []
        expected_returns_10 = []
        for backtest in df_results_portfolio_used['backtest_iteration'].unique():
            print('Calculating portfolio annual volatility for backtest {v_backtest}'.format(v_backtest =str(backtest)))
            print('Calculating portfolio expected returns for backtest {v_backtest}'.format(v_backtest =str(backtest)))
            df_original_backtest = df_original_close_full.head(len(df_original_close_full) - n_forecast * (backtest - 1))
            df_original_backtest_next = df_original_close_full.head(
                len(df_original_close_full) - n_forecast * (backtest))

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
                    data_next =df_original_backtest_next[stocks]
                    #print(data.shape)
                except:
                    print("Some of these stocks are not in the dataset: {b}_{s}".format(b=str(backtest), s=str(stocks)))

                # Calculating Portfolio volatility:

                # Calculate covariance matrix
                # cov_matrix_annual = returns.cov() * 252
                # Now calculate and show the portfolio variance using the formula :
                # Expected portfolio variance= WT * (Covariance Matrix) * W
                # Python: port_variance = np.dot(weights.T, np.dot(cov_matrix_annual, weights))

                # Now calculate and show the portfolio volatility using the formula :
                # Expected portfolio volatility= SQRT (WT * (Covariance Matrix) * W)
                # Python port_volatility = np.sqrt(port_variance)


                returns = data.pct_change()
                portfolio_vol_252 = np.sqrt(np.dot(weights.T, np.dot(returns.tail(252).cov() * 252, weights)))
                portfolio_vol_10 = np.sqrt(np.dot(weights.T, np.dot(returns.tail(10).cov() * 10, weights)))
                portfolio_vol_40 = np.sqrt(np.dot(weights.T, np.dot(returns.tail(40).cov() * 40, weights)))


                #cov_matrix_10 = log_returns.tail(10).cov() * 10
                #portfolio_var_10 =np.dot(weights.T, np.dot(cov_matrix_10, weights))
                #portfolio_vol_10 = np.sqrt(np.dot(weights.T, portfolio_var_10, weights))

                #expected_return_10 = np.sum(returns.mean() * weights) * 10
                expected_return_10 =  np.sum(weights * ((data_next.tail(1).values -data.tail(1).values)/data.tail(1).values) )
                risk_free_rate = 0.02
                expected_return_10 = expected_return_10 - risk_free_rate

                volatility_252.append(portfolio_vol_252)
                volatility_10.append(portfolio_vol_10)
                expected_returns_10.append(expected_return_10)

        df_volatility = pd.DataFrame(volatility_252, columns=['portfolio_volatility_252'])
        df_volatility['portfolio_volatility_10'] = volatility_10
        df_volatility['expected_returns_10'] = expected_returns_10

        df_results_portfolio_used_with_volatility = df_results_portfolio_used.join(df_volatility, how='left')
        df_results_portfolio_used_with_volatility['cumsum_profit'] = df_results_portfolio_used_with_volatility.groupby('portfolio_type')['profit'].cumsum()
        df_results_portfolio_used_with_volatility['10_expected_return'] = df_results_portfolio_used_with_volatility['expected_returns_10']
        df_results_portfolio_used_with_volatility['sharpe_ratio_10'] = df_results_portfolio_used_with_volatility['10_expected_return']\
                                                                       /df_results_portfolio_used_with_volatility['portfolio_volatility_10']

        metrics = ['sharpe_ratio',
         'sharpe_ratio_10',
         'expected_annual_return',
         'expected_returns_10',
         'annual_volatility',
         'portfolio_volatility_10']

        df_results_portfolio_used_with_volatility_pivot = df_results_portfolio_used_with_volatility.pivot(index='backtest_iteration', columns='portfolio_type', values=metrics)
        #  sharpe ratio deviation Model 1
        def calculate_percentage_deviation(baseline_model, challenger_model):
            return (challenger_model - baseline_model ) #/ baseline_model * 100

        common_metric_names = ['sharpe_ratio','return', 'volatility' ]
        i = 0
        for metric in ['sharpe_ratio','expected_annual_return','annual_volatility']:
            baseline_model = df_results_portfolio_used_with_volatility_pivot[(metric, 'markowitz_portfolio_baseline_full')]
            challenger_model = df_results_portfolio_used_with_volatility_pivot[(metric, 'markowitz_portfolio_baseline')]
            df_results_portfolio_used_with_volatility_pivot[('in-sample','markowitz_portfolio_{v_common_metric_names}_deviation_model_1'.format(v_common_metric_names = common_metric_names[i]))] = calculate_percentage_deviation(baseline_model, challenger_model)
            i = i+1

        i= 0
        for metric in ['sharpe_ratio_10','expected_returns_10','portfolio_volatility_10']:
            baseline_model = df_results_portfolio_used_with_volatility_pivot[(metric, 'markowitz_portfolio_baseline_full')]
            challenger_model = df_results_portfolio_used_with_volatility_pivot[(metric, 'markowitz_portfolio_baseline')]
            df_results_portfolio_used_with_volatility_pivot[('out-of-sample','markowitz_portfolio_{v_common_metric_names}_deviation_model_1'.format(v_common_metric_names = common_metric_names[i]))] = calculate_percentage_deviation(baseline_model, challenger_model)
            i = i + 1


        # Model 2
        i = 0
        for metric in ['sharpe_ratio','expected_annual_return','annual_volatility']:
            baseline_model = df_results_portfolio_used_with_volatility_pivot[(metric, 'markowitz_portfolio_baseline')]
            challenger_model = df_results_portfolio_used_with_volatility_pivot[(metric, 'markowitz_portfolio_with_forecast')]
            df_results_portfolio_used_with_volatility_pivot[('in-sample','markowitz_portfolio_{v_common_metric_names}_deviation_model_2'.format(v_common_metric_names = common_metric_names[i]))] = calculate_percentage_deviation(baseline_model, challenger_model)
            i = i+1

        i= 0
        for metric in ['sharpe_ratio_10','expected_returns_10','portfolio_volatility_10']:
            baseline_model = df_results_portfolio_used_with_volatility_pivot[(metric, 'markowitz_portfolio_baseline')]
            challenger_model = df_results_portfolio_used_with_volatility_pivot[(metric, 'markowitz_portfolio_with_forecast')]
            df_results_portfolio_used_with_volatility_pivot[('out-of-sample','markowitz_portfolio_{v_common_metric_names}_deviation_model_2'.format(v_common_metric_names = common_metric_names[i]))] = calculate_percentage_deviation(baseline_model, challenger_model)
            i = i + 1




        # model 3
        i = 0
        for metric in ['sharpe_ratio','expected_annual_return','annual_volatility']:
            baseline_model = df_results_portfolio_used_with_volatility_pivot[(metric, 'markowitz_portfolio_baseline')]
            challenger_model = df_results_portfolio_used_with_volatility_pivot[(metric, 'markowitz_portfolio_with_forecast_and_latent_features')]
            df_results_portfolio_used_with_volatility_pivot[('in-sample','markowitz_portfolio_{v_common_metric_names}_deviation_model_3'.format(v_common_metric_names = common_metric_names[i]))] = calculate_percentage_deviation(baseline_model, challenger_model)
            i = i+1

        i= 0
        for metric in ['sharpe_ratio_10','expected_returns_10','portfolio_volatility_10']:
            baseline_model = df_results_portfolio_used_with_volatility_pivot[(metric, 'markowitz_portfolio_baseline')]
            challenger_model = df_results_portfolio_used_with_volatility_pivot[(metric, 'markowitz_portfolio_with_forecast_and_latent_features')]
            df_results_portfolio_used_with_volatility_pivot[('out-of-sample','markowitz_portfolio_{v_common_metric_names}_deviation_model_3'.format(v_common_metric_names = common_metric_names[i]))] = calculate_percentage_deviation(baseline_model, challenger_model)
            print(baseline_model)
            print(challenger_model)
            print(calculate_percentage_deviation(baseline_model, challenger_model))
            i = i + 1



        return df_results_portfolio_used_with_volatility_pivot

    df_backtest_results_stocks = create_dataset(run_id=run_ids[0])
    df_backtest_results_etf = create_dataset(run_id=run_ids[2])



    import matplotlib.pyplot as plt
    import numpy as np

    models_dict = ['model_1' ,'model_2','model_3' ]
    metrics_dict = ['sharpe_ratio','volatility','return']
    metrics_name = ['sharpe_ratio','volatility','return']
    data_type_dict = ['in-sample','out-of-sample']


    for model in models_dict:
        i = 0
        for metric in metrics_dict:
            name = 'markowitz_portfolio_{v_metric}_deviation_{v_model}'.format(v_model = model, v_metric = metric)
            print(name)
            #PLOT 1
            labels1 = df_backtest_results_etf.index
            labels2 = df_backtest_results_stocks.index

            ylabel = '{v_metrics_name} absolute deviation from baseline model'.format(v_metrics_name = metrics_name[i])
            xlabel = 'Backtest iterations'

            x1 = np.arange(len(labels1))  # the label locations
            x2 = np.arange(len(labels2))  # the label locations

            width = 0.35  # the width of the bars
            fig, axs = plt.subplots(1, 2, figsize=(16, 5), constrained_layout=True,
                                    sharex=True, sharey=True,)
            j=0
            for nn, ax in enumerate(axs.flat):
                data_type = data_type_dict[j]
                title = '{v_data_type} results'.format(v_data_type=data_type)

                bar1 = df_backtest_results_etf[(data_type, name)]
                bar1_mean = bar1.mean().round(3)
                bar2 = df_backtest_results_stocks[(data_type, name)]
                bar2_mean = bar2.mean().round(3)

                rects1 = ax.bar(x1 - width / 2, bar1, width, label='Final model (ETF dataset) average: {v_bar1_mean}'.format(v_bar1_mean = bar1_mean), color=['gray'])
                rects2 = ax.bar(x2 + width / 2, bar2, width, label='Final model (stock dataset) average: {v_bar2_mean}'.format(v_bar2_mean = bar2_mean),
                                color=['darkblue'])

                # Add some text for labels, title and custom x-axis tick labels, etc.
                ax.set_ylabel(ylabel)
                ax.set_xlabel(xlabel)

                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                #ax.spines['left'].set_visible(False)

                ax.set_title(title)
                ax.set_xticks(x1)
                ax.set_xticklabels(labels1)
                ax.legend(frameon=False)
                #plt.legend(frameon=False)


                j = j +1

            fig.tight_layout()

            #plt.box(False)
            plt.savefig('img/{v_run_id}/backtest_results_{v_model}_{v_metric}.png'.format(
                v_data_type=data_type.replace('-', '_'), v_model=model, v_metric=metric, i=str(column),
                v_run_id=run_id), dpi=300)

            i = i + 1




    '''
    #PLOT 1
    labels1 = df_backtest_results_etf.index
    labels2 = df_backtest_results_stocks.index
    bar1 = df_backtest_results_etf[(data_type,name)]
    bar2 = df_backtest_results_stocks[(data_type,name)]
    ylabel = '{v_metrics_name} absolute deviation from baseline model'.format(v_metrics_name = metrics_name[i])
    xlabel = 'Backtest iterations'
    title  = '{v_data_type} results'.format(v_data_type = data_type)
    x1 = np.arange(len(labels1))  # the label locations
    x2 = np.arange(len(labels2))  # the label locations

    width = 0.35  # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(x1 - width / 2, bar1, width, label='Final model (ETF dataset)', color=['gray'])
    rects2 = ax.bar(x2 + width / 2, bar2, width, label='Final model (stock dataset)' , color=['darkblue'])

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)

    ax.set_title(title)
    ax.set_xticks(x1)
    ax.set_xticklabels(labels1)
    ax.legend()
    fig.tight_layout()
    plt.legend(frameon=False)
    plt.box(False)
    plt.savefig('img/{v_run_id}/{v_data_type}_backtest_results_{v_model}_{v_metric}.png'.format(v_data_type = data_type.replace('-','_') ,v_model = model, v_metric = metric, i=str(column), v_run_id=run_id), dpi=300)

    i = i +1

    
    # PLOT 2
    labels1 = df_backtest_results_etf.index
    labels2 = df_backtest_results_stocks.index
    bar1 = df_backtest_results_etf[('in-sample','markowitz_portfolio_sharpe_ratio_deviation_model_1')]
    bar2 = df_backtest_results_stocks[('in-sample','markowitz_portfolio_sharpe_ratio_deviation_model_1')]
    ylabel = 'Sharpe ratio percentage deviation from baseline model'
    xlabel = 'Backtest iterations'
    title  = 'In-sample results'
    x1 = np.arange(len(labels1))  # the label locations
    x2 = np.arange(len(labels2))  # the label locations

    width = 0.35  # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(x1 - width / 2, bar1, width, label='Final model (ETF dataset)', color=['gray'])
    rects2 = ax.bar(x2 + width / 2, bar2, width, label='Final model (stock dataset)' , color=['darkblue'])

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)

    ax.set_title(title)
    ax.set_xticks(x1)
    ax.set_xticklabels(labels1)
    ax.legend()
    fig.tight_layout()
    plt.legend(frameon=False)
    plt.box(False)
    plt.savefig('img/{v_run_id}/in_sample_backtest_results_change.png'.format(i=str(column), v_run_id=run_id), dpi=300)
    #plt.show()


    # PLOT 3
    labels1 = df_backtest_results_etf.index
    labels2 = df_backtest_results_stocks.index
    bar1 = df_backtest_results_etf[('out-of-sample', 'markowitz_portfolio_sharpe_ratio_deviation_model_2')]
    bar2 = df_backtest_results_stocks[('out-of-sample', 'markowitz_portfolio_sharpe_ratio_deviation_model_2')]
    ylabel = 'Sharpe ratio percentage deviation from baseline model'
    xlabel = 'Backtest iterations'
    title = 'Out-ot-sample results'
    x1 = np.arange(len(labels1))  # the label locations
    x2 = np.arange(len(labels2))  # the label locations

    width = 0.35  # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(x1 - width / 2, bar1, width, label='Final model (ETF dataset)', color=['gray'])
    rects2 = ax.bar(x2 + width / 2, bar2, width, label='Final model (stock dataset)', color=['darkblue'])

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)

    ax.set_title(title)
    ax.set_xticks(x1)
    ax.set_xticklabels(labels1)
    #ax.legend()
    fig.tight_layout()
    plt.legend(frameon=False)
    plt.box(False)
    plt.savefig('img/{v_run_id}/out_of_sample_backtest_results_full_change.png'.format(i=str(column), v_run_id=run_id),
                dpi=300)

    # PLOT 4
    labels1 = df_backtest_results_etf.index
    labels2 = df_backtest_results_stocks.index
    bar1 = df_backtest_results_etf[('in-sample', 'markowitz_portfolio_sharpe_ratio_deviation_model_2')]
    bar2 = df_backtest_results_stocks[('in-sample', 'markowitz_portfolio_sharpe_ratio_deviation_model_2')]
    ylabel = 'Sharpe ratio percentage deviation from baseline model'
    xlabel = 'Backtest iterations'
    title = 'In-sample results'
    x1 = np.arange(len(labels1))  # the label locations
    x2 = np.arange(len(labels2))  # the label locations

    width = 0.35  # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(x1 - width / 2, bar1, width, label='Final model (ETF dataset)', color=['gray'])
    rects2 = ax.bar(x2 + width / 2, bar2, width, label='Final model (stock dataset)', color=['darkblue'])

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)

    ax.set_title(title)
    ax.set_xticks(x1)
    ax.set_xticklabels(labels1)
    ax.legend()
    fig.tight_layout()
    plt.legend(frameon=False)
    plt.box(False)
    plt.savefig('img/{v_run_id}/in_sample_backtest_results_full_change.png'.format(i=str(column), v_run_id=run_id), dpi=300)
    # plt.show()

    # PLOT 4 adjusted
    labels1 = df_backtest_results_etf.index
    labels2 = df_backtest_results_stocks.index
    bar1 = df_backtest_results_etf[('in-sample', 'markowitz_portfolio_sharpe_ratio_deviation_model_2')]
    bar2 = df_backtest_results_stocks[('in-sample', 'markowitz_portfolio_sharpe_ratio_deviation_model_2')]
    ylabel = 'Sharpe ratio percentage deviation from baseline model'
    xlabel = 'Backtest iterations'
    title = 'In-sample results'
    x1 = np.arange(len(labels1))  # the label locations
    x2 = np.arange(len(labels2))  # the label locations

    width = 0.35  # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(x1 - width / 2, bar1, width, label='Final model (ETF dataset)', color=['gray'])
    rects2 = ax.bar(x2 + width / 2, bar2, width, label='Final model (stock dataset)', color=['darkblue'])

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)

    ax.set_title(title)
    ax.set_xticks(x1)
    ax.set_xticklabels(labels1)
    ax.legend()
    fig.tight_layout()
    plt.legend(frameon=False)
    plt.box(False)
    plt.savefig('img/{v_run_id}/in_sample_backtest_results_full_change.png'.format(i=str(column), v_run_id=run_id),
                dpi=300)
    # plt.show()
    '''
    '''
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

    df_stock_allocation = df_stock_allocation.join(df_metadata, how='left')



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

    df_stock_allocation['index'] = df_stock_allocation.index
    y_value = df_stock_allocation.columns[0]
    y_value ='20_markowitz_portfolio_with_forecast_and_latent_features'
    df_stock_allocation_filtered = df_stock_allocation[[y_value,'longName']].dropna().sort_values(by=[y_value], ascending=False)
    ax = df_stock_allocation_filtered.plot.barh(x='longName', y=y_value)
    fig = ax.get_figure()
    fig.savefig('img/{v_run_id}/discrete allocation.png'.format(v_type =type, v_run_id=run_id), dpi=300)
    print(df_stock_allocation_filtered)

    '''



