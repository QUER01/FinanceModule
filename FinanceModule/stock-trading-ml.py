import matplotlib.pyplot as plt
from keras.models import Model
import numpy as np
from tensorflow import set_random_seed
set_random_seed(4)
from FinanceModule.util import createDataset\
    ,transformDataset\
    ,splitTrainTest\
    ,defineModel\
    ,defineAutoencoder\
    ,predictAutoencoder\
    ,getLatentFeaturesSimilariryAndReturns\
    ,getReconstructionErrorsAndReturns

from FinanceModule.quandlModule import Quandl
import copy
import pandas as pd
from sklearn import preprocessing
from termcolor import colored
from pulp import *
from pandas.tseries.offsets import DateOffset


plt.xkcd()










# --------------------------------------- #
#             MAIN
# --------------------------------------- #

apiKey = "fKx3jVSpXasnsXKGnovb"
market = 'FSE'


# Download list of API codes
Quandl = Quandl(apiKey)
df_tickers = Quandl.getStockExchangeCodes()
# filter data API codes
l_tickers = df_tickers['code'].tolist()


print('-'* 50)
print('PART I: Timeseries Cleaning' )
print('-'* 50)


# parameters
history_points = 150
test_split = 0.9
n_forecast = 10
n_tickers = 100
n_days = 250*4
sectors = ['FINANCE', 'CONSUMER SERVICES', 'TECHNOLOGY',
       'CAPITAL GOODS', 'BASIC INDUSTRIES', 'HEALTH CARE',
       'CONSUMER DURABLES', 'ENERGY','TRANSPORTATION', 'CONSUMER NON-DURABLES']
backtest_days = 100
scaled_mse_arr = []
plot_reults = True



transformDataset( input_path='data/historical_stock_prices.csv', input_sep=','
                 , metadata_input_path = 'data/historical_stocks.csv', metadata_sep = ','
                 ,output_path='data/historical_stock_prices_filtered.csv', output_sep=';'
                 ,filter_sectors = sectors
                 ,n_tickers = n_tickers, n_last_values = n_days )

print('-' * 5 + 'Loading the dataset from disk')
df = pd.read_csv('data/historical_stock_prices_filtered.csv', sep=';',index_col='date')
df.index = pd.to_datetime(df.index)



# Get tickers as a list
print('-' * 5 + 'Getting list of unique tickers')
l_tickers_new = df.columns.str.split('_')
def column(matrix, i):
    return [row[i] for row in matrix]
l_tickers_unique = np.unique(column(l_tickers_new,0))

df_all = df


for d in range(int(backtest_days/n_forecast)+1)[::-1] :
    if d != 0 :
        print('-' * 5 + 'Backtest Iteration ' + str(d))
        df = df_all.head(len(df_all) - n_forecast*d)


        print('-' * 5 + 'Starting for loop over all tickers')
        for i in range(0,len(l_tickers_unique)):

            column = l_tickers_unique[i]
            print('-' * 10 + 'Iteration: ' + str(i)+'/' + str(len(l_tickers_unique)) + '  ticker name:' + column)
            df_filtered = df.filter(like= column, axis=1)
            if len(df_filtered.columns) != 5:
                print(colored('-' * 15 + 'Not all columns available' ,'red'))

            else:
                # data imputation
                df_filtered = df_filtered.T.fillna(df_filtered.mean(axis=1)).T
                df_filtered = df_filtered.fillna(method='ffill')
                df_filtered = df_filtered.tail(n_days)

                '''
                print('-' * 15 + ' PART I: Timeseries Evaluation for : ' + column )
                print('-' * 20 + 'Transform the timeseries into an supervised learning problem' )
                next_day_open_values_normalised,next_day_open_values, ohlcv_histories_normalised, technical_indicators, data_normaliser, y_normaliser  = createDataset(df_filtered)
            
            
                # split data into train and test datasets
                print('-' * 20 + 'Split data into train and test datasets' )
                n = int(ohlcv_histories_normalised.shape[0] * test_split)
                ohlcv_train, ohlcv_test = splitTrainTest(values=ohlcv_histories_normalised,n=n)
                tech_ind_train, tech_ind_test = splitTrainTest(values=technical_indicators,n=n)
                y_train, y_test = splitTrainTest(values=next_day_open_values_normalised,n=n)
                unscaled_y_train, unscaled_y_test = splitTrainTest(values=next_day_open_values,n=n)
            
            
                # model architecture
                print('-' * 20 + 'Design the model' )
                model = defineModel(ohlcv_histories_normalised = ohlcv_histories_normalised,technical_indicators = technical_indicators , verbose = 0)
                # fit model
                print('-' * 20 + 'Fit the model' )
                model.fit(x=[ohlcv_train, tech_ind_train], y=y_train, batch_size=32, epochs=50, shuffle=True, validation_split=0.1, verbose=0)
            
            
                # evaluate model
                print('-' * 20 + 'Evaluate the model' )
                y_test_predicted = model.predict([ohlcv_test, tech_ind_test])
                y_test_predicted = y_normaliser.inverse_transform(y_test_predicted)
                y_predicted = model.predict([ohlcv_histories_normalised, technical_indicators])
                y_predicted = y_normaliser.inverse_transform(y_predicted)
                assert unscaled_y_test.shape == y_test_predicted.shape
                real_mse = np.mean(np.square(unscaled_y_test - y_test_predicted))
                scaled_mse = real_mse / (np.max(unscaled_y_test) - np.min(unscaled_y_test)) * 100
                print('-' * 25 + 'Scaled MSE: ' + str(scaled_mse) )
                scaled_mse_arr.append([column, d scaled_mse])
            
            
                # plot the results
                print('-' * 20 + 'Plot the results' )
                plt.gcf().set_size_inches(9, 16, forward=True)
                fig, (ax1, ax2) = plt.subplots(2, 1)
                #fig.suptitle('Horizontally stacked subplots')
                start = 0
                end = -1
                real = ax1.plot(next_day_open_values[start:end], label='real')
                pred = ax1.plot(y_predicted[start:end], label='predicted')
            
                real = ax2.plot(unscaled_y_test[start:end], label='real')
                pred = ax2.plot(y_test_predicted[start:end], label='predicted')
            
                plt.legend(['Real', 'Predicted'])
                plt.show()
                plt.savefig('img/' +column + '_evaluation.png')
            
                from datetime import datetime
                model.save('img/' +column + '_basic_model.h5')
            
                '''

                ## forecast
                print('-' * 15 + 'PART III: ' + str(n_forecast) + '-Step-Forward Prediction ')
                for j in range(0, n_forecast):

                    print('-' * 20 + 'Transform the timeseries into an supervised learning problem')
                    next_day_open_values_normalised, next_day_open_values, ohlcv_histories_normalised, technical_indicators, data_normaliser, y_normaliser = createDataset(
                        df_filtered)

                    if j == 0:
                        # initialize the dataset
                        print('-' * 20 + 'Initialize certain objects')

                        # model architecture
                        print('-' * 25 + 'Design the model')
                        model = defineModel(ohlcv_histories_normalised=ohlcv_histories_normalised,
                                            technical_indicators=technical_indicators, verbose=0)
                        # fit model
                        print('-' * 25 + 'Fit the model')
                        model.fit(x=[ohlcv_histories_normalised, technical_indicators],
                                  y=next_day_open_values_normalised,
                                  batch_size=32, epochs=50, shuffle=True, validation_split=0.1, verbose=0)

                    # evaluate model
                    print('-' * 20 + 'Predict with the model')
                    y_predicted = model.predict([ohlcv_histories_normalised, technical_indicators])
                    y_predicted = y_normaliser.inverse_transform(y_predicted)

                    print('-' * 20 + 'Creating the result dataset')
                    n_output = 1
                    # identifying the predicted output
                    newValue = y_predicted[-n_output:, 0].flat[0]
                    # identifying the date index
                    add_dates = [df_filtered.index[-1] + DateOffset(days=x) for x in range(1, n_output + 1)]

                    df_predict_ohlcv = pd.DataFrame(data=np.array([df_filtered.iloc[-1, 0]
                                                                      , df_filtered.iloc[-1, 1]
                                                                      , df_filtered.iloc[-1, 2]
                                                                      , newValue
                                                                      , df_filtered.iloc[-1, 4]]).reshape(1, 5),
                                                    index=add_dates[0:n_output], columns=df_filtered.columns)

                    df_filtered = df_filtered.append(df_predict_ohlcv, sort=False)

                    # initialize the result dataset
                # We need to initialize these values here because they depend on the firsts computations
                if 'df_result' not in locals():
                    print('-' * 20 + 'Iteration: ' + str(i) + '   Initialize the result dataset')
                    df_result = pd.DataFrame(index=df_filtered.index)

                print('-' * 15 + ' Creating the result dataset')
                df_predicted = pd.DataFrame(data=y_predicted, index=df_filtered.tail(len(y_predicted)).index,
                                            columns=[column + '/prediction'])

                # add ohlcv columns to the dataset
                df_result = df_result.join(df_filtered)
                # add model prediction to the dataset
                df_result = df_result.join(df_predicted)

                if plot_reults:
                    print('-' * 15 + ' Plot the results of the ' + str(n_forecast) + '-Step-Forward Prediction ')
                    plt.figure(figsize=(20, 5))
                    plt.plot(df_filtered.index, df_filtered[df_filtered.columns[0]])
                    plt.plot(df_filtered.index, df_filtered[df_filtered.columns[1]])
                    plt.plot(df_filtered.index, df_filtered[df_filtered.columns[2]])
                    plt.plot(df_filtered.index, df_filtered[df_filtered.columns[3]])
                    plt.plot(df_predicted.index, df_predicted[df_predicted.columns[0]])

                    # plt.plot(df_filtered.index, df_filtered['Prediction_Future'], color='r')
                    # plt.plot(df_proj.index, df_proj['Prediction'], color='y')
                    plt.legend(
                        [df_filtered.columns[0], df_filtered.columns[1], df_filtered.columns[2], df_filtered.columns[3],
                         df_predicted.columns[0]])
                    plt.xticks(fontsize=18)
                    plt.yticks(fontsize=16)
                    plt.show()

                plt.savefig('img/' + column + '.png')

                # clean up
                del df_predict_ohlcv
                del add_dates
                del newValue
                del y_predicted
                del model
                del next_day_open_values_normalised, next_day_open_values, ohlcv_histories_normalised, technical_indicators, data_normaliser, y_normaliser
                del df_filtered

        # save to disk
        print('-' * 10 + ' Save results to disk Backtest number: ' + str(d) + ' as: data/df_result_' + str(d) + '.csv')
        df_result.to_csv('data/df_result_' + str(d) + '.csv', sep = ';')


# End of for loops

profits_option_1 = []
tickers_option_1 = []

profits_option_2 = []
tickers_option_2 = []

profits_option_3 = []
tickers_option_3 = []


avg_return_column = 'avg_returns_last50_days'
avg_return_days = 50

forecasting_days = 10

n_stocks_per_bin = 4
budget = 1000
n_bins = 6



for d in range(int(backtest_days/n_forecast)+1)[::-1] :

    if d != 0:
        print('-' * 5 + 'Backtest Iteration ' + str(d))

        df_result = pd.read_csv('data/df_result_' + str(d) + '.csv', sep=';', index_col='Unnamed: 0')

        ## Deep Portfolio
        print('-' * 15 + 'PART IV: Autoencoder Deep Portfolio Optimization')

        print('-' * 20 + 'Create dataset')
        df_result_close = df_result.filter(like='Close',axis=1)
        df_result_close = df_result_close.dropna(axis=1, how='any',thresh = 0.90*len(df_result))
        original_data = df_result_close.tail(n_days)


        print('-' * 20 + 'Transform dataset')
        df_pct_change = df_result_close.pct_change(1).astype(float)
        df_pct_change = df_pct_change.replace([np.inf, -np.inf], np.nan)
        df_pct_change = df_pct_change.fillna(method='ffill')
        # the percentage change function will make the firstrow equal to nan
        df_pct_change = df_pct_change.tail(len(df_pct_change)-1)



        print('-' * 20 + 'Step 1 : Returns vs. recreation eroor (L2-norm)')
        print('-' * 25 + 'Transform dataset with MinMax Scaler')
        df_scaler = preprocessing.MinMaxScaler()
        df_pct_change_normalised = df_scaler.fit_transform(df_pct_change)

        #define autoencoder
        print('-' * 25 + 'Define autoencoder model')
        num_stock = len(df_pct_change.columns)
        autoencoder = defineAutoencoder(num_stock= num_stock, encoding_dim = 5, verbose =0)

        # train autoencoder
        print('-' * 25 + 'Train autoencoder model')
        autoencoder.fit(df_pct_change_normalised, df_pct_change_normalised, shuffle=True, epochs=500, batch_size=50, verbose= 0)

        # predict autoencoder
        print('-' * 25 + 'Predict autoencoder model')
        reconstruct = autoencoder.predict(df_pct_change_normalised)

        # Inverse transform dataset with MinMax Scaler
        print('-' * 25 + 'Inverse transform dataset with MinMax Scaler')
        reconstruct_real = df_scaler.inverse_transform(reconstruct)


        print('-' * 25 + 'Calculate L2 norm as reconstruction loss metric')
        df_returns_l2norm = getReconstructionErrorsAndReturns(  original_data = df_pct_change
                                                              , reconstructed_data = reconstruct_real
                                                              , original_data_abs = original_data
                                                                , forecasting_days=forecasting_days  )








        print('-' * 20 + 'Step 2 : Returns vs. latent feature similarity')

        print('-' * 25 + 'Transpose dataset')
        df_pct_change_transposed = df_pct_change.transpose()

        print('-' * 25 + 'Transform dataset with MinMax Scaler')
        df_scaler = preprocessing.MinMaxScaler()
        df_pct_change_transposed_normalised = df_scaler.fit_transform(df_pct_change_transposed)

        # define autoencoder
        print('-' * 25 + 'Define autoencoder model')
        num_stock = len(df_pct_change_transposed.columns)
        autoencoderTransposed = defineAutoencoder(num_stock= num_stock, encoding_dim = 5, verbose = 0)

        # train autoencoder
        print('-' * 25 + 'Train autoencoder model')
        autoencoderTransposed.fit(df_pct_change_transposed_normalised, df_pct_change_transposed_normalised, shuffle=True, epochs=500, batch_size=50, verbose=0)

        # Get the latent feature vector
        print('-' * 25 + 'Get the latent feature vector')
        autoencoderTransposedLatent = Model(inputs=autoencoderTransposed.input, outputs=autoencoderTransposed.get_layer('my_latent').output)

        # predict autoencoder model
        print('-' * 25 + 'Predict autoencoder model')
        latent_features = autoencoderTransposedLatent.predict(df_pct_change_transposed_normalised)

        print('-' * 25 + 'Calculate L2 norm as similarity metric')
        df_returns_similarity = getLatentFeaturesSimilariryAndReturns( original_data= df_pct_change
                                                                      ,latent_features = latent_features
                                                                      ,original_data_abs = original_data
                                                                      , forecasting_days = forecasting_days)

        if plot_reults:
            print('-' * 25 + 'Plot the results')
            top_n = 5
            df_returns_similarity_top_n = df_returns_similarity.iloc[0:top_n,:]
            df_returns_l2norm_top_n = df_returns_l2norm.iloc[0:top_n,:]

            bottom_n = 5
            df_returns_similarity_bottom_n = df_returns_similarity.tail(bottom_n)
            df_returns_l2norm_bottom_n = df_returns_l2norm.tail(bottom_n)


            print('-' * 30 + 'Plot top 5 most similar time series')
            df_plot = df_returns_similarity_top_n
            plt.figure(figsize=(11, 6))
            for stock in df_plot['stock_name']:
                plt.plot(df_result.index, df_result.filter(like=stock).filter(like='Close'), label= stock)

            plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            plt.title('Top ' + str(top_n) + ' most similar stocks based on latent feature value')
            plt.xlabel("Dates")
            plt.ylabel("Stock Value")
            plt.show()




            print('-' * 30 + 'Plot the 5 least similar time series')
            df_plot = df_returns_similarity_bottom_n
            plt.figure(figsize=(11, 6))
            for stock in df_plot['stock_name']:
                plt.plot(df_result.index, df_result.filter(like=stock).filter(like='Close'), label= stock)

            plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            plt.title('Bottom ' + str(top_n) + ' most similar stocks based on latent feature value')
            plt.xlabel("Dates")
            plt.ylabel("Stock Value")
            plt.show()


        print('-' * 20 + 'Step 3: Create Portfolio ')
        print('-' * 25 + 'Join the datasets from the similarity score and the reconstruction error with some metadata')
        df_returns_similarity.stock_name = df_returns_similarity['stock_name'].str.split('_', n=1 ,expand = True)
        df_portfolio = df_returns_similarity[['stock_name'
                                            ,'latent_value'
                                            ,'avg_returns_last10_days'
                                            ,'avg_returns_last50_days'
                                            ,'avg_returns_last100_days'
                                            ,'current_price']].join(df_returns_l2norm[['L2norm']] , how = 'left' )

        df_portfolio = df_portfolio.set_index('stock_name')

        df_metadata = pd.read_csv('data/historical_stocks.csv', sep=',')
        df_metadata = df_metadata.rename(columns={'ticker': 'stock_name'})
        df_metadata = df_metadata.set_index('stock_name')

        df_portfolio = df_portfolio.join(df_metadata, how='left')


        # remove very high values
        df_portfolio = df_portfolio[df_portfolio['L2norm']< df_portfolio['L2norm'].quantile(0.99)]

        # calculate bins
        df_portfolio['latent_value_quartile'] = pd.qcut(df_portfolio.latent_value, n_bins, precision=0)

        # calculate return*recreation error
        df_scaler_recreation_error = preprocessing.MinMaxScaler()
        df_portfolio['recreation_error_scaled_inverse'] = 1 - df_scaler_recreation_error.fit_transform(df_portfolio[['L2norm']].values)
        df_portfolio['avg_return_recreation_error'] = df_portfolio[avg_return_column] * df_portfolio['recreation_error_scaled_inverse']

        if plot_reults:
            # plot the results
            print('-' * 25 + 'Plot the results')
            df_plot =df_portfolio[df_portfolio[avg_return_column] > 0]
            df_plot = df_plot[df_plot['latent_value']< df_plot['latent_value'].quantile(0.99)]
            groups = df_plot.groupby('sector')

            # Plot
            fig, ax = plt.subplots(figsize=(10, 6))
            for name, group in groups:
                ax.plot(group.latent_value, group[avg_return_column]*100, marker='o', linestyle='', ms=12, label=name)

            plt.title('Average retuns vs. similarity metric (latent feature values)')
            plt.xlabel("Latent Feature Value")
            plt.ylabel(avg_return_column)
            ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            plt.show()




            print('-' * 25 + 'Plot results')
            df_plot = df_portfolio[df_portfolio[avg_return_column] > 0]
            df_plot = df_plot[df_plot['L2norm']< df_plot['L2norm'].quantile(0.99)]
            groups = df_plot.groupby('sector')

            # Plot
            fig, ax = plt.subplots(figsize=(10, 6))
            for name, group in groups:
                ax.plot(group.L2norm, group[avg_return_column]*100, marker='o', linestyle='', ms=12, label=name)

            plt.title('Average return vs. recreation error')
            plt.xlabel("Recreation error")
            plt.ylabel("average returns last 10 days in %")
            ax.legend(loc='best')
            plt.show()




            df_plot = df_portfolio[df_portfolio[avg_return_column] > 0]

            df_plot = df_plot[df_plot[avg_return_column]< df_plot[avg_return_column].quantile(0.9)]
            df_plot = df_plot[df_plot['L2norm']< df_plot['L2norm'].quantile(0.9)]
            groups = df_plot.groupby('latent_value_quartile')

            # Plot
            fig, ax = plt.subplots(figsize=(10, 6))
            for name, group in groups:
                ax.plot(group.L2norm, group[avg_return_column]*100, marker='o', linestyle='', ms=12, label=name)

            plt.title('Average return vs. recreation error by similarity quartiles (colors)')
            plt.xlabel("Recreation error")
            plt.ylabel("Average returns last 10 days in %")
            ax.legend(title='Similarity Quartiles', loc='best')
            plt.show()


        """
        total money available: 1000 €
        distribute at 
            least 100€ in x1
            least 100€ in x2
            least 100€ in x3
            least 100€ in x4
        
        
        Profit = max 2*x1 + 3*x2 + 4*x3 + 5*x4 
        s.t.
        
        x1 >= 1
        x2 >= 1
        x3 >= 1
        x4 >= 1
        
        2*x1 >= 100
        3*x2 >= 100
        4*x3 >= 100
        5*x4 >= 100
        """


        def portfolio_selection(df_portfolio , ranking_colum , profits, tickers,  n_stocks_per_bin, budget ,n_bins,group_by = True):
            n_stocks_total = n_stocks_per_bin * n_bins
            if group_by == True:
                df_portfolio['rn'] = df_portfolio.sort_values([ranking_colum], ascending=[False]) \
                                                 .groupby(['latent_value_quartile']) \
                                                 .cumcount() + 1

                df_portfolio_selected_stocks = df_portfolio[df_portfolio['rn'] <= n_stocks_per_bin]
                print(df_portfolio_selected_stocks.__len__())
            else:
                df_portfolio['rn'] = df_portfolio[ranking_colum].rank(method='max', ascending=False)
                df_portfolio_selected_stocks = df_portfolio[df_portfolio['rn'] <= n_stocks_total]
                print(df_portfolio_selected_stocks.__len__())

            var_names = ['x' + str(i) for i in range(n_stocks_total)]
            x_int = [LpVariable(i, lowBound=0, cat='Integer') for i in var_names]

            my_lp_problem = pulp.LpProblem("Portfolio_Selection_LP_Problem", pulp.LpMaximize)
            # Objective function
            my_lp_problem += lpSum([x_int[i] * df_portfolio_selected_stocks['current_price'].iloc[i] for i in range(n_stocks_total)]) <= budget
            # Constraints
            my_lp_problem += lpSum(
                [x_int[i] * df_portfolio_selected_stocks['current_price'].iloc[i] for i in range(n_stocks_total)])

            for i in range(n_stocks_total):
                my_lp_problem += x_int[i] * df_portfolio_selected_stocks['current_price'].iloc[
                    i] <= budget / n_stocks_total

            my_lp_problem.solve()
            pulp.LpStatus[my_lp_problem.status]


            bought_volume_arr = []
            for variable in my_lp_problem.variables():
                print("{} = {}".format(variable.name, variable.varValue))
                bought_volume_arr.append(variable.varValue)

            df_portfolio_selected_stocks['bought_volume'] = bought_volume_arr
            print('1')

            value_after_x_days_arr = []
            for index in df_portfolio_selected_stocks.index:
                value_after_x_days = df_result.filter(like=index + '_Close').tail(1).iloc[0,0]
                value_after_x_days_arr.append(value_after_x_days)

            df_portfolio_selected_stocks['value_after_x_days'] = value_after_x_days_arr

            df_portfolio_selected_stocks['delta'] = df_portfolio_selected_stocks['value_after_x_days'] -df_portfolio_selected_stocks['current_price']
            df_portfolio_selected_stocks['pnl'] = df_portfolio_selected_stocks['delta'] * df_portfolio_selected_stocks['bought_volume']
            print('Profit for iteration ' + str(d) + ': '  + str(df_portfolio_selected_stocks['pnl'].sum()))

            profits = [df_portfolio_selected_stocks['pnl'].sum()]

            return profits , df_portfolio_selected_stocks


        # merge the reults


        profits_option_1, df_portfolio_selected_stocks_option_1 = portfolio_selection(df_portfolio= df_portfolio
                                               , ranking_colum=avg_return_column
                                               , profits = profits_option_1
                                               , tickers=tickers_option_1
                                               , n_stocks_per_bin = n_stocks_per_bin
                                               , n_bins = n_bins
                                               , budget = budget)

        profits_option_2, df_portfolio_selected_stocks_option_2 = portfolio_selection(df_portfolio= df_portfolio
                                               , ranking_colum=avg_return_column
                                               , profits = profits_option_2
                                               , tickers=tickers_option_2
                                               , group_by=False
                                               , n_stocks_per_bin=n_stocks_per_bin
                                               , n_bins = n_bins
                                               , budget = budget)

        profits_option_3, df_portfolio_selected_stocks_option_3 = portfolio_selection(df_portfolio= df_portfolio
                                               , ranking_colum='avg_return_recreation_error'
                                               , profits = profits_option_3
                                               , tickers=tickers_option_3
                                               , n_stocks_per_bin=n_stocks_per_bin
                                               , n_bins = n_bins
                                               , budget=budget)


        print('-' * 25 + 'Merging the portfolio optimization results and compare them')
        df_portfolio_selected_stocks_option_1['options'] = 'avgerage returns last x days with grouping'
        df_portfolio_selected_stocks_option_2['options'] = 'average returns last x days'
        df_portfolio_selected_stocks_option_3['options'] = 'avgerage returns last x days * recreation error with grouping'

        df_portfolio_selected_stocks_option_1['backtest_iteration'] = d
        df_portfolio_selected_stocks_option_2['backtest_iteration'] = d
        df_portfolio_selected_stocks_option_3['backtest_iteration'] = d

        df_portfolio_selected_stocks_option_1['total_profit'] = profits_option_1[0]
        df_portfolio_selected_stocks_option_2['total_profit'] = profits_option_2[0]
        df_portfolio_selected_stocks_option_3['total_profit'] = profits_option_3[0]

        df_portfolio_selection_results = df_portfolio_selected_stocks_option_1.append(df_portfolio_selected_stocks_option_2).append(df_portfolio_selected_stocks_option_3)

    if 'df_portfolio_selection_results_final' not in locals():
        df_portfolio_selection_results_final = df_portfolio_selection_results
    else:
        df_portfolio_selection_results_final = df_portfolio_selection_results_final.append(df_portfolio_selection_results)

print(df_portfolio_selection_results_final.to_string())

"""
if plot_reults:
    print('-' * 10 + 'Plot results')
    df_plot = df_portfolio_selection_results_final
    plt.plot(df_plot.backtest_iteration, df_plot.total_profit, label=df_plot.index)
    
    plt.title('Backtest Profits')
    plt.xlabel("Recreation error")
    plt.ylabel("average returns last 10 days in %")
    ax.legend(loc='best')
    plt.show()



# --------------------------------------------- #
#       WORK IN PROGRESS
# --------------------------------------------- #

latent_value_qcut = pd.qcut(df_portfolio['latent_value'], 4)





which_stock1 = 11
which_stock2 = 45

# now decoded last price plot
stock_autoencoder_1 = copy.deepcopy(reconstruct_real[:, which_stock1])
stock_autoencoder_1[0] = 0
stock_autoencoder_1 = stock_autoencoder_1.cumsum()
stock_autoencoder_1 += (df_pct_change.iloc[0, which_stock1])

# now decoded last price plot
stock_autoencoder_2 = copy.deepcopy(reconstruct_real[:, which_stock2])
stock_autoencoder_2[0] = 0
stock_autoencoder_2 = stock_autoencoder_2.cumsum()
stock_autoencoder_2 += (df_pct_change.iloc[0, which_stock2])



## plot for comparison

plt.figure(figsize=(20, 5))
plt.plot(df_pct_change.index, original_data.iloc[:, which_stock1], color='b')
plt.plot(df_pct_change.index, original_data.iloc[:, which_stock2], color='r')
#plt.plot(df_pct_change.index, stock_autoencoder_1, color='y')
plt.legend(['Original' + str(which_stock1), 'Original' + str(which_stock2)])
plt.xticks(fontsize=18)
plt.yticks(fontsize=16)
plt.show()

"""