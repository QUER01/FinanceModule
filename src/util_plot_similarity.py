
def plot_stock_similarity(df, stocks, n, type):

    print('-' * 15 + ' Plot the original timeseries')
    plt.figure(figsize=(22, 10))
    plt.box(False)
    df_filtered_base = df.filter(regex='^' + stocks[0] + '_Close', axis=1).tail(n).pct_change()
    df_filtered_latent = df.filter(regex='^' + stocks[1] + '_Close', axis=1).tail(n).pct_change()
    df_filtered_cov = df.filter(regex='^' + stocks[2] + '_Close', axis=1).tail(n).pct_change()


    df_filtered_cov = df_filtered_base.iloc[:,0] -df_filtered_cov.iloc[:,0]
    df_filtered_latent = df_filtered_base.iloc[:,0] - df_filtered_latent.iloc[:,0]

    mean_cov = round(df_filtered_cov.mean()*100,4)
    mean_lat = round(df_filtered_latent.mean()*100,4)

    plt.plot(df_filtered.index, df_filtered_cov)
    plt.plot(df_filtered.index, df_filtered_latent)


    # plt.plot(df_plot.index, df_plot['Prediction_Future'], color='r')
    # plt.plot(df_proj.index, df_proj['Prediction'], color='y')
    plt.title('Comparison between covariance and latent features of {v_type} stocks for ({vstock})'.format(vstock = stocks[0], v_type = type))
    plt.legend(
        ['mean difference covariance {mean}:  ±% {v_stock1} - ±% {v_stock2}'.format(mean=mean_cov, v_stock1=stocks[0] , v_stock2=stocks[2]),'mean difference latent feature {mean}: ±% {v_stock1} - ±% {v_stock2}'.format(mean=mean_lat, v_stock1=stocks[0] , v_stock2=stocks[1])], frameon=False)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=16)
    # plt.show()
    path = 'img/ts_{v_d}_{v_type}_difference_close_{v_n}_{v_stocks}.png'.format(v_d = str(d), v_n = str(n), v_stocks=stocks, v_type = type)
    plt.savefig(
        path,
        dpi=300)

    return path





#l_tickers_unique = ['GOOGL',  "CPE","SSRM"]
stocks = ['AAPL',  "GOLD","ALDR"]
n = 200
type = 'least similar'
plot_stock_similarity(df, stocks, n, type)


stocks = ['AAPL',  "AVGO","ADP"]
n = 200
type = 'most similar'
plot_stock_similarity(df, stocks, n, type)





