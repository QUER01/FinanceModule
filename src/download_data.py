from yahooquery import Ticker
import os
import pandas as pd
import numpy as np
from datetime import datetime
import json
from pandas.io.json import json_normalize
from pytrends.request import TrendReq

'''


EUNL.DE    : iShares Core MSCI World UCITS ETF USD (Acc) (EUNL.DE)
USPY.DE    : Legal & General UCITS ETF Plc - L&G Cyber Security UCITS ETF (USPY.DE)
E908.DE     : Lyxor 1 TecDAX UCITS ETF (E908.DE)
X010.DE    : Lyxor MSCI World (LUX) UCITS ETF (X010.DE)
'''

print('START')
path = os.environ['MYPATH']
#path = r"\\DISKSTATION\docker\python_stock_data"
dstprefix = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
ticker_list_personal = ['E908.DE']
ticker_list_etfs = pd.read_csv(r'etf_yahoo_tickers.csv', sep=';', usecols=[0, 2])
ticker_list_etfs = ticker_list_etfs[ticker_list_etfs['Exchange'] == 'GER']

ticker_list = ticker_list_etfs['Ticker'].values
ticker_list = np.append(ticker_list, ticker_list_personal)
number_of_items = 10
number_of_chunks = len(ticker_list) / number_of_items

ticker_list_chunks = np.array_split(ticker_list, number_of_chunks)

df_metadata_all = pd.DataFrame()
df_historic_all = pd.DataFrame()
for chunk in ticker_list_chunks:
    print(chunk)

    tickers = Ticker(chunk, formatted=True, asynchronous=True)

    metadata_dict = tickers.all_modules
    # iterate over all tickers

    for ticker in metadata_dict:
        print(ticker)
        df_metadata_ticker = pd.DataFrame()

        try:
            # iterate over all modules
            ticker_data = metadata_dict[ticker]

            for module in ticker_data:

                # print(module)
                # print(module)
                try:
                    df_metadata = pd.DataFrame.from_dict(metadata_dict[ticker][module])

                    if df_metadata.__len__() > 1:

                        columns_new = []
                        data_new = []

                        indices = df_metadata.index
                        columns = df_metadata.columns
                        i = 0
                        for df_metadata_row in df_metadata.iterrows():
                            # print(indices[i])
                            j = 0
                            for column in columns:
                                columns_new.append(indices[i] + '.' + columns[j])
                                data_new.append(df_metadata[column].iloc[i])

                                # print('{v_j} : {v_column}'.format(v_j=j, v_column=column))

                                # df_metadata_new[columns_new[j]] = df_metadata[column].iloc[i]
                                j = j + 1
                            i = i + 1

                        df_metadata_new = pd.DataFrame(data_new)
                        df_metadata_new = df_metadata_new.transpose()
                        df_metadata_new.columns = columns_new

                    else:
                        df_metadata_new = df_metadata

                    df_metadata_ticker = df_metadata_new.join(df_metadata_ticker, lsuffix=module)

                except:
                    df_metadata_new = pd.json_normalize(metadata_dict[ticker][module])
                    # print(df_metadata)

                    df_metadata_ticker = df_metadata_new.join(df_metadata_ticker, lsuffix=module)

            df_metadata_ticker['insert_ts'] = dstprefix
            df_metadata_ticker['ticker'] = ticker

            # now get Google trends
            '''
            try:
                pytrend = TrendReq(hl='en-US', tz=360
                    ,retries=2
                    ,backoff_factor=1)
                keywords = [df_metadata_ticker['shortName'].iloc[0]]

                pytrend.build_payload(
                    kw_list=keywords,
                    timeframe='now 1-d'
                )
                data = pytrend.interest_over_time()
                df_metadata_ticker['Google_trends'] = str(data.to_dict())

            except:
                df_metadata_ticker['Google_trends'] = {}
            '''
            df_metadata_all = df_metadata_all.append(df_metadata_ticker)

        except Exception as e:
            print('FAIL: Could not load data for ticker {v_ticker}'.format(v_ticker=ticker))

    df = tickers.history(period='5d', adj_ohlc=True)
    # df = tickers.history(period="max", adj_ohlc=True)
    print('Download data for {v_ticker}'.format(v_ticker=str(ticker)))
    print('length: ' + str(df.__len__()))
    try:
        df_historic_all = df_historic_all.append(df)
    except:
        print('FAIL: Could not load {v_ticker}'.format(v_ticker=str(ticker)))



# fine tuning
df_riskOverviewStatistics_all = pd.DataFrame()
l = 0
l_max = df_metadata_all.__len__()
for l in range(l_max):

    try:

        df_riskOverviewStatistics = pd.json_normalize(df_metadata_all['riskStatistics.riskOverviewStatistics'].str[0].iloc[l])
        df_riskOverviewStatistics['ticker'] = df_metadata_all['ticker'].iloc[l]
        df_riskOverviewStatistics['insert_ts'] = df_metadata_all['insert_ts'].iloc[l]

        df_riskOverviewStatistics_all = df_riskOverviewStatistics_all.append(df_riskOverviewStatistics)

    except Exception as e:
       print(e)


df_metadata_all = df_metadata_all.merge(df_riskOverviewStatistics_all, on =['ticker', 'insert_ts'], how='left', suffixes=["","riskStatistics.riskOverviewStatistics"])


# ---------------
# HOLDINGS
# ---------------


# fine tuning
df_holdings_all = pd.DataFrame()
l = 0
l_max = df_metadata_all.__len__()
for l in range(l_max):

    try:

        df_holdings = pd.json_normalize(df_metadata_all['holdings'].iloc[l])
        df_holdings['ticker'] = df_metadata_all['ticker'].iloc[l]
        df_holdings['insert_ts'] = df_metadata_all['insert_ts'].iloc[l]

        df_holdings_all = df_holdings_all.append(df_holdings)

    except Exception as e:
       print(e)








# ---------------
# bondRatings
# ---------------


# fine tuning
df_bondRatings_all = pd.DataFrame()
l = 0
l_max = df_metadata_all.__len__()
for l in range(l_max):

    try:

        df_bondRatings = pd.json_normalize(df_metadata_all['bondRatings'].str[0].iloc[l])
        df_bondRatings['ticker'] = df_metadata_all['ticker'].iloc[l]
        df_bondRatings['insert_ts'] = df_metadata_all['insert_ts'].iloc[l]

        df_bondRatings_all = df_bondRatings_all.append(df_bondRatings)

    except Exception as e:
       print(e)


df_metadata_all = df_metadata_all.merge(df_bondRatings_all, on =['ticker', 'insert_ts'], how='left', suffixes=["","bondRatings"])




# ---------------
# sectorWeightings
# ---------------


# fine tuning
df_sectorWeightings_all = pd.DataFrame()
l = 0
l_max = df_metadata_all.__len__()
for l in range(l_max):

    try:

        df_sectorWeightings = pd.json_normalize(df_metadata_all['sectorWeightings'].str[0].iloc[l])
        df_sectorWeightings['ticker'] = df_metadata_all['ticker'].iloc[l]
        df_sectorWeightings['insert_ts'] = df_metadata_all['insert_ts'].iloc[l]

        df_sectorWeightings_all = df_sectorWeightings_all.append(df_sectorWeightings)

    except Exception as e:
       print(e)


df_metadata_all = df_metadata_all.merge(df_sectorWeightings_all, on =['ticker', 'insert_ts'], how='left', suffixes=["","sectorWeightings"])




df_holdings_all.to_csv('.' + path + '/' + dstprefix + '_metadata_holdings.csv.csv', sep=';')
df_historic_all.to_csv('.' + path + '/' + dstprefix + '_5d_data.csv', sep=';')
df_metadata_all = df_metadata_all.set_index(['ticker', 'insert_ts'])
df_metadata_all.to_csv('.' + path + '/' + dstprefix + '_metadata.csv', sep=';')




