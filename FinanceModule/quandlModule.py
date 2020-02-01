class Quandl:
    def __init__(self, apiKey):
        self.apiKey = apiKey
    pass

    def getStockExchangeCodes(self):

        import pandas as pd
        from urllib.request import urlopen
        import os

        # open and save the zip file onto computer
        url = urlopen('https://www.quandl.com/api/v3/databases/FSE/metadata?api_key=' + self.apiKey )
        output = open('zipFile.zip', 'wb')  # note the flag:  "wb"
        output.write(url.read())
        output.close()

        # read the zip file as a pandas dataframe
        df = pd.read_csv('zipFile.zip')  # pandas version 0.18.1 takes zip files
        print(df.head(10))
        # if keeping on disk the zip file is not wanted, then:
        os.remove('zipFile.zip')  # remove the copy of the zipfile on disk
        return df

    def getStockMarketData(self, market, ListQuandleCodes):
        import quandl
        import pandas as pd

        # Downloading the data
        quandl.ApiConfig.api_key = self.apiKey
        n = 1
        for i in ListQuandleCodes:
            i = market + "/" + i
            print(i)
            if n == 1:
                # initialize the dataframe
                df = quandl.get(i)
                df = df.rename(
                    columns={'Open': i + '_Open', 'High': i + '_High', 'Low': i + '_Low', 'Close': i + '_Close',
                             'Volume': i + '_Volume'})


            else:
                df_new = quandl.get(i)
                df_new = df_new.rename(
                    columns={'Open': i + '_Open', 'High': i + '_High', 'Low': i + '_Low', 'Close': i + '_Close',
                             'Volume': i + '_Volume'})
                df = pd.merge(df, df_new, how='outer', left_index=True, right_index=True)

            n = n + 1

        # Drop rows where any value is nan
        # df = df.dropna()

        df['year'] = df.index.year
        df['month'] = df.index.month
        df['weekday'] = df.index.weekday
        return df
