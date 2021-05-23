import streamlit as st
# To make things easier later, we're also importing numpy and pandas for
# working with sample data.
import numpy as np
import pandas as pd



data = pd.read_csv(r'data/etf_20200217/df_backtest_portfolio.csv', sep = ';')


# LAYING OUT THE TOP SECTION OF THE APP
row1_1, row1_2 = st.beta_columns((2,3))

with row1_1:
    st.title("ETF Portfolio")
    portfolio_type_selected = st.selectbox(
    'Portfolio Type',
     data['portfolio_type'].unique())


with row1_2:
    st.write(
    """
    ##
    text
    text
    """)
# FILTERING DATA FOR THE HISTOGRAM
filtered = data[data['portfolio_type'] == portfolio_type_selected]

st.bar_chart(filtered[['profit','annual_volatility']])

with st.echo():
    def calculation1():
        print('hello')
