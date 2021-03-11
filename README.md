# Project Name: Stock Price Prediction and Portfolio Optimization Using Recurrent Neural Networks and Autoencoders
This project has been presented at the Predictive Analytics World Conference 2020 - Berlin in the finance deep dive track.

#### -- Project Status: [Active]

## Project Intro/Objective
This project shows how to improve portfolio optimization techniques to form an optimal stock portfolio based on stock returns and stock risks by using non-linear methods. I am extending and cleaning the inputs (returns and risk) by using deep learning methods. 



### Methods Used
* Recurrent neural Networks with LSTM cells
* Autoencoders for subset creation
* Autoencoders used for cleaning the sample covariance matrix

### Technologies
* Python
* Docker

## Project Description
Financial time series forecasting is a challenging problem. Deep learning approaches, such as recurrent neural networks (RNNs), have proven powerful in modelling the volatility of financial stocks and other assets, as they are able to capture non-linearities in sequential data. Recent studies have shown that RNNs have surpassed well-known autoregressive forecasting models (Siami-Namini, 2018). Besides forecasting the next value of a stock, creating an optimal portfolio is equally important. Deep portfolio theory (Heaton et.al.,2018) uses autoencoders to model the non-linearity of the time series to accurately predict returns. Within this session, Julian will extend this approach by first performing a 10-day ahead forecast and then train an autoencoder model to construct an ideal portfolio that incorporates previous and future stock market information. You will be provided with a theoretical understanding of how RNNs and autoencoders work and how to apply them on multivariate timeseries forecasting and portfolio optimization problems. The new model  is then applied on financial stock data and compared to traditional portfolio optimization methods. Finally, the session will end with presenting an overview of key challenges and current research topics within that field.



## Getting Started

1. Clone this repo (for help see this [tutorial](https://help.github.com/articles/cloning-a-repository/)).
2. Raw Data is being kept here: 
    * https://www.kaggle.com/ehallmar/daily-historical-stock-prices-1970-2018/version/1

    
3. You can run this repo using docker or your own python virtual environment. A neccessary step is to install the python libraries listed in the requirements.txt file.

    ```cmd
    docker-compose up
    ```
    
    ```cmd
    pip install -r requirements.txt
    ```
    
     



## Contact
* Feel free to contact me directly on github or on linkedin: 


