# **Stock Market Forecasting with ARIMA and LSTM**

## **Introduction**

Welcome to stock market forecasting project! In this project, we'll explore two powerful techniques for predicting stock prices: **ARIMA (AutoRegressive Integrated Moving Average)** and **LSTM (Long Short-Term Memory)** networks

**ARIMA Model:** ARIMA is a time series forecasting method that captures the linear relationships and patterns in historical data to make predictions about future values. It's great for modeling data with trends and seasonality.
for more details watch [Here](https://www.youtube.com/watch?v=4Fiz3dQM_i8)

**LSTM Architecture:** LSTM is a type of recurrent neural network (RNN) that excels at capturing long-term dependencies and patterns in sequential data, making it well-suited for time series forecasting tasks like stock market prediction.
for more details watch [Here](https://www.youtube.com/watch?v=b61DPVFX03I)

## **How it Works**

**ARIMA:** The ARIMA model analyzes historical stock price data, identifying patterns such as trends and seasonality. It then uses this information to make predictions about future price movements. ARIMA is based on the idea that future values can be estimated by combining past values and past forecast errors.

**LSTM:** LSTM networks use a special architecture that allows them to remember information over long periods of time. This makes them effective at capturing complex patterns in sequential data, such as stock prices. LSTM models are trained on historical stock price data and learn to predict future price movements based on past trends and patterns.

## **Get Started**

To get started with our stock market forecasting project, simply follow the instructions in the README file. We provide step-by-step guidance on how to use both the ARIMA model and LSTM architecture to forecast stock prices.

Let's start predicting the future of the stock market together!

## **Flow of Operations**

This project uses LSTM and ARIMA model for stock market forecasting which can be helpful for investors, traders, academicians and researchers. the file named 'ticker.py' contains the ticker symbol for companies which are used for downloading the historical data from yahoo finance. For now I have added some companies that are listed in the National Stock Exchange (NSE) in India. you can add the companies of your preference. you can add them as lists and the name of the list can be assigned to a variable named 'company_list' which will be used all around the project. so, i recommend not to change the name of the variable. The flow of execution is as follows

1. Add the stock ticker to the variable **'comapny_list'**. The project uses Yahoo finance library for fetching historical data, so for Indian companies a suffix of '.NS' is required with each ticker symbol to make them work. so you may have to make adjustments in the ticker.

2. Once you have added the ticker symbols for companies, you can run the project with their default values just by running the file **'full_run.py'**. This will take the ticker symbols specified in the variable **'company_list'** and download the historical data. then the file **'main.py'** will be executed which contains the code for forecasting using LSTM after that the file **'plot.py'** will be executed which will produce the plottings for the prediction. this helps to visualise the prediction with actual data. after that, the file **'ARIMA.py'** will be executed. this contains the time series decomposition and the visual representation of the decomposition. the project was coded in a way all the output values will be stored as csv files and the diagrams will be saved as png files. if you want to view the plots live, you can add plt.show() at the end of each visualisation function.

3. In the files **'main.py'** and **'ARIMA.py'**, there is a variable named 'split_size' which is defined in the beginning, this is the ratio at which the data has to be split into training data and testing data. make sure that you are giving same value in both the cases, only then the performance of both can be compared.

4. The path for storing the results must be changed according to your preference.
