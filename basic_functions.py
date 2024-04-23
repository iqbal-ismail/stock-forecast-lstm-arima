import yfinance as yf
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt

def download_stock_data(ticker, start_date="2010-01-01", end_date=None):
    """
    Download historical stock data for a given ticker symbol within a specified date range.
    
    Args:
    ticker (str): The ticker symbol of the stock.
    start_date (str): The start date in the format 'YYYY-MM-DD'. Default is "2020-01-01".
    end_date (str): The end date in the format 'YYYY-MM-DD'. Default is None, which means current date will be used.
    
    Returns:
    pandas.DataFrame: DataFrame containing the historical stock data.
    """
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')

    stock_data = yf.download(ticker, start=start_date, end=end_date)
    filename = f'C:\\IQBAL\\PhD\\Datasets\\LSTM\\output csv\\data\\{ticker}.csv'
    stock_data.to_csv(filename)
    return stock_data


def diff_plot(dataset,ticker):
    plt.clf()
    plt.figure(figsize=(10, 6))
    diff = dataset['Adj Close'].diff()
    up = diff >= 0
    down = diff < 0
    dataset['Adj Close'][up].plot(style='g.', markersize = 1) #the markersize is for the size of plots
    dataset['Adj Close'][down].plot(style='r.',markersize = 1)
    plt.ylabel('Adj Close')
    plt.xlabel(None)
    plt.title(f"Closing Price of {ticker}")
    plt.savefig(f'C:\\IQBAL\\PhD\\Datasets\\LSTM\\output csv\\graph\\closing_price_diff\\{ticker}.png')
    plt.close()

def close_plot(dataset,ticker):
    plt.clf()
    plt.figure(figsize=(10, 6))
    plt.plot(dataset['Adj Close'])
    plt.ylabel('Adj Close')
    plt.xlabel(None)
    plt.title(f"Closing Price of {ticker}")
    plt.savefig(f'C:\\IQBAL\\PhD\\Datasets\\LSTM\\output csv\\graph\\closing_plot\\{ticker}.png')
    plt.close()

#the following function shows the volume of shares traded in each day
def volume_plot(dataset,ticker):
    plt.clf()
    plt.figure(figsize=(10, 6))
    plt.plot(dataset['Volume'])
    plt.ylabel('Volume')
    plt.xlabel(None)
    plt.title(f"Volume of {ticker}")
    plt.savefig(f'C:\\IQBAL\\PhD\\Datasets\\LSTM\\output csv\\graph\\volume_traded\\{ticker}.png')
    plt.close()

#plotting moving average
def ma_plot(dataset,ticker):
    ma_day = [10,20,50]
    for ma in ma_day:
        column_name = f"MA_{ma}_days"
        dataset[column_name] = dataset['Adj Close'].rolling(ma).mean()
        plt.clf()
        plt.figure(figsize=(10, 6))
        plt.plot(dataset['Adj Close'])
        plt.plot(dataset[f"MA for {ma} days"])
        plt.ylabel('Price')
        plt.xlabel(None)
        plt.title(f"{column_name} of {ticker}")
        plt.savefig(f'C:\\IQBAL\\PhD\\Datasets\\LSTM\\output csv\\graph\\moving_average\\{ticker}_{column_name}.png')
        plt.close()
        dataset.to_csv(f"C:\\IQBAL\\PhD\\Datasets\\LSTM\\output csv\\moving_average\\{ticker}.csv")

def close_up_down(dataset, ticker, start = '2018-01-01', end = '2023-12-31'):
    plt.figure(figsize=(16,6))
    plt.title('Close Price History')

    # Initialize the color of the first point to black
    color = 'black'

    # Iterate through each data point and plot it with the appropriate color
    for i in range(1, len(dataset)):
        if dataset['Close'][i] > dataset['Close'][i-1]:
            color = 'green'  # Use green for points that go up
        else:
            color = 'red'    # Use red for points that go down
        plt.plot(dataset.index[i-1:i+1], dataset['Close'][i-1:i+1], color=color)

    plt.xlabel('Date', fontsize=18)
    plt.ylabel('Close Price', fontsize=18)
    plt.xlim(pd.Timestamp(start), pd.Timestamp(end)) #here the date range in which we want the plot 
    plt.savefig(f'C:\\IQBAL\\PhD\\Datasets\\LSTM\\output csv\\graph\\close_up_and_down\\{ticker}.png')
    plt.close()

# ticker = 'INFY.NS'
# start = '2020-01-01'
# end = datetime.now().strftime('%Y-%m-%d')
# data = download_stock_data(ticker,start)
# diff_plot(data,ticker)
# close_plot(data,ticker)
# volume_plot(data, ticker)
# ma_plot(data,ticker)
# close_up_down(data,ticker,start,end)