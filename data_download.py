import ticker
from datetime import datetime
import yfinance as yf

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


tickers = ticker.company_list
start_date = '2020-01-01'

for tick in tickers:
    download_stock_data(tick,start_date)
    print(f"data for {tick} is downloaded")