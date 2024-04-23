import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
import ticker
import ARIMA_functions as af
from statsmodels.tsa.seasonal import seasonal_decompose
import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)

split_size = 0.90
tickers = ticker.company_list

for company in tickers:
    data = pd.read_csv(f"C:\\IQBAL\\PhD\\Datasets\\LSTM\\output csv\\data\\{company}.csv")
    data['Date'] = pd.to_datetime(data['Date'])

    #setting the date as the index of the dataset
    data = data.set_index('Date')

    #taking the close column of the dataset and converting it to pandas dataframe
    df = data['Close']
    df = pd.DataFrame(df)

    #for time series decomposition
    decomposition = seasonal_decompose(df, model='additive', period=5)
    af.plot_seasonal_decomposition(decomposition, company)
    print(f'Decomposition for {company} is complete')

    #for ACF Chart
    af.plot_acf(df,company,40)
    print(f'ACF for {company} is saved')

    #taking first difference of the data
    df['Close_price_first_difference'] = df['Close'] - df['Close'].shift(1)
    stock_first_diff = df['Close_price_first_difference'].dropna()

    # Resetting the index and then selecting the 'Close_price_first_difference' column
    first_diff_arr = np.array(df.reset_index()['Close_price_first_difference'].dropna())

    #plotting the first difference data
    af.first_dif_plot(first_diff_arr,company)
    print(f'First difference plot for {company} has saved')


    #train-test split
    training_length = int(len(first_diff_arr) * split_size)
    train_diff = first_diff_arr[:training_length]
    test_diff = first_diff_arr[training_length:]

    #for selecting the best ARIMA p,d,q values
    n = len(train_diff)
    exog = np.ones(n)
    best_order, aic_values, bic_values = af.searchARIMA(train_diff, exog, max_p=6, max_q=4)
    print(f'Optimal ARIMA model for {company} is calculated')

    #Implementing the ARIMA Model
    model = ARIMA(train_diff, order=best_order)
    # Fit the model
    ARIMA_res = model.fit()

    # Forecasting (the number of steps to forecast can be adjusted)
    forecast = ARIMA_res.forecast(steps=7)
    print(f"Forecast: {forecast}")

    #creating a copy of train_diff that needs to be used for continuous analysis
    train_df = train_diff

    # Initialize an empty DataFrame to store predictions
    predictions_df = pd.DataFrame()
    # Loop through test_diff
    for i in range(len(test_diff)):
        model = ARIMA(train_df, order=best_order)
        ARIMA_res = model.fit()
        forecast = ARIMA_res.forecast(steps=7)
        forecast_df = pd.DataFrame(forecast)
        forecast_df = forecast_df.transpose()
        predictions_df = pd.concat([predictions_df,forecast_df], axis = 0)
        train_df = np.append(train_df,test_diff[i])
        print(f'Iteration {i} for {company} is complete')
    
    predictions_df.to_csv(f'C:\\IQBAL\\PhD\\Datasets\\ARIMA\\Predictions\\Values\\{company}.csv', index=False)
    print(f'Predictions for {company} is saved')

    """
    The prediction has been saved, now the last column of the dataset has to be called, which is the seventh day
    and the prediction has to be plotted with the actual values. to do that, the predicted returns has to be added 
    with a base known price which is a last known price.
    """

    # seven_day = predictions_df.iloc[:,-1]
    seven_day = pd.read_csv(f'C:\\IQBAL\\PhD\\Datasets\\ARIMA\\Predictions\\Values\\{company}.csv')
    seven_day = seven_day['6']
    seven_day = pd.DataFrame(seven_day)

    af.plot_combine_nc(test_diff,seven_day,company)

    #finding the last known price
    mark_index = round(int(len(df))*split_size)
    mark_index

    last_known_price = df['Close'].iloc[mark_index]

    # Add the last known actual price to the cumulative returns to get the forecasted prices
    actual_prices = last_known_price + test_diff.cumsum()
    predicted_prices = last_known_price + seven_day.cumsum()

    actual_prices = pd.DataFrame(actual_prices)
    predicted_prices = pd.DataFrame(predicted_prices)

    #plotting
    af.plot_combine_cumulated(actual_prices,predicted_prices,company)


