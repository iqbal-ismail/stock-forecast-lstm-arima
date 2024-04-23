import numpy as np
import ticker
import pandas as pd
import matplotlib.pyplot as plt
import evaluation as evl


def min_max_scale(array):
    min_val = np.min(array)
    max_val = np.max(array)
    scaled_array = (array - min_val) / (max_val - min_val)
    return scaled_array

mean_sqr_error_ARIMA = {'company':[], 'RMSE': [], 'MAPE' : [], 'R_Squared' : [], 'Adj_R_Squared' : []}
mean_sqr_error_LSTM = {'company':[], 'RMSE': [], 'MAPE' : [], 'R_Squared' : [], 'Adj_R_Squared' : []}

tickers = ticker.company_list
for company in tickers:
    arima_data = pd.read_csv(f'C:\\IQBAL\\PhD\\Datasets\\ARIMA\\Predictions\\Values\\{company}.csv')
    actual_data = pd.read_csv(f'C:\\IQBAL\\PhD\\Datasets\\LSTM\\output csv\\seven_day_prediction\\actual\\{company}.csv')
    lstm_data = pd.read_csv(f'C:\\IQBAL\\PhD\\Datasets\\LSTM\\output csv\\seven_day_prediction\\predicted\\{company}.csv')

    arima_values = arima_data['6']
    actual_values = actual_data['0']
    lstm_values = lstm_data['6']

    arima_values_scaled = min_max_scale(arima_values)
    actual_values_scaled = min_max_scale(actual_values)
    lstm_values_scaled = min_max_scale(lstm_values)

    plt.figure(figsize=(10, 6))
    plt.clf()
    plt.plot(arima_values_scaled, label = 'ARIMA Prediction')
    plt.plot(actual_values_scaled, label = 'Actual Values')
    plt.plot(lstm_values_scaled, label = 'LSTM Prediction')
    plt.legend()
    plt.title(f'Combined Plot for {company}')
    plt.savefig(f'C:\\IQBAL\\PhD\\Datasets\\comparison\\graph\\{company}.png')
    
    print(f'Comparison plot for {company} has completed')

    mean_sqr_error_ARIMA['company'].append(company)
    rmse = evl.calculate_rmse(arima_values_scaled,actual_values_scaled)
    mean_sqr_error_ARIMA['RMSE'].append(rmse)

    mean_sqr_error_LSTM['company'].append(company)
    rmse = evl.calculate_rmse(lstm_values_scaled,actual_values_scaled)
    mean_sqr_error_LSTM['RMSE'].append(rmse)

    adj_r_sqr = evl.calculate_adjusted_r_squared(arima_values_scaled,actual_values_scaled)
    mean_sqr_error_ARIMA['Adj_R_Squared'].append(adj_r_sqr)

    adj_r_sqr = evl.calculate_adjusted_r_squared(lstm_values_scaled,actual_values_scaled)
    mean_sqr_error_LSTM['Adj_R_Squared'].append(adj_r_sqr)

    mape = evl.calculate_mape(arima_values_scaled,actual_values_scaled)
    mean_sqr_error_ARIMA['MAPE'].append(mape)

    mape = evl.calculate_mape(lstm_values_scaled,actual_values_scaled)
    mean_sqr_error_LSTM['MAPE'].append(mape)

    r_sqr = evl.calculate_r_squared(arima_values_scaled,actual_values_scaled)
    mean_sqr_error_ARIMA['R_Squared'].append(r_sqr)

    r_sqr = evl.calculate_r_squared(lstm_values_scaled,actual_values_scaled)
    mean_sqr_error_LSTM['R_Squared'].append(r_sqr)



mean_sqr_error_ARIMA = pd.DataFrame(mean_sqr_error_ARIMA)
mean_sqr_error_ARIMA.to_csv('C:\\IQBAL\\PhD\\Datasets\\ARIMA\\mean_sqr_error_ARIMA.csv', index=False)

mean_sqr_error_LSTM = pd.DataFrame(mean_sqr_error_LSTM)
mean_sqr_error_LSTM.to_csv('C:\\IQBAL\\PhD\\Datasets\\LSTM\\mean_sqr_error_LSTM.csv', index=False)

    


