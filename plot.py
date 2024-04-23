import matplotlib.pyplot as plt
import ticker
import pandas as pd
# from sklearn.preprocessing import MinMaxScaler
import numpy as np
# import main

def min_max_scale(array):
    min_val = np.min(array)
    max_val = np.max(array)
    scaled_array = (array - min_val) / (max_val - min_val)
    return scaled_array

tickers = ticker.company_list
#this is for the plotting of LSTM forecasted values and the actual values
for comp in tickers:
    predicted = pd.read_csv(f"C:\\IQBAL\\PhD\\Datasets\\LSTM\\output csv\\seven_day_prediction\\predicted\\{comp}.csv")
    predicted = predicted['6']
    scaled_prediction = min_max_scale(predicted)
    actual = pd.read_csv(f"C:\\IQBAL\\PhD\\Datasets\\LSTM\\output csv\\seven_day_prediction\\actual\\{comp}.csv")
    # actual = actual[6:]
    scaled_actual = min_max_scale(actual)

    #plotting
    plt.clf()
    plt.figure(figsize=(10, 6))
    plt.plot(scaled_prediction, color = 'red', label = 'Predicted Price')
    plt.plot(scaled_actual, color = 'blue', label = 'Actual Price')
    plt.title(f'Actual vs Predicted Price {comp}')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.savefig(f'C:\\IQBAL\\PhD\\Datasets\\LSTM\\output csv\\graph\\prediction_plot\\{comp}.png')
    plt.close()
    print(f"{comp} has plotted")
    