import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
import numpy as np
from statsmodels.tsa.arima.model import ARIMA


def plot_seasonal_decomposition(decomposition, company):
    plt.clf()

    """
    Plot the decomposed time series components.

    Args:
    - decomposition: statsmodels seasonal decomposition object

    Returns:
    - None
    """
    # Plotting the decomposed time series components
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 10))

    decomposition.trend.plot(ax=ax1)
    ax1.set_title('Trend Component')
    ax1.set_ylabel('Trend')
    decomposition.trend.to_csv(f'C:\\IQBAL\\PhD\\Datasets\\ARIMA\\Decomposition\\Values\\{company}_trend.csv')

    decomposition.seasonal.plot(ax=ax2)
    ax2.set_title('Seasonal Component')
    ax2.set_ylabel('Seasonality')
    decomposition.seasonal.to_csv(f'C:\\IQBAL\\PhD\\Datasets\\ARIMA\\Decomposition\\Values\\{company}_seasonality.csv')

    decomposition.resid.plot(ax=ax3)
    ax3.set_title('Residual Component')
    ax3.set_ylabel('Residuals')
    decomposition.resid.to_csv(f'C:\\IQBAL\\PhD\\Datasets\\ARIMA\\Decomposition\\Values\\{company}_residual.csv')

    plt.tight_layout()
    plt.savefig(f'C:\\IQBAL\\PhD\\Datasets\\ARIMA\\Decomposition\\Graphs\\{company}.png')
    # plt.show()
    print(f'Decomposition Plot for {company} has saved')

# Example usage:
# decomposition = seasonal_decompose(AMD_df['Close'], model='additive', period=5)
# plot_seasonal_decomposition(decomposition)


# ACF plot 
def plot_acf(data, company, lags=40):
    plt.clf()
    """
    Plot the autocorrelation function (ACF) for the given time series data.

    Args:
    - data: time series data
    - title: title for the plot
    - lags: number of lags to include in the plot (default=40)

    Returns:
    - None
    """
    fig, ax = plt.subplots(figsize=(15, 6))

    # ACF plot
    sm.graphics.tsa.plot_acf(data, lags=lags, ax=ax)
    ax.set_title(f'ACF for {company}')
    ax.set_xlabel('Lags')
    ax.set_ylabel('Autocorrelation')

    plt.tight_layout()
    plt.savefig(f'C:\\IQBAL\\PhD\\Datasets\\ARIMA\\ACF Chart\\{company}.png')
    # plt.show()
    print(f'ACF Plot for {company} has saved')


    #for ADF Test
    # Function to perform ADF test and print results

def adf_test(data, stock_name = 'Stock'):
    result = adfuller(data, autolag='AIC')
    print(f"ADF Statistic for {stock_name}: {result[0]}")
    print(f"P-value: {result[1]}")
    print("Critical Values:")
    for key, value in result[4].items():
        print(f"\t{key}: {value}")
    print("\n")

def first_dif_plot(data, company):
    plt.clf()
    plt.figure(figsize=(12, 6))
    plt.plot(data)
    plt.title(f'{company} First Difference Close Price', fontsize=12)
    plt.xlabel('Observation', fontsize=12)
    plt.ylabel('First Difference', fontsize=12)
    plt.savefig(f'C:\\IQBAL\\PhD\\Datasets\\ARIMA\\First Difference Plot\\{company}.png')
    # plt.show()
    print(f'First Difference Plot for {company} has saved')


#Function to search for best ARIMA Model
def searchARIMA(data, exog, max_p, max_q):
    best_aic = np.inf
    best_bic = np.inf
    best_order = None

    aic_values = []
    bic_values = []

    for i in range(max_p):
        for j in range(max_q):
            try:
                if len(exog) == len(data):
                    model = ARIMA(data, order=(i, 0, j), exog=exog, trend='n')
                else:
                    model = ARIMA(data, order=(i, 0, j), trend='n')
                res = model.fit()
                aic = res.aic
                bic = res.bic

                # Storing AIC and BIC values
                aic_values.append((i, j, aic))
                bic_values.append((i, j, bic))

                print(f'p: {i}, q: {j}, aic: {aic}, bic: {bic}')

                # Update best AIC and BIC if the current model is better
                if aic < best_aic:
                    best_aic = aic
                    best_order_aic = (i, 0, j)
                if bic < best_bic:
                    best_bic = bic
                    best_order_bic = (i, 0, j)

            except Exception as e:
                print(f'Error with p: {i}, q: {j} - {e}')
                continue

    # Select the best model based on AIC or BIC
    best_order = best_order_aic if best_aic < best_bic else best_order_bic
    print("Best model based on AIC:", best_order_aic)
    print("Best model based on BIC:", best_order_bic)
    print("Best model overall:", best_order)

    return best_order, aic_values, bic_values

def plot_combine_nc(actual, predicted, company):
    plt.clf()
    plt.plot(actual, label='Actual Close Prices')
    plt.plot(predicted, label = 'Predicted Prices')
    plt.title(f'{company} Actual V/s Predicted', fontsize=12)
    plt.xlabel('Time', fontsize=12)
    plt.ylabel('Price', fontsize=12)
    plt.legend()
    plt.savefig(f'C:\\IQBAL\\PhD\\Datasets\\ARIMA\\Predictions\\Graphs\\Non Cumulated\\{company}.png')
    print(f'Non cumulated prediction plot for {company} has saved')

def plot_combine_cumulated(actual, predicted, company):
    plt.clf()
    plt.plot(actual, label='Actual Close Prices')
    plt.plot(predicted, label = 'Predicted Prices')
    plt.title(f'{company} Actual V/s Predicted', fontsize=12)
    plt.xlabel('Time', fontsize=12)
    plt.ylabel('Price', fontsize=12)
    plt.legend()
    plt.savefig(f'C:\\IQBAL\\PhD\\Datasets\\ARIMA\\Predictions\\Graphs\\Cumulated\\{company}.png')
    print(f'Cumulated prediction plot for {company} has saved')


