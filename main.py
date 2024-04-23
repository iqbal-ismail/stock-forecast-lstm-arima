import basic_functions as bf
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt
import pandas as pd
import ticker
import evaluation as evl

tickers = ticker.company_list
start_date = '2020-01-01'
mean_sqr_error = {'company':[], 'RMSE': [], 'MAPE' : [], 'R_Squared' : [], 'Adj_R_Squared' : []}

for company in tickers:

    df = pd.read_csv(f"C:\\IQBAL\\PhD\\Datasets\\LSTM\\output csv\\data\\{company}.csv")
    #making the plots

    #plotting the difference between values
    bf.diff_plot(df,company)
    print(f'***** difference plot for {company} has plotted *****')

    #close plot
    bf.close_plot(df,company)
    print(f'***** close price plot for {company} has plotted *****')

    #volume plot
    bf.volume_plot(df,company)
    print(f'***** volume plot for {company} has plotted *****')

    # #moving averages plot
    # bf.ma_plot(df,company)
    # print(f'***** moving averages plot for {company} has plotted *****')


    data = df.filter(['Close'])
    dataset = data.values




    #this is the place from where we can change the split value
    # Get the number of rows to train the model on
    split_size = .90
    training_data_len = int(np.ceil( len(dataset) * split_size ))
    print(f'***** training data for {company} has created *****')

    #scaling the data
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_data = scaler.fit_transform(dataset)
    print(f'***** training data for {company} has scaled to 0-1 range *****')

    #create training dataset
    # Create the scaled training data set
    train_data = scaled_data[0:int(training_data_len), :]
    # Split the data into x_train and y_train data sets
    x_train = []
    y_train = []

    past_points = 60
    future_points = 7
    for i in range(past_points, len(train_data) - future_points + 1):  # Adjusted loop range
        x_train.append(train_data[i-past_points:i, 0])
        y_train.append(train_data[i:i+future_points, 0])
    
    print(f'***** training data for {company} for using in LSTM has created *****')

    # Convert the x_train and y_train to numpy arrays
    x_train, y_train = np.array(x_train), np.array(y_train)

    # Reshape the data
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    y_train = np.reshape(y_train, (y_train.shape[0], future_points, 1))  # Reshaped to include future points

    # Build the LSTM model
    model = Sequential()
    model.add(LSTM(128, return_sequences=True, input_shape= (x_train.shape[1], 1)))
    model.add(LSTM(64, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(7))

    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')


    # Train the model
    model.fit(x_train, y_train, batch_size=1, epochs=1)
    print(f'***** training of {company} has completed *****')

    # Create the testing data set
    # Create a new array containing scaled values from index 1543 to 2002
    test_data = scaled_data[training_data_len - past_points: , :]
    # Create the data sets x_test and y_test
    x_test = []
    y_test = dataset[training_data_len:, :]
    for i in range(past_points, len(test_data)- future_points + 1):
        x_test.append(test_data[i-past_points:i, 0])

    # Convert the data to a numpy array
    x_test = np.array(x_test)

    # Reshape the data
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1 ))

    print(f'***** test data for {company} has created *****')

    # Get the models predicted price values
    predictions = model.predict(x_test)
    # predictions = scaler.inverse_transform(predictions)

    print(f'***** prediction for {company} has completed *****')

    # # Calculate RMSE for each future point
    # rmse_per_point = np.sqrt(np.mean((predictions - y_test[:len(predictions)]) ** 2, axis=0))

    # # Calculate overall RMSE for all future points
    # rmse = np.mean(rmse_per_point)
    rmse = evl.calculate_rmse(predictions,y_test[:len(predictions)])
    #adding rmse to a dictionary
    mean_sqr_error['company'].append(company)
    mean_sqr_error['RMSE'].append(rmse)

    mape = evl.calculate_mape(predictions,y_test[:len(predictions)])
    mean_sqr_error['MAPE'].append(mape)

    r_sqr = evl.calculate_r_squared(predictions,y_test[:len(predictions)])
    mean_sqr_error['R_Squared'].append(r_sqr)

    adj_r_sqr = evl.calculate_adjusted_r_squared(predictions,y_test[:len(predictions)],1)
    mean_sqr_error['Adj_R_Squared'].append(adj_r_sqr)

    # Plot the last seven points of predictions and y_test
    plt.clf()
    plt.figure(figsize=(10, 6))
    plt.plot(y_test[:len(predictions), 0], color='blue', label='Actual Price')
    plt.plot(predictions[:, -1], color='red', label='Predicted Price')
    plt.title('Actual vs Predicted Price')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.savefig(f'C:\\IQBAL\\PhD\\Datasets\\LSTM\\output csv\\graph\\prediction_plot\\{company}.png')

    #saving the predictions
    check = pd.DataFrame(predictions)
    check.to_csv(f'C:\\IQBAL\\PhD\\Datasets\\LSTM\\output csv\\seven_day_prediction\\predicted\\{company}.csv', index = False)

    #saving the actual data
    actual = pd.DataFrame(y_test)
    scaler = MinMaxScaler(feature_range=(0,1))
    actual = scaler.fit_transform(actual)
    actual = pd.DataFrame(actual)
    actual.to_csv(f'C:\\IQBAL\\PhD\\Datasets\\LSTM\\output csv\\seven_day_prediction\\actual\\{company}.csv', index = False)
    print('************************************************')
    print('************************************************')
    print(f'***** all the process {company} has completed *****')
    print('************************************************')
    print('************************************************')


mean_sqr_error = pd.DataFrame(mean_sqr_error)
mean_sqr_error.to_csv('C:\\IQBAL\\PhD\\Datasets\\LSTM\\mean_sqr_error_LSTM.csv', index=False)


