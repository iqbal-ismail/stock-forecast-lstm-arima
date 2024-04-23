import numpy as np

def calculate_rmse(predictions, actual):
    """
    Calculate Root Mean Squared Error (RMSE) between predictions and actual values.
    
    Args:
    predictions (numpy.ndarray): Predicted values.
    actual (numpy.ndarray): Actual values.
    
    Returns:
    float: RMSE value.
    """
    rmse_per_point = np.sqrt(np.mean((predictions - actual) ** 2, axis=0))
    rmse = np.mean(rmse_per_point)
    return rmse

def calculate_mape(predictions, actual):
    """
    Calculate Mean Absolute Percentage Error (MAPE) between predictions and actual values.
    
    Args:
    predictions (numpy.ndarray): Predicted values.
    actual (numpy.ndarray): Actual values.
    
    Returns:
    float: MAPE value.
    """
    mape = np.mean(np.abs((actual - predictions) / actual)) * 100
    return mape

def calculate_r_squared(predictions, actual):
    """
    Calculate R-squared value between predictions and actual values.
    
    Args:
    predictions (numpy.ndarray): Predicted values.
    actual (numpy.ndarray): Actual values.
    
    Returns:
    float: R-squared value.
    """
    ss_res = np.sum((actual - predictions) ** 2)
    ss_tot = np.sum((actual - np.mean(actual)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)
    return r_squared

def calculate_adjusted_r_squared(predictions, actual, num_features = 1):
    """
    Calculate Adjusted R-squared value between predictions and actual values.
    
    Args:
    predictions (numpy.ndarray): Predicted values.
    actual (numpy.ndarray): Actual values.
    num_features (int): Number of features in the model.
    
    Returns:
    float: Adjusted R-squared value.
    """
    n = len(actual)
    r_squared = calculate_r_squared(predictions, actual)
    adjusted_r_squared = 1 - (1 - r_squared) * ((n - 1) / (n - num_features - 1))
    return adjusted_r_squared

