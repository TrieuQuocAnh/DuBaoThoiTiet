import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA

def train_holt_winters(train_data, seasonal_period=365):
    """Huấn luyện mô hình Holt-Winters"""
    model = ExponentialSmoothing(
        train_data, 
        seasonal='add', 
        seasonal_periods=seasonal_period
    ).fit()
    return model

def train_arima(train_data, order=(1, 1, 1)):
    """Huấn luyện mô hình ARIMA"""
    model = ARIMA(train_data, order=order).fit()
    return model

def run_arima_forecast(train, test_len, order=(1,1,1)):
    model = ARIMA(train, order=order).fit()
    return model.forecast(steps=test_len), model

def run_holt_winters_forecast(train, test_len, seasonal_periods=365):
    # Holt-Winters với thành phần Mùa vụ (Additive)
    model = ExponentialSmoothing(
        train, 
        seasonal='add', 
        seasonal_periods=seasonal_periods
    ).fit()
    return model.forecast(test_len), model