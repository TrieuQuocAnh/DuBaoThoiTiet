import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA

def check_stationarity(timeseries):
    """Kiểm tra tính dừng bằng kiểm định Dickey-Fuller"""
    result = adfuller(timeseries.dropna())
    print(f'ADF Statistic: {result[0]:.4f}')
    print(f'p-value: {result[1]:.4f}')
    return result[1] <= 0.05

def plot_acf_pacf(timeseries, lags=40):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 4))
    plot_acf(timeseries, lags=lags, ax=ax1)
    plot_pacf(timeseries, lags=lags, ax=ax2)
    plt.show()