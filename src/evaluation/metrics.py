from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import numpy as np

def evaluate_model(model, df, config):
    target_col = config.get('eda', {}).get('target_col', 'target')
    if target_col not in df.columns:
        raise ValueError(f'Target column {target_col} not found in df')

    X = df.drop(columns=[target_col], errors='ignore').select_dtypes(include=['number']).fillna(0)
    y = df[target_col]

    if y.dtype == 'object' or y.dtype.name == 'category':
        y = LabelEncoder().fit_transform(y.fillna('missing'))

    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    y_pred = model.predict(X_test)

    metrics = {
        'rmse': mean_squared_error(y_test, y_pred, squared=False),
        'mae': mean_absolute_error(y_test, y_pred),
        'r2': r2_score(y_test, y_pred)
    }

    print('Evaluation results:', metrics)
    return metrics


def calculate_ts_metrics(y_true, y_pred):
    """Tính toán MAE và RMSE cho dự báo"""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return {"MAE": mae, "RMSE": rmse}

def detect_ts_outliers(y_true, y_pred, threshold_sigma=2):
    """Phát hiện ngoại lai dựa trên thặng dư (Residuals)"""
    residuals = y_true - y_pred
    std_res = residuals.std()
    outliers = residuals[np.abs(residuals) > (threshold_sigma * std_res)]
    return residuals, outliers



def evaluate_forecast(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return {"MAE": mae, "RMSE": rmse}

def analyze_residuals(y_true, y_pred):
    residuals = y_true - y_pred
    return residuals

def detect_forecast_outliers(residuals, threshold_sigma=2):
    std_res = residuals.std()
    outliers = residuals[np.abs(residuals) > (threshold_sigma * std_res)]
    return outliers