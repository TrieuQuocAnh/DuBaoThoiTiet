import pandas as pd
from sklearn.preprocessing import StandardScaler


def build_features(df, config):
    df = df.copy()
    # Example: add time features if available
    if 'Formatted Date' in df.columns:
        df['year'] = df['Formatted Date'].dt.year
        df['month'] = df['Formatted Date'].dt.month
        df['weekday'] = df['Formatted Date'].dt.dayofweek
        df['hour'] = df['Formatted Date'].dt.hour

    # basic numeric scaling
    numeric_cols = config.get('eda', {}).get('numerical_cols', [])
    numeric_cols = [c for c in numeric_cols if c in df.columns]
    if numeric_cols:
        scaler = StandardScaler()
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    return df



def scale_features(df, features_to_scale):
    """Chuẩn hóa dữ liệu về phân phối chuẩn (Mean=0, Std=1)"""
    scaler = StandardScaler()
    df_scaled = df.copy()
    df_scaled[features_to_scale] = scaler.fit_transform(df[features_to_scale])
    return df_scaled, scaler