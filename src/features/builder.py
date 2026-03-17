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
