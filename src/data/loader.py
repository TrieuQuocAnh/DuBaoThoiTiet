import pandas as pd

def load_data(path):
    df = pd.read_csv(path)
    # Chuyển đổi datetime ngay khi load
    df['Formatted Date'] = pd.to_datetime(df['Formatted Date'], utc=True)
    return df

def get_season(month):
    if month in [12, 1, 2]: return 'Winter'
    elif month in [3, 4, 5]: return 'Spring'
    elif month in [6, 7, 8]: return 'Summer'
    else: return 'Autumn'

def add_time_features(df):
    df['Year'] = df['Formatted Date'].dt.year
    df['Month'] = df['Formatted Date'].dt.month
    df['Day'] = df['Formatted Date'].dt.day
    df['Season'] = df['Month'].apply(get_season)
    return df

def check_invalid_pressure(df):
    invalid_count = len(df[df['Pressure (millibars)'] <= 0])
    print(f"Số lượng bản ghi có áp suất không hợp lệ (<= 0): {invalid_count}")