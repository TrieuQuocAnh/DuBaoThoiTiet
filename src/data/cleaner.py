import pandas as pd
import numpy as np

def handle_pressure_outliers(df, column='Pressure (millibars)'):
    """
    Xử lý các giá trị áp suất bằng 0 bằng cách thay thế bằng trung vị (Median).
    """
    # Tính trung vị của các giá trị khác 0
    median_pressure = df[df[column] > 0][column].median()
    
    # Thay thế các giá trị <= 0 bằng median
    df[column] = df[column].apply(lambda x: median_pressure if x <= 0 else x)
    
    return df

def clean_data(df, config):
    """
    Hàm tổng hợp các bước làm sạch dữ liệu.
    """
    # 1. Xóa các cột không cần thiết (Loud Cover, ...)
    df = df.drop(columns=config['preprocessing']['drop_cols'])
    
    # 2. Xử lý Pressure rác
    df = handle_pressure_outliers(df)
    
    # 3. Xử lý Missing Values cho Precip Type
    # Điền giá trị phổ biến nhất (mode)
    if df['Precip Type'].isnull().any():
        mode_val = df['Precip Type'].mode()[0]
        df['Precip Type'] = df['Precip Type'].fillna(mode_val)
        
    return df