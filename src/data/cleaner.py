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



def handle_missing_values(df):
    """Xử lý giá trị thiếu cho Precip Type"""
    if 'Precip Type' in df.columns:
        # Điền bằng giá trị xuất hiện nhiều nhất (mode)
        mode_val = df['Precip Type'].mode()[0]
        df['Precip Type'] = df['Precip Type'].fillna(mode_val)
    return df

def fix_pressure(df):
    """Thay thế áp suất bằng 0 bằng giá trị trung vị (Median)"""
    median_p = df[df['Pressure (millibars)'] > 0]['Pressure (millibars)'].median()
    df['Pressure (millibars)'] = df['Pressure (millibars)'].replace(0, median_p)
    return df

def discretize_features(df, config):
    """Rời rạc hóa dữ liệu phục vụ Luật kết hợp"""
    conf = config['preprocessing']
    
    # Chuyển số thành nhãn chữ
    df['Temp_Class'] = pd.cut(df['Temperature (C)'], 
                             bins=conf['temp_bins'], labels=conf['temp_labels'])
    
    df['Humidity_Class'] = pd.cut(df['Humidity'], 
                                 bins=conf['humidity_bins'], labels=conf['humidity_labels'])
    
    df['Visibility_Class'] = pd.cut(df['Visibility (km)'], 
                                   bins=conf['visibility_bins'], labels=conf['visibility_labels'])
    return df

def preprocess_pipeline(df, config):
    """Pipeline tổng hợp các bước tiền xử lý"""
    # 1. Xóa cột thừa
    df = df.drop(columns=config['preprocessing']['drop_cols'])
    
    # 2. Xử lý Missing & Outliers (Pressure)
    df = handle_missing_values(df)
    df = fix_pressure(df)
    
    # 3. Rời rạc hóa (Cho bài toán Luật kết hợp)
    df = discretize_features(df, config)
    
    return df