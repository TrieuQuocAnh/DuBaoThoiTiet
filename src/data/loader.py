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

import pandas as pd

def prepare_ts_index(df, date_col='Formatted Date'):
    """
    Chuyển đổi DataFrame sang dạng TimeSeries Index sạch.
    Sửa lỗi AttributeError bằng cách kiểm tra kiểu Index chuẩn.
    """
    ts_df = df.copy()
    
    # 1. Đảm bảo date_col trở thành Index và có kiểu Datetime
    if date_col in ts_df.columns:
        ts_df[date_col] = pd.to_datetime(ts_df[date_col], utc=True)
        ts_df = ts_df.set_index(date_col)
    else:
        # Nếu đã là Index, đảm bảo nó là DatetimeIndex
        if not isinstance(ts_df.index, pd.DatetimeIndex):
            ts_df.index = pd.to_datetime(ts_df.index, utc=True)

    # 2. Sắp xếp theo thời gian
    ts_df = ts_df.sort_index()
    
    # 3. Loại bỏ múi giờ (Timezone) một cách an toàn
    # Kiểm tra xem Index có thông tin múi giờ không
    if hasattr(ts_df.index, 'tz') and ts_df.index.tz is not None:
        ts_df.index = ts_df.index.tz_localize(None)
    elif str(ts_df.index.dtype).contains('datetime64[ns, '): # Cách kiểm tra phụ cho phiên bản cũ
        ts_df.index = ts_df.index.tz_localize(None)
        
    return ts_df

def resample_weather_data(ts_df, target_col, rule='D'):
    """
    Hàm mới: Resample dữ liệu theo chu kỳ (mặc định là Ngày - 'D').
    """
    return ts_df[target_col].resample(rule).mean().ffill()