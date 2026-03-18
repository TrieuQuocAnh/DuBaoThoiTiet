import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, f1_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import LabelEncoder
import pandas as pd

def prepare_classification_data(df, config):
    features = config['classification']['features']
    target = config['classification']['target']
    
    # 1. Lọc bớt các nhãn Summary quá hiếm (dưới 100 mẫu) để tránh nhiễu
    threshold = 100
    counts = df[target].value_counts()
    valid_summaries = counts[counts >= threshold].index
    df_filtered = df[df[target].isin(valid_summaries)].copy()
    
    # 2. Mã hóa các biến phân loại đầu vào (như Precip Type)
    le_precip = LabelEncoder()
    df_filtered['Precip Type'] = le_precip.fit_transform(df_filtered['Precip Type'])
    
    # 3. Mã hóa nhãn Summary
    le_target = LabelEncoder()
    y = le_target.fit_transform(df_filtered[target])
    X = df_filtered[features]
    
    return train_test_split(X, y, test_size=config['classification']['test_size'], 
                            random_state=config['classification']['random_state']), le_target

def train_classifiers(X_train, y_train, config):
    """Huấn luyện đồng thời RF và XGBoost"""
    rf = RandomForestClassifier(**config['classification']['models']['rf_params'])
    rf.fit(X_train, y_train)
    
    xgb = XGBClassifier(**config['classification']['models']['xgb_params'])
    xgb.fit(X_train, y_train)
    
    return rf, xgb



from sklearn.preprocessing import LabelEncoder
import pandas as pd

def prepare_classification_data(df, config):
    features = config['classification']['features']
    target = config['classification']['target']
    
    # 1. Lọc bớt các nhãn Summary quá hiếm (dưới 100 mẫu) để tránh nhiễu
    threshold = 100
    counts = df[target].value_counts()
    valid_summaries = counts[counts >= threshold].index
    df_filtered = df[df[target].isin(valid_summaries)].copy()
    
    # 2. Mã hóa các biến phân loại đầu vào (như Precip Type)
    le_precip = LabelEncoder()
    df_filtered['Precip Type'] = le_precip.fit_transform(df_filtered['Precip Type'])
    
    # 3. Mã hóa nhãn Summary
    le_target = LabelEncoder()
    y = le_target.fit_transform(df_filtered[target])
    X = df_filtered[features]
    
    return train_test_split(X, y, test_size=config['classification']['test_size'], 
                            random_state=config['classification']['random_state']), le_target


def train_classifiers(X_train, y_train, config, num_classes):
    """
    Huấn luyện đồng thời RF và XGBoost với cấu hình đa lớp.
    """
    # 1. Cấu hình Random Forest
    rf = RandomForestClassifier(
        **config['classification']['models']['rf_params'],
        random_state=config['classification']['random_state']
    )
    rf.fit(X_train, y_train)
    
    # 2. Cấu hình XGBoost cho Multi-class
    xgb = XGBClassifier(
        **config['classification']['models']['xgb_params'],
        objective='multi:softprob', # Trả về xác suất cho từng lớp
        num_class=num_classes,      # Số lượng nhãn Summary duy nhất
        random_state=config['classification']['random_state'],
        use_label_encoder=False,    # Đã dùng LabelEncoder thủ công
        eval_metric='mlogloss'      # Tránh cảnh báo logloss
    )
    xgb.fit(X_train, y_train)
    
    return rf, xgb