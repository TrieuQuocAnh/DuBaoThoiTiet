import joblib
import os
import pandas as pd

def save_model_artifact(model, model_name, config, base_path="../"):
    """Lưu các mô hình ML (.pkl)"""
    model_dir = os.path.join(base_path, config['outputs']['models_dir'])
    os.makedirs(model_dir, exist_ok=True)
    
    save_path = os.path.join(model_dir, f"{model_name}.pkl")
    joblib.dump(model, save_path)
    print(f"📦 Đã lưu mô hình: {save_path}")

def save_rules_artifact(rules_dict, config, base_path="../"):
    """Lưu kết quả luật kết hợp (.csv)"""
    table_dir = os.path.join(base_path, config['outputs']['mining_dir'])
    os.makedirs(table_dir, exist_ok=True)
    
    # Gộp các mùa thành 1 file để dễ quản lý
    all_rules = []
    for season, df in rules_dict.items():
        if not df.empty:
            df['Season'] = season
            all_rules.append(df)
    
    if all_rules:
        final_df = pd.concat(all_rules)
        save_path = os.path.join(table_dir, "seasonal_association_rules.csv")
        final_df.to_csv(save_path, index=False, encoding='utf-8-sig')
        print(f"📄 Đã lưu danh sách luật: {save_path}")



def save_timeseries_model(model, model_name, config, base_path="../"):
    """Lưu mô hình ARIMA/Holt-Winters"""
    model_dir = os.path.join(base_path, config['outputs']['models_dir'])
    os.makedirs(model_dir, exist_ok=True)
    
    save_path = os.path.join(model_dir, f"{model_name}.pkl")
    # Statsmodels models dùng joblib hoặc chính hàm .save() của nó
    model.save(save_path) 
    print(f"📦 Đã lưu mô hình chuỗi thời gian: {save_path}")

def save_metrics_to_table(metrics_dict, filename, config, base_path="../"):
    """Lưu các chỉ số MAE, RMSE vào thư mục tables"""
    table_dir = os.path.join(base_path, config['outputs']['tables'])
    os.makedirs(table_dir, exist_ok=True)
    
    df_metrics = pd.DataFrame([metrics_dict])
    save_path = os.path.join(table_dir, f"{filename}.csv")
    df_metrics.to_csv(save_path, index=False)
    print(f"📊 Đã lưu bảng chỉ số: {save_path}")