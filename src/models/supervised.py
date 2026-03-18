from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder
import joblib
import os

def run_supervised(df, config):
    target_col = config.get('eda', {}).get('target_col', 'target')
    if target_col not in df.columns:
        raise ValueError(f'Target column {target_col} not found in dataframe')

    X = df.drop(columns=[target_col], errors='ignore').select_dtypes(include=['number']).fillna(0)
    y = df[target_col]

    if y.dtype == 'object' or y.dtype.name == 'category':
        y = LabelEncoder().fit_transform(y.fillna('missing'))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    metrics = {
        'rmse': mean_squared_error(y_test, y_pred, squared=False),
        'mae': mean_absolute_error(y_test, y_pred),
        'r2': r2_score(y_test, y_pred)
    }

    print('Supervised training complete', metrics)
    return model, metrics



def save_model(model, model_name, config, base_path="../"):
    model_dir = os.path.join(base_path, config['outputs']['models_dir'])
    os.makedirs(model_dir, exist_ok=True)
    save_path = os.path.join(model_dir, f"{model_name}.pkl")
    joblib.dump(model, save_path)
    print(f"📦 Đã lưu model tại: {save_path}")