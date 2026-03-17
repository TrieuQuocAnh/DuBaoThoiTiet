from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder


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
