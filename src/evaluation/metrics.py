from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def evaluate_model(model, df, config):
    target_col = config.get('eda', {}).get('target_col', 'target')
    if target_col not in df.columns:
        raise ValueError(f'Target column {target_col} not found in df')

    X = df.drop(columns=[target_col], errors='ignore').select_dtypes(include=['number']).fillna(0)
    y = df[target_col]

    if y.dtype == 'object' or y.dtype.name == 'category':
        y = LabelEncoder().fit_transform(y.fillna('missing'))

    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    y_pred = model.predict(X_test)

    metrics = {
        'rmse': mean_squared_error(y_test, y_pred, squared=False),
        'mae': mean_absolute_error(y_test, y_pred),
        'r2': r2_score(y_test, y_pred)
    }

    print('Evaluation results:', metrics)
    return metrics
