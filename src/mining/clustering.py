import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.ensemble import IsolationForest


def run_clustering(df, config):
    data = df.select_dtypes(include=['number']).fillna(0)
    if data.empty:
        raise ValueError('No numeric features for clustering')

    scaler = StandardScaler()
    X = scaler.fit_transform(data)

    k = config.get('clustering', {}).get('n_clusters', 3)
    kmeans = KMeans(n_clusters=k, random_state=42)
    df['cluster_kmeans'] = kmeans.fit_predict(X)

    eps = config.get('clustering', {}).get('eps', 0.5)
    min_samples = config.get('clustering', {}).get('min_samples', 5)
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    df['cluster_dbscan'] = dbscan.fit_predict(X)

    iso = IsolationForest(contamination=config.get('clustering', {}).get('contamination', 0.05), random_state=42)
    df['anomaly'] = iso.fit_predict(X)

    print('clustering done: kmeans n=', k, 'dbscan eps=', eps)
    return df
