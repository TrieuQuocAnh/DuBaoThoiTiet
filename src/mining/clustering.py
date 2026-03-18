import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.ensemble import IsolationForest
import pandas as pd
from sklearn.cluster import KMeans
import joblib
import os

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

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

def calculate_clustering_metrics(X, k_range):
    """
    Tính toán Inertia (Elbow) và Silhouette Score cho một khoảng giá trị k.
    """
    inertia = []
    silhouette_avg = []
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X)
        inertia.append(kmeans.inertia_)
        # Silhouette tốn tài nguyên, nên lấy mẫu nếu dữ liệu quá lớn (>10k)
        if len(X) > 10000:
            sample_idx = pd.Series(range(len(X))).sample(5000, random_state=42)
            score = silhouette_score(X.iloc[sample_idx], labels[sample_idx])
        else:
            score = silhouette_score(X, labels)
        silhouette_avg.append(score)
        
    return inertia, silhouette_avg



def perform_kmeans(df, features, n_clusters=4, random_state=42):
    """Thực hiện phân cụm K-Means"""
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    clusters = kmeans.fit_predict(df[features])
    return kmeans, clusters

def get_cluster_profiles(df, features, cluster_col):
    """
    Tạo Hồ sơ cụm (Cluster Profile): 
    Tính giá trị trung bình của các biến trong từng cụm.
    """
    profile = df.groupby(cluster_col)[features].mean()
    # Thêm số lượng bản ghi mỗi cụm
    profile['Count'] = df.groupby(cluster_col).size()
    return profile
