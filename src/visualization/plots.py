import seaborn as sns
import matplotlib.pyplot as plt

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

def plot_distributions(df, cols):
    fig, axes = plt.subplots(len(cols)//2, 2, figsize=(15, 10))
    axes = axes.flatten()
    for i, col in enumerate(cols):
        sns.histplot(df[col], bins=30, kde=True, ax=axes[i])
        axes[i].set_title(f'Distribution of {col}')
    plt.tight_layout()
    plt.show()

def plot_correlation(df, cols):
    plt.figure(figsize=(10, 8))
    sns.heatmap(df[cols].corr(), annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Correlation Matrix")
    plt.show()

def plot_boxplots(df, cols, title="Kiểm tra Outliers"):
    """
    Vẽ Boxplot cho các cột được chọn.
    """
    if not cols: return
    
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df[cols], palette="Set3")
    plt.title(title)
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.show()



def plot_time_series(df, target_col):
    """Vẽ biến thiên dữ liệu theo thời gian (đã resample theo ngày)"""
    plt.figure(figsize=(15, 5))
    # Giả định df đã được set_index là Formatted Date trong notebook hoặc loader
    df[target_col].resample('D').mean().plot() 
    plt.title(f"Biến thiên {target_col} trung bình theo ngày")
    plt.ylabel(target_col)
    plt.grid(True, alpha=0.3)
    plt.show()

def plot_acf_pacf(series, lags=50):
    """Vẽ biểu đồ ACF và PACF để xác định p, d, q cho ARIMA"""
    fig, ax = plt.subplots(1, 2, figsize=(16, 4))
    plot_acf(series.dropna(), lags=lags, ax=ax[0])
    plot_pacf(series.dropna(), lags=lags, ax=ax[1])
    ax[0].set_title("Autocorrelation (ACF)")
    ax[1].set_title("Partial Autocorrelation (PACF)")
    plt.show()

def plot_seasonal_distribution(df, x_col='Month', y_col='Temperature (C)'):
    """Vẽ boxplot theo tháng/mùa để phân tích tính chu kỳ"""
    plt.figure(figsize=(12, 6))
    sns.boxplot(x=x_col, y=y_col, data=df, palette="viridis")
    plt.title(f"Phân bổ {y_col} theo {x_col}")
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.show()



def plot_interactive_correlation(df, cols):
    """Ma trận tương quan tương tác (giống Pywedge)"""
    corr = df[cols].corr()
    fig = px.imshow(corr, text_auto=True, aspect="auto", 
                    color_continuous_scale='RdBu_r', title="Interactive Correlation Matrix")
    fig.show()

def plot_seasonal_analysis_interactive(df, x="Season", y="Temperature (C)"):
    """So sánh sự biến thiên nhiệt độ/độ ẩm giữa các mùa (Yêu cầu đề bài)"""
    fig = px.box(df, x=x, y=y, color=x, points="outliers",
                 title=f"Phân tích {y} theo {x} (Interactive Boxplot)")
    fig.show()

def plot_scatter_matrix(df, cols, color_col='Season'):
    """Biểu đồ Scatter Matrix để xem sự phân tách các cụm dữ liệu"""
    fig = px.scatter_matrix(df, dimensions=cols, color=color_col,
                            title="Scatter Matrix by Season",
                            opacity=0.5)
    fig.update_traces(diagonal_visible=False)
    fig.show()



def plot_rules_scatter(rules, season_name):
    """Trực quan hóa luật kết hợp bằng biểu đồ Scatter (Support vs Confidence)"""
    if rules.empty:
        print(f"Không có luật để vẽ cho mùa {season_name}")
        return

    # Chuyển frozenset sang string để hiển thị trên tooltip
    plot_df = rules.copy()
    plot_df['rule'] = plot_df['antecedents'].astype(str) + " -> " + plot_df['consequents'].astype(str)

    fig = px.scatter(plot_df, x="support", y="confidence", 
                     size="lift", color="lift",
                     hover_data=['rule'],
                     title=f"Association Rules Analysis - {season_name}",
                     labels={"support": "Support (Độ phổ biến)", "confidence": "Confidence (Độ tin cậy)"},
                     color_continuous_scale="Viridis")
    fig.show()

def plot_elbow_method(k_range, inertia, silhouette_avg):
    """Vẽ biểu đồ Elbow và Silhouette để chọn k"""
    fig, ax1 = plt.subplots(figsize=(10, 5))

    # Vẽ đường Elbow (Inertia)
    color = 'tab:blue'
    ax1.set_xlabel('Số lượng cụm (k)')
    ax1.set_ylabel('Inertia (Sum of Squares)', color=color)
    ax1.plot(k_range, inertia, 'o-', color=color, label='Inertia')
    ax1.tick_params(axis='y', labelcolor=color)

    # Vẽ đường Silhouette (Trục y thứ 2)
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Silhouette Score', color=color)
    ax2.plot(k_range, silhouette_avg, 's--', color=color, label='Silhouette')
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title('Phương pháp Elbow & Silhouette để chọn K tối ưu')
    fig.tight_layout()
    plt.show()

import plotly.graph_objects as go

def plot_cluster_radar(profile_df, features):
    """Vẽ biểu đồ Radar để so sánh đặc điểm các cụm thời tiết"""
    fig = go.Figure()

    for i in profile_df.index:
        fig.add_trace(go.Scatterpolar(
            r=profile_df.loc[i, features].values,
            theta=features,
            fill='toself',
            name=f'Cluster {i}'
        ))

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[df_scaled.min().min(), df_scaled.max().max()])),
        showlegend=True,
        title="Hồ sơ cụm thời tiết (Cluster Profiles)"
    )
    fig.show()