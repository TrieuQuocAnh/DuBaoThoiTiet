import seaborn as sns
import matplotlib.pyplot as plt

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

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

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(y_true, y_pred, labels, title="Ma trận nhầm lẫn"):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=labels, yticklabels=labels, cmap='Blues')
    plt.title(title)
    plt.ylabel('Thực tế')
    plt.xlabel('Dự báo')
    plt.show()

def plot_weather_counts(df):
    """
    Vẽ biểu đồ cột so sánh số lượng các giá trị trong cột 'Summary' và 'Precip Type'
    """
    # 1. Chuẩn bị dữ liệu cho Summary
    # Sắp xếp giảm dần để biểu đồ dễ nhìn hơn
    summary_counts = df['Summary'].value_counts().reset_index()
    summary_counts.columns = ['Value', 'Count']
    
    # 2. Chuẩn bị dữ liệu cho Precip Type
    # Lưu ý: Xử lý giá trị Null nếu cần (ở đây hiển thị cả Null dưới tên 'Missing')
    precip_counts = df['Precip Type'].fillna('Missing').value_counts().reset_index()
    precip_counts.columns = ['Value', 'Count']

    # Tạo Subplots: 1 hàng, 2 cột
    fig = make_subplots(
        rows=1, cols=2, 
        subplot_titles=("Phân phối các nhãn trong Summary", "Phân phối các nhãn trong Precip Type"),
        horizontal_spacing=0.15
    )

    # Thêm biểu đồ cho Summary (Cột 1)
    fig.add_trace(
        go.Bar(
            x=summary_counts['Value'], 
            y=summary_counts['Count'],
            marker_color='indianred',
            name='Summary'
        ),
        row=1, col=1
    )

    # Thêm biểu đồ cho Precip Type (Cột 2)
    fig.add_trace(
        go.Bar(
            x=precip_counts['Value'], 
            y=precip_counts['Count'],
            marker_color='lightseagreen',
            name='Precip Type'
        ),
        row=1, col=2
    )

    # Cập nhật giao diện (Layout)
    fig.update_layout(
        title_text="Thống kê số lượng giá trị trong Dataset Thời tiết",
        template="plotly_white",
        showlegend=False,
        height=600
    )
    
    # Xoay nhãn trục X của Summary vì có quá nhiều nhãn dài
    fig.update_xaxes(tickangle=45, row=1, col=1)
    
    fig.show()