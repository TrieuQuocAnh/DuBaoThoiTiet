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