import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from mlxtend.frequent_patterns import apriori, association_rules
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Cấu hình trang
st.set_page_config(
    page_title="🌤️ Dự Báo Thời Tiết & Khai Thác Dữ Liệu",
    page_icon="🌦️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS tùy chỉnh
st.markdown("""
<style>
    .main-header {
        text-align: center;
        color: #FF6B6B;
        font-size: 2em;
        margin-bottom: 2em;
    }
    .section-title {
        color: #4ECDC4;
        font-size: 1.5em;
        border-bottom: 2px solid #4ECDC4;
        padding-bottom: 0.5em;
    }
</style>
""", unsafe_allow_html=True)

# Load dữ liệu
@st.cache_data
def load_data():
    try:
        # Cố gắng load dữ liệu xử lý
        df = pd.read_csv('data/processed/weather_featured.csv')
    except:
        try:
            # Nếu không có, load dữ liệu gốc
            df = pd.read_csv('data/raw/weatherHistory.csv')
        except:
            # Nếu không có, tạo dữ liệu mẫu
            np.random.seed(42)
            dates = pd.date_range('2020-01-01', periods=1000)
            df = pd.DataFrame({
                'datetime': dates,
                'temperature': np.random.normal(20, 10, 1000),
                'humidity': np.random.normal(60, 15, 1000),
                'windSpeed': np.random.normal(10, 5, 1000),
                'pressure': np.random.normal(1013, 10, 1000),
                'precip': np.random.exponential(2, 1000),
                'visibility': np.random.normal(10, 2, 1000),
                'uvIndex': np.random.uniform(0, 11, 1000)
            })
    return df

# Tạo header
st.markdown("<div class='main-header'>🌤️ Dự Báo Thời Tiết & Khai Thác Dữ Liệu</div>", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.title("📋 Menu Chính")
    page = st.radio(
        "Chọn chức năng:",
        ["🏠 Trang Chủ", 
         "📊 EDA - Phân Tích Khám Phá",
         "🔗 Quy Tắc Liên Kết",
         "🎯 Phân Cụm",
         "🏆 Phân Loại",
         "📈 Dự Báo Chuỗi Thời Gian",
         "📉 Đánh Giá Mô Hình"]
    )

# Load dữ liệu
df = load_data()

# ========== TRANG CHỦ ==========
if page == "🏠 Trang Chủ":
    st.markdown("<div class='section-title'>Chào Mừng Đến Dự Án Dự Báo Thời Tiết</div>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.info("📌 **Thông Tin Dữ Liệu**")
        st.metric("Số hàng dữ liệu", len(df))
        st.metric("Số cột", len(df.columns))
        st.metric("Khoảng thời gian", f"{len(df)} bản ghi")
    
    with col2:
        st.info("🎯 **Các Chức Năng Chính**")
        st.markdown("""
        - 📊 Phân tích khám phá (EDA)
        - 🔗 Quy tắc liên kết
        - 🎯 Phân cụm K-means
        - 🏆 Phân loại dữ liệu
        - 📈 Dự báo chuỗi thời gian
        - 📉 Đánh giá mô hình
        """)
    
    st.markdown("---")
    st.markdown("<div class='section-title'>5 Hàng Dữ Liệu Đầu Tiên</div>", unsafe_allow_html=True)
    st.dataframe(df.head(), use_container_width=True)

# ========== EDA - PHÂN TÍCH KHÁM PHÁ ==========
elif page == "📊 EDA - Phân Tích Khám Phá":
    st.markdown("<div class='section-title'>📊 Phân Tích Khám Phá Dữ Liệu (EDA)</div>", unsafe_allow_html=True)
    
    # Thống kê mô tả
    st.subheader("📈 Thống Kê Mô Tả")
    st.dataframe(df.describe(), use_container_width=True)
    
    # Chọn cột để phân tích
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Phân Phối của Các Biến")
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        selected_col = st.selectbox("Chọn biến để xem phân phối:", numeric_cols)
        
        fig, ax = plt.subplots(figsize=(10, 5))
        df[selected_col].hist(bins=30, ax=ax, edgecolor='black', alpha=0.7)
        ax.set_title(f'Phân Phối của {selected_col}')
        ax.set_xlabel(selected_col)
        ax.set_ylabel('Số lượng')
        st.pyplot(fig)
    
    with col2:
        st.subheader("🔍 Kiểm Tra Giá Trị Thiếu")
        missing_data = df.isnull().sum()
        if missing_data.sum() > 0:
            fig, ax = plt.subplots(figsize=(10, 5))
            missing_data[missing_data > 0].plot(kind='barh', ax=ax, color='coral')
            ax.set_title('Giá Trị Thiếu')
            ax.set_xlabel('Số lượng giá trị thiếu')
            st.pyplot(fig)
        else:
            st.success("✅ Không có giá trị thiếu!")
    
    # Ma trận tương quan
    st.subheader("🔗 Ma Trận Tương Quan")
    numeric_df = df.select_dtypes(include=[np.number])
    if len(numeric_df.columns) > 0:
        fig, ax = plt.subplots(figsize=(10, 8))
        corr_matrix = numeric_df.corr()
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', ax=ax, cbar_kws={'label': 'Tương Quan'})
        ax.set_title('Ma Trận Tương Quan')
        st.pyplot(fig)

# ========== QUY TẮC LIÊN KẾT ==========
elif page == "🔗 Quy Tắc Liên Kết":
    st.markdown("<div class='section-title'>🔗 Phân Tích Quy Tắc Liên Kết</div>", unsafe_allow_html=True)
    
    # Chuẩn bị dữ liệu cho association rules
    numeric_df = df.select_dtypes(include=[np.number])
    
    # Rời rạc hóa dữ liệu
    st.subheader("📊 Tạo Quy Tắc Liên Kết")
    
    # Rời rạc hóa từng biến
    df_binary = pd.DataFrame()
    for col in numeric_df.columns[:5]:  # Lấy 5 cột đầu
        df_binary[f'{col}_high'] = (numeric_df[col] > numeric_df[col].median()).astype(int)
        df_binary[f'{col}_low'] = (numeric_df[col] <= numeric_df[col].median()).astype(int)
    
    # Áp dụng Apriori
    frequent_itemsets = apriori(df_binary, min_support=0.1, use_colnames=True)
    
    if len(frequent_itemsets) > 0:
        rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.3)
        
        if len(rules) > 0:
            rules['antecedent_str'] = rules['antecedents'].apply(lambda x: ', '.join(list(x)))
            rules['consequent_str'] = rules['consequents'].apply(lambda x: ', '.join(list(x)))
            
            st.subheader(f"Tìm Thấy {len(rules)} Quy Tắc")
            
            # Hiển thị quy tắc
            display_rules = rules[['antecedent_str', 'consequent_str', 'support', 'confidence', 'lift']].copy()
            display_rules.columns = ['Tiền Đề', 'Hệ Quả', 'Support', 'Confidence', 'Lift']
            st.dataframe(display_rules.head(20), use_container_width=True)
            
            # Biểu đồ
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.scatter(rules, x='support', y='confidence', size='lift', 
                               color='lift', title='Biểu Đồ Quy Tắc Liên Kết')
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = px.bar(rules.head(10).sort_values('lift'), x='lift', 
                           y=display_rules.head(10)['Tiền Đề'], title='Top 10 Quy Tắc')
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("⚠️ Không tìm thấy quy tắc nào")
    else:
        st.info("ℹ️ Cần tăng min_support để tìm itemsets")

# ========== PHÂN CỤM ==========
elif page == "🎯 Phân Cụm":
    st.markdown("<div class='section-title'>🎯 Phân Cụm K-means</div>", unsafe_allow_html=True)
    
    # Chuẩn bị dữ liệu
    numeric_df = df.select_dtypes(include=[np.number]).head(200)  # Lấy 200 hàng
    
    # Chuẩn hóa dữ liệu
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(numeric_df)
    
    # Chọn số cụm
    n_clusters = st.slider("Chọn số cụm:", 2, 10, 3)
    
    # Áp dụng K-means
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(df_scaled)
    
    # Thêm label cụm vào dữ liệu gốc
    numeric_df['Cluster'] = clusters
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Số Điểm trong Mỗi Cụm")
        cluster_counts = pd.Series(clusters).value_counts().sort_index()
        fig, ax = plt.subplots(figsize=(8, 5))
        cluster_counts.plot(kind='bar', ax=ax, color='skyblue', edgecolor='black')
        ax.set_title('Số Điểm trong Mỗi Cụm')
        ax.set_xlabel('Cụm')
        ax.set_ylabel('Số lượng')
        st.pyplot(fig)
    
    with col2:
        st.subheader("🔍 Thông Tin Cụm")
        st.info(f"""
        - **Số cụm**: {n_clusters}
        - **Tổng điểm**: {len(clusters)}
        - **Inertia**: {kmeans.inertia_:.2f}
        """)
    
    # Biểu đồ phân tán (2D)
    st.subheader("📈 Biểu Đồ Phân Tán 2D")
    fig = px.scatter(numeric_df, x=numeric_df.columns[0], y=numeric_df.columns[1], 
                     color='Cluster', title='Phân Cụm K-means')
    st.plotly_chart(fig, use_container_width=True)
    
    # Hiển thị trung tâm cụm
    st.subheader("📍 Trung Tâm Cụm")
    centers_df = pd.DataFrame(scaler.inverse_transform(kmeans.cluster_centers_), 
                             columns=numeric_df.columns[:-1])
    st.dataframe(centers_df, use_container_width=True)

# ========== PHÂN LOẠI ==========
elif page == "🏆 Phân Loại":
    st.markdown("<div class='section-title'>🏆 Phân Loại Dữ Liệu (Classification)</div>", unsafe_allow_html=True)
    
    # Chuẩn bị dữ liệu
    numeric_df = df.select_dtypes(include=[np.number]).head(300)
    
    if len(numeric_df.columns) > 0:
        # Tạo biến mục tiêu (chia thành lớp dựa trên trung vị)
        target_col = numeric_df.columns[0]
        y = (numeric_df[target_col] > numeric_df[target_col].median()).astype(int)
        X = numeric_df.drop(columns=[target_col])
        
        # Chia train-test
        train_size = int(0.8 * len(X))
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        
        # Huấn luyện Random Forest
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_model.fit(X_train, y_train)
        
        # Dự đoán
        y_pred = rf_model.predict(X_test)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Hiệu Suất Mô Hình")
            accuracy = accuracy_score(y_test, y_pred)
            st.metric("Độ Chính Xác (Accuracy)", f"{accuracy:.2%}")
            
            st.subheader("📈 Ma Trận Nhầm Lẫn")
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, 
                       xticklabels=['Âm', 'Dương'], yticklabels=['Âm', 'Dương'])
            ax.set_title('Ma Trận Nhầm Lẫn')
            ax.set_ylabel('Thực Tế')
            ax.set_xlabel('Dự Đoán')
            st.pyplot(fig)
        
        with col2:
            st.subheader("📋 Báo Cáo Phân Loại")
            report = classification_report(y_test, y_pred, output_dict=True)
            report_df = pd.DataFrame(report).transpose()
            st.dataframe(report_df, use_container_width=True)
        
        st.subheader("🌳 Tầm Quan Trọng Đặc Trưng")
        importance_df = pd.DataFrame({
            'Đặc Trưng': X.columns,
            'Tầm Quan Trọng': rf_model.feature_importances_
        }).sort_values('Tầm Quan Trọng', ascending=False)
        
        fig = px.bar(importance_df, x='Tầm Quan Trọng', y='Đặc Trưng', 
                    title='Tầm Quan Trọng Đặc Trưng', orientation='h')
        st.plotly_chart(fig, use_container_width=True)

# ========== DỰ BÁO CHUỖI THỜI GIAN ==========
elif page == "📈 Dự Báo Chuỗi Thời Gian":
    st.markdown("<div class='section-title'>📈 Dự Báo Chuỗi Thời Gian</div>", unsafe_allow_html=True)
    
    numeric_df = df.select_dtypes(include=[np.number])
    
    if len(numeric_df.columns) > 0:
        # Chọn cột để dự báo
        col_to_forecast = st.selectbox("Chọn biến để dự báo:", numeric_df.columns)
        series = numeric_df[col_to_forecast].head(200).values
        
        # Dự báo đơn giản (trung bình động)
        window = st.slider("Chọn kích thước cửa sổ (window):", 5, 20, 10)
        
        moving_avg = pd.Series(series).rolling(window=window).mean()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Biểu Đồ Chuỗi Thời Gian")
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(series, label='Giá Trị Gốc', linewidth=2)
            ax.plot(moving_avg, label=f'Trung Bình Động ({window})', linewidth=2, color='red')
            ax.set_title(f'Chuỗi Thời Gian - {col_to_forecast}')
            ax.set_xlabel('Thời Gian')
            ax.set_ylabel('Giá Trị')
            ax.legend()
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
        
        with col2:
            st.subheader("📈 Thống Kê Chuỗi")
            stats = pd.Series(series).describe()
            st.dataframe(stats, use_container_width=True)
        
        # Dự báo tương lai
        st.subheader("🔮 Dự Báo Tương Lai")
        forecast_periods = st.slider("Số kỳ dự báo:", 5, 30, 10)
        
        last_values = series[-window:]
        forecast = []
        for _ in range(forecast_periods):
            next_val = np.mean(last_values)
            forecast.append(next_val)
            last_values = np.append(last_values[1:], next_val)
        
        # Vẽ dự báo
        fig, ax = plt.subplots(figsize=(12, 6))
        all_values = np.concatenate([series, forecast])
        ax.plot(range(len(series)), series, 'b-', label='Dữ Liệu Lịch Sử', linewidth=2)
        ax.plot(range(len(series)-1, len(all_values)), np.concatenate([[series[-1]], forecast]), 
               'r--', label='Dự Báo', linewidth=2)
        ax.set_title(f'Dự Báo {col_to_forecast}')
        ax.set_xlabel('Thời Gian')
        ax.set_ylabel('Giá Trị')
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        
        st.success(f"✅ Dự báo {forecast_periods} kỳ tới: {[f'{v:.2f}' for v in forecast[:5]]}...")

# ========== ĐÁNH GIÁ MÔ HÌNH ==========
elif page == "📉 Đánh Giá Mô Hình":
    st.markdown("<div class='section-title'>📉 Đánh Giá & So Sánh Mô Hình</div>", unsafe_allow_html=True)
    
    numeric_df = df.select_dtypes(include=[np.number]).head(300)
    
    if len(numeric_df.columns) > 1:
        # Chuẩn bị dữ liệu
        X = numeric_df.iloc[:, 1:]
        y = numeric_df.iloc[:, 0]
        
        # Chia train-test
        train_size = int(0.8 * len(X))
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        
        st.subheader("🔧 Huấn Luyện Mô Hình")
        
        # Huấn luyện mô hình đơn giản
        from sklearn.linear_model import LinearRegression
        from sklearn.preprocessing import PolynomialFeatures
        
        model_lr = LinearRegression()
        model_lr.fit(X_train, y_train)
        y_pred_lr = model_lr.predict(X_test)
        
        # Tính các chỉ số
        mae_lr = mean_absolute_error(y_test, y_pred_lr)
        rmse_lr = np.sqrt(mean_squared_error(y_test, y_pred_lr))
        r2_lr = r2_score(y_test, y_pred_lr)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("MAE (Linear)", f"{mae_lr:.4f}")
        with col2:
            st.metric("RMSE (Linear)", f"{rmse_lr:.4f}")
        with col3:
            st.metric("R² (Linear)", f"{r2_lr:.4f}")
        
        st.subheader("📊 Biểu Đồ Dự Đoán vs Thực Tế")
        
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.scatter(y_test, y_pred_lr, alpha=0.6, s=50)
        ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        ax.set_xlabel('Giá Trị Thực Tế')
        ax.set_ylabel('Giá Trị Dự Đoán')
        ax.set_title('Dự Đoán vs Thực Tế')
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        
        st.subheader("📈 Phân Bố Sai Số")
        residuals = y_test.values - y_pred_lr
        
        col1, col2 = st.columns(2)
        with col1:
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.hist(residuals, bins=30, edgecolor='black', alpha=0.7)
            ax.set_title('Phân Bố Sai Số')
            ax.set_xlabel('Sai Số')
            ax.set_ylabel('Số Lượng')
            st.pyplot(fig)
        
        with col2:
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.scatter(y_pred_lr, residuals, alpha=0.6)
            ax.axhline(y=0, color='r', linestyle='--')
            ax.set_xlabel('Giá Trị Dự Đoán')
            ax.set_ylabel('Sai Số')
            ax.set_title('Biểu Đồ Sai Số')
            st.pyplot(fig)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #999; margin-top: 2em;'>
    <p>🌤️ Dự Án Dự Báo Thời Tiết & Khai Thác Dữ Liệu | Phiên Bản 1.0 | Tháng 3 năm 2026</p>
</div>
""", unsafe_allow_html=True)
