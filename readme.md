# 🌤️ Dự Báo Thời Tiết Và Khai Thác Dữ Liệu Thời Tiết

Dự án này khai thác và phân tích dữ liệu thời tiết bằng các kỹ thuật học máy tiên tiến, bao gồm phân tích khám phá, khai thác quy tắc liên kết, phân cụm, phân loại và dự báo chuỗi thời gian.

## 📋 Mục Đích Dự Án

Dự án nhằm mục đích:
- **Thu thập và làm sạch** dữ liệu thời tiết lịch sử
- **Khám phá** các mẫu và xu hướng trong dữ liệu thời tiết
- **Tạo đặc trưng** nhằm nâng cao hiệu năng mô hình
- **Phân tích quy tắc liên kết** để tìm mối quan hệ giữa các biến thời tiết
- **Phân cụm** dữ liệu để nhóm các ngày có điều kiện thời tiết tương tự
- **Phân loại** để dự đoán các sự kiện thời tiết
- **Dự báo chuỗi thời gian** cho các biến thời tiết chính
- **Đánh giá** hiệu suất các mô hình

## 🗂️ Cấu Trúc Dự Án

```
DuBaoThoiTiet/
│
├── readme.md                    # Tài liệu dự án này
├── requirements.txt             # Các thư viện Python cần thiết
│
├── configs/
│   └── params.yaml              # Tham số cấu hình
│
├── data/
│   ├── raw/
│   │   └── weatherHistory.csv   # Dữ liệu thô gốc
│   └── processed/
│       ├── weather_cleaned.csv      # Dữ liệu sau làm sạch
│       ├── weather_featured.csv     # Dữ liệu sau tạo đặc trưng
│       └── weather_mining.csv       # Dữ liệu sẵn sàng cho khai thác
│
├── notebooks/                   # Jupyter Notebooks cho phân tích từng bước
│   ├── 01_eda.ipynb                 # Phân tích khám phá dữ liệu (EDA)
│   ├── 01_eda_executed.ipynb        # EDA đã chạy
│   ├── 02_preprocess_feature.ipynb  # Tiền xử lý & tạo đặc trưng
│   ├── 03_association_rules.ipynb   # Phân tích quy tắc liên kết
│   ├── 03_mining_or_clustering.ipynb    # Lựa chọn khai thác/phân cụm
│   ├── 03b_clustering.ipynb         # Phân cụm dữ liệu
│   ├── 03c_classification.ipynb     # Phân loại dữ liệu
│   ├── 03d_time_series.ipynb        # Phân tích chuỗi thời gian
│   ├── 04_modeling.ipynb            # Xây dựng mô hình dự báo
│   └── 05_evaluation_report.ipynb   # Báo cáo đánh giá
│
├── src/                         # Mã nguồn Python chính
│   ├── __init__.py
│   ├── data/                    # Xử lý dữ liệu
│   │   ├── loader.py            # Tải dữ liệu
│   │   └── cleaner.py           # Làm sạch dữ liệu
│   ├── features/                # Tạo đặc trưng
│   │   └── builder.py           # Xây dựng đặc trưng
│   ├── mining/                  # Khai thác dữ liệu
│   │   ├── association.py       # Quy tắc liên kết
│   │   └── clustering.py        # Phân cụm
│   ├── models/                  # Mô hình học máy
│   │   ├── supervised.py        # Mô hình phân loại
│   │   └── forecasting.py       # Mô hình dự báo
│   ├── evaluation/              # Đánh giá mô hình
│   │   ├── metrics.py           # Các chỉ số đánh giá
│   │   └── report.py            # Báo cáo kết quả
│   └── visualization/           # Trực quan hoá dữ liệu
│       └── plots.py             # Các biểu đồ
│
├── scripts/                     # Script chạy quy trình
│   ├── run_pipeline.py          # Chạy toàn bộ quy trình
│   └── run_papermill.py         # Chạy notebooks tự động
│
└── outputs/                     # Kết quả đầu ra
    └── tables/
        └── association_rules_seasonal.csv
```

## 🚀 Bắt Đầu Nhanh

### Yêu Cầu Hệ Thống
- Python 3.8+
- pip hoặc conda

### Cài Đặt

1. **Clone hoặc tải dự án**
   ```bash
   cd DuBaoThoiTiet
   ```

2. **Cài đặt các thư viện cần thiết**
   ```bash
   pip install -r requirements.txt
   ```

3. **Cấu hình tham số** (nếu cần)
   - Sửa đổi `configs/params.yaml` theo nhu cầu

### Chạy Dự Án

#### Phương pháp 1: Chạy toàn bộ quy trình
```bash
python scripts/run_pipeline.py
```

#### Phương pháp 2: Chạy từng notebook
```bash
jupyter notebook notebooks/01_eda.ipynb
```

#### Phương pháp 3: Chạy notebooks tự động với Papermill
```bash
python scripts/run_papermill.py
```

## 📊 Quy Trình Phân Tích

### 1. **Phân Tích Khám Phá (EDA)** 📈
- File: `notebooks/01_eda.ipynb`
- Xem xét thống kê mô tả, phân phối, và mối quan hệ giữa các biến
- Xác định giá trị thiếu, ngoại lệ
- Trực quan hoá dữ liệu

### 2. **Tiền Xử Lý & Tạo Đặc Trưng** 🔧
- File: `notebooks/02_preprocess_feature.ipynb`
- Xử lý giá trị thiếu
- Chuẩn hoá/chuẩn hóa dữ liệu
- Tạo các đặc trưng mới từ các biến hiện có
- Lưu dữ liệu đã xử lý

### 3. **Khai Thác Dữ Liệu** 🔍
Dự án cung cấp ba phương pháp khai thác:

#### a) **Quy Tắc Liên Kết** (Association Rules)
- File: `notebooks/03_association_rules.ipynb`
- Phát hiện các mẫu và quy tắc trong dữ liệu
- Tìm mối quan hệ giữa các sự kiện thời tiết
- Output: `outputs/tables/association_rules_seasonal.csv`

#### b) **Phân Cụm** (Clustering)
- File: `notebooks/03b_clustering.ipynb`
- Nhóm các ngày có điều kiện thời tiết tương tự
- Sử dụng: K-means, Hierarchical Clustering, DBSCAN

#### c) **Phân Loại** (Classification)
- File: `notebooks/03c_classification.ipynb`
- Dự đoán các sự kiện thời tiết cụ thể
- Sử dụng: Decision Tree, Random Forest, SVM, Logistic Regression

### 4. **Phân Tích Chuỗi Thời Gian** ⏰
- File: `notebooks/03d_time_series.ipynb`
- Nhận diện xu hướng, mùa vụ, tính chu kỳ
- Làm mịn và phân rã chuỗi thời gian

### 5. **Xây Dựng & Huấn Luyện Mô Hình** 🤖
- File: `notebooks/04_modeling.ipynb`
- Huấn luyện các mô hình dự báo
- Tối ưu hóa siêu tham số
- Lưu mô hình đã huấn luyện

### 6. **Đánh Giá & Báo Cáo** 📋
- File: `notebooks/05_evaluation_report.ipynb`
- Đánh giá hiệu suất mô hình
- So sánh các mô hình khác nhau
- Tạo báo cáo tóm tắt kết quả

## 🛠️ Các Thư Viện Chính

Dự án sử dụng các thư viện sau (xem `requirements.txt` để danh sách đầy đủ):

- **pandas**: Xử lý và phân tích dữ liệu
- **numpy**: Tính toán số học
- **scikit-learn**: Học máy
- **matplotlib / seaborn**: Trực quan hoá dữ liệu
- **plotly**: Biểu đồ tương tác
- **mlxtend**: Khai thác dữ liệu, quy tắc liên kết
- **statsmodels**: Phân tích chuỗi thời gian
- **TensorFlow / Keras**: Deep Learning (nếu cần)
- **jupyter / jupyterlab**: Phát triển tương tác
- **papermill**: Chạy notebooks một cách lập trình

## 📁 Mô Tả Các Module Chính

### `src/data/`
- `loader.py`: Hàm tải dữ liệu từ các tập tin CSV
- `cleaner.py`: Hàm làm sạch, chuẩn hóa dữ liệu

### `src/features/`
- `builder.py`: Tạo các đặc trưng mới từ dữ liệu gốc

### `src/mining/`
- `association.py`: Phân tích quy tắc liên kết
- `clustering.py`: Các thuật toán phân cụm

### `src/models/`
- `supervised.py`: Mô hình phân loại (supervised learning)
- `forecasting.py`: Mô hình dự báo chuỗi thời gian

### `src/evaluation/`
- `metrics.py`: Các hàm tính toán chỉ số đánh giá
- `report.py`: Tạo báo cáo kết quả

### `src/visualization/`
- `plots.py`: Các hàm vẽ biểu đồ và trực quan hoá

## 📝 Ghi Chú Sử Dụng

1. **Dữ liệu gốc**: Đặt tập tin CSV gốc vào `data/raw/`
2. **Cấu hình**: Chỉnh sửa `configs/params.yaml` để thay đổi tham số
3. **Tham số mô hình**: Sửa đổi trong các notebook hoặc tập tin cấu hình
4. **Kết quả**: Kiểm tra `outputs/` để xem kết quả phân tích

## 🎯 Kết Quả Mong Đợi

- ✅ Dữ liệu làm sạch và xử lý tốt
- ✅ Hiểu rõ về mẫu và xu hướng thời tiết
- ✅ Các quy tắc liên kết có ý nghĩa
- ✅ Cụm dữ liệu rõ ràng và có thể giải thích
- ✅ Mô hình phân loại với độ chính xác cao
- ✅ Dự báo chuỗi thời gian chính xác
- ✅ Báo cáo chi tiết về kết quả

## 🐛 Khắc Phục Sự Cố

| Sự Cố | Giải Pháp |
|-------|----------|
| Thiếu thư viện | Chạy `pip install -r requirements.txt` |
| Dữ liệu không được tìm thấy | Kiểm tra đường dẫn tập tin trong `src/data/loader.py` |
| Lỗi Jupyter Notebook | Chạy `pip install jupyterlab` |
| Bộ nhớ không đủ | Giảm kích thước dữ liệu hoặc dùng xử lý dữ liệu theo batch |

## 👤 Liên Hệ & Đóng Góp

Nếu có câu hỏi hoặc đề xuất cải tiến, vui lòng liên hệ hoặc tạo issue.

## 📄 Giấy Phép

Dự án này được cấp phép dưới [Chỉ định giấy phép của bạn, ví dụ: MIT License]

---

**Cập nhật lần cuối**: Tháng 3 năm 2026
