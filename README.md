# 🚖 NYC Taxi Trip Duration Prediction System

Hệ thống dự báo thời gian di chuyển taxi tại New York City dựa trên dữ liệu lịch sử từ **New York City Taxi & Limousine Commission (TLC)**, sử dụng **Machine Learning** và kiến trúc **Microservice** với **FastAPI**.

![Python](https://img.shields.io/badge/Python-3.11-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104-green.svg)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3-orange.svg)
![XGBoost](https://img.shields.io/badge/XGBoost-3.1.2-red.svg)
![Docker](https://img.shields.io/badge/Docker-Enabled-blue.svg)

---

## 1. Giới thiệu

Dự án giải quyết bài toán **ước lượng thời gian chuyến đi (Trip Duration)** dựa trên các thông tin đầu vào như:
- Thời gian đón khách
- Vị trí đón/trả
- Số lượng hành khách
- Hãng taxi lựa chọn

Hệ thống được triển khai dưới dạng **Web Application** với bản đồ tương tác, giúp người dùng dễ dàng dự đoán thời gian di chuyển thực tế.

### 1.1. Tính năng chính
- **Dự báo thời gian thực:** Nhận dự đoán ngay lập tức qua API.
- **Bản đồ tương tác:** Kéo thả điểm đón/trả trực quan với Leaflet.js.
- **Tự động trích xuất đặc trưng:** Tính toán khoảng cách Haversine, giờ cao điểm, ngày cuối tuần từ dữ liệu thô.
- **API Documentation:** Tích hợp Swagger UI để kiểm thử API.
- **Đa dạng mô hình ML:** Sử dụng XGBoost, Random Forest, Linear Regression với pipeline chuẩn hóa dữ liệu.

---

## 2. Cấu trúc dự án

```text
taxi-duration-in-NYC-prediction/
├── artifacts/                  
│   ├── model.pkl          		# Mô hình ML đã huấn luyện
│   ├── scaler.pkl              # StandardScaler cho các feature numeric
│   └── features.pkl       		# Danh sách các feature input
├── baseline_result/ 
│   ├── download.png		  	# Ảnh minh họa kết quả baseline
│   └── submission.csv	  		# Kết quả cuối cùng nộp lên Kaggle
├── src/                        
│   ├── preprocessing.py        # Pipeline tiền xử lý dữ liệu
│   └── __init__.py             # Khởi tạo package
├── app/                        
│   ├── main.py                 # Entry point của FastAPI server
│   ├── templates/              # HTML templates
│   │   └── index.html		    # Trang chính với bản đồ
│   └── static/                 # CSS, Images, JS
├── notebooks/                 
│   ├── data/
│   │   ├── train.csv           # Dữ liệu huấn luyện mẫu
│   │   └── test.csv            # Dữ liệu kiểm thử mẫu
│   └── pipeline.ipynb          # Notebook xây dựng pipeline & huấn luyện model
├── tests/                      
│   ├── check_features.py	  	# Unit tests cho tiền xử lý dữ liệu
│   ├── check_scaler.py	  		# Unit tests cho StandardScaler
│   └── test_api.py				# Unit tests cho API
├── requirements.txt            # Danh sách thư viện Python
├── LICENSE                    	# Giấy phép sử dụng
├── Dockerfile                  # Cấu hình Docker
├── .gitignore                  # Loại trừ file/thư mục không cần thiết
├── env/						# Cấu hình môi trường ảo
└── README.md                   # Tài liệu hướng dẫn
```
## 3. Hướng dẫn cài đặt

### 3.1. Tạo môi trường ảo

``` python
# Kiểm tra Python 3.11
python3.11 --version
# Tạo virtual environment
python3.11 -m venv venv
# Activate
source venv/bin/activate # Linux/macOS
venv\Scripts\activate     # Windows
```

### 3.2. Cài đặt dependencies

```python
pip install --upgrade pip  pip install -r requirements.txt
```

## 4. Chạy ứng dụng
### 4.1. Chạy bằng Uvicorn
```python
# Chạy server FastAPI
uvicorn app.main:app --reload
```
### 4.2. Chạy Docker

1.  **Build image**:
```bash 
docker build -t taxi-app:latest .
```

2.  **Chạy container**:
```bash
docker run -d -p 8000:8000 --name taxi-container taxi-app:latest
```

*   \-d → chạy ở background
    
*   \-p 8000:8000 → map port host → container
    
*   \--name taxi-container → đặt tên container
    

3.  **Kiểm tra logs** (nếu muốn xem output):

```bash
docker logs -f taxi-container
```
