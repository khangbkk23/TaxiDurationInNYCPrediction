# NYC Taxi Trip Duration Prediction System

Hệ thống dự báo thời gian di chuyển taxi tại New York City dựa trên dữ liệu lịch sử từ **New York City Taxi & Limousine Commission (TLC)**, sử dụng **Machine Learning** và kiến trúc **Microservice** với **FastAPI**.

![Python](https://img.shields.io/badge/Python-3.11-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104-green.svg)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3-orange.svg)
![XGBoost](https://img.shields.io/badge/XGBoost-3.1.2-red.svg)
![Docker](https://img.shields.io/badge/Docker-Enabled-blue.svg)

---

## 1. Giới thiệu

Dự án giải quyết bài toán **ước lượng thời gian chuyến đi (Trip Duration)** dựa trên các thông tin đầu vào như:
- Thời gian đón khách (`pickup_datetime`)
- Vị trí đón / trả khách (kinh độ, vĩ độ)
- Số lượng hành khách (`passenger_count`)
- Hãng taxi (`vendor_id`)

Hệ thống được triển khai dưới dạng **Web Application** với bản đồ tương tác, giúp người dùng dễ dàng dự đoán thời gian di chuyển thực tế.

### 1.1. Tính năng chính
- ⏱ **Dự báo thời gian thực**: Nhập thông tin chuyến đi → nhận thời gian dự kiến ngay lập tức.
- 🗺 **Bản đồ tương tác**: Tích hợp **Leaflet.js**, hỗ trợ chọn điểm đón/trả trực quan.
- 🧮 **Tự động trích xuất đặc trưng**:
  - Khoảng cách Haversine (`distance_km`)
  - Hướng di chuyển (`direction`)
  - Tâm tuyến đường (`center_latitude`, `center_longitude`)
  - Tháng, ngày, thứ, giờ, phút, weekend, rush-hour, night…
- 📚 **API Documentation**:
  - Swagger UI tại `/docs`
- 🤖 **Pipeline Machine Learning hoàn chỉnh**:
  - Tiền xử lý + feature engineering + scaling
  - Huấn luyện với **XGBoost**, **Random Forest**, **Linear Regression**…
  - Lưu lại `model.pkl`, `scaler.pkl`, `features.pkl` để dùng cho API.

### 1.2. Mục đích của dự án

Đây là sản phẩm được phát triển dựa trên đề tài của nhóm, phục vụ cho Bài tập lớn môn **Học máy**, học kỳ **251** tại **Trường Đại học Bách khoa – ĐHQG-HCM**.
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
*Lưu ý*: requirements.txt đã được cấu hình khớp với bản train mô hình
(Python 3.11.13, XGBoost 3.1.2, scikit-learn 1.7.2, ...).

## 4. Chạy ứng dụng
### 4.1. Chạy bằng Uvicorn (local)

```python
# Chạy server FastAPI
uvicorn app.main:app --reload
```

* Mở trình duyệt và truy cập: `http://localhost:8000`
* Để xem tài liệu API: `http://localhost:8000/docs`
* Dừng server: `CTRL + C`
  
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
4. Mở trình duyệt và truy cập: `http://localhost:8000`

5. Dừng container khi không sử dụng:
```bash
docker stop taxi-container
```

6. **Ở những lần chạy sau, chỉ cần truy cập trên `http://localhost:8000`**.
* Nếu container đã bị stop, chạy lại container:
  ```bash
  docker start taxi-container
  ```
* Kiểm tra container đã chạy chưa:
  ```bash
  docker ps
  ```
## 5. Cách retrain và cập nhật model

1. Mở notebook: `notebooks/pipeline.ipynb`.

2. Chạy lại toàn bộ pipeline với dữ liệu mới hoặc tuning tham số.

3. Đảm bảo bước cuối cùng lưu lại:

	* model.pkl

	* scaler.pkl

	* features.pkl
vào thư mục artifacts/.

4. Khởi động lại server FastAPI / container Docker để dùng model mới.
__THE END__
