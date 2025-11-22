# 🚖 NYC Taxi Trip Duration Prediction System

Hệ thống dự báo thời gian di chuyển taxi tại New York dựa trên dữ liệu lịch sử từ New York City Taxi & Limousine Commission , sử dụng Machine Learning và kiến trúc Microservice với FastAPI.

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104-green.svg)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3-orange.svg)
![Docker](https://img.shields.io/badge/Docker-Enabled-blue)

## 📖 Giới thiệu

Dự án này giải quyết bài toán ước lượng thời gian chuyến đi (Trip Duration) dựa trên thông tin đầu vào như thời gian đón, địa điểm đón/trả và số lượng hành khách. Hệ thống được triển khai dưới dạng Web Application tích hợp bản đồ tương tác, giúp người dùng dễ dàng ước lượng thời gian di chuyển thực tế.

### ✨ Tính năng chính
- **Dự báo Real-time:** Tính toán thời gian dự kiến ngay lập tức.
- **Bản đồ Tương tác (Interactive Map):** Tích hợp *Leaflet.js* cho phép kéo thả điểm đón/trả trực quan.
- **Tự động trích xuất đặc trưng:** Hệ thống tự động tính toán khoảng cách Haversine, xác định giờ cao điểm, ngày cuối tuần từ dữ liệu thô.
- **API Documentation:** Tích hợp sẵn Swagger UI để kiểm thử API.

---

## 📂 Cấu trúc Dự án

```text
taxi-duration-in-NYC-prediction/
├── artifacts/                  # Chứa các file nhị phân quan trọng (Model, Scaler)
│   ├── best_model.pkl         # Mô hình ML đã huấn luyện tốt nhất
│   ├── scaler.pkl              # Bộ chuẩn hóa dữ liệu (StandardScaler)
│   └── feature_names.pkl       # Danh sách đặc trưng đầu vào
├── src/                        # Source code xử lý logic
│   ├── preprocessing.py        # Pipeline tiền xử lý dữ liệu
│   └── utils.py                # Các hàm tiện ích (Haversine, v.v.)
├── app/                        # Ứng dụng FastAPI
│   ├── main.py                 # Entry point của server
│   ├── templates/              # Giao diện người dùng (HTML/JS)
│   └── static/                 # File tĩnh (CSS/Images)
├── notebooks/                
│   ├── data/
│   │   ├── test.csv            # Dữ liệu kiểm thử mẫu
│   │   └── train.csv           # Dữ liệu huấn luyện mẫu
│   └── pipeline.ipynb          # Notebook xây dựng pipeline và huấn luyện mô hình                 
├── tests/                       # Unit tests cho các module
|   └──test_api.py
├── requirements.txt            # Danh sách thư viện
├── Dockerfile                  # Cấu hình Docker
└── README.md                   # Tài liệu hướng dẫn
```