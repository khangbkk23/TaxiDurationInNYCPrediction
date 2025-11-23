from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import Optional
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
import os
import sys

# Thêm đường dẫn src để import (phòng trường hợp lỗi path)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# --- 1. CẤU HÌNH CỘT ---
ALL_FEATURES = [
    'vendor_id', 'passenger_count', 'pickup_longitude', 'pickup_latitude',
    'dropoff_longitude', 'dropoff_latitude', 'store_and_fwd_flag',
    'pickup_year', 'pickup_month', 'pickup_day', 'pickup_hour',
    'pickup_minute', 'pickup_weekday', 'pickup_yday', 'pickup_weekend',
    'is_rush_hour', 'is_night', 'distance_km', 'direction',
    'center_latitude', 'center_longitude'
]

SCALED_FEATURES = [
    'vendor_id', 'passenger_count',
    'pickup_longitude', 'pickup_latitude',
    'dropoff_longitude', 'dropoff_latitude',
    'pickup_hour', 'pickup_weekday', 'pickup_month',
    'distance_km', 'direction',
    'center_latitude', 'center_longitude'
]

# --- 2. HÀM TÍNH TOÁN ---
def haversine_array(lat1, lon1, lat2, lon2):
    R = 6378.137
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c

def transform_raw_data(data_dict):
    df = pd.DataFrame([data_dict])
    
    # Thời gian
    df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'])
    df['pickup_year'] = df['pickup_datetime'].dt.year
    df['pickup_month'] = df['pickup_datetime'].dt.month
    df['pickup_day'] = df['pickup_datetime'].dt.day
    df['pickup_hour'] = df['pickup_datetime'].dt.hour
    df['pickup_minute'] = df['pickup_datetime'].dt.minute
    df['pickup_weekday'] = df['pickup_datetime'].dt.weekday
    df['pickup_yday'] = df['pickup_datetime'].dt.dayofyear
    
    # Logic
    df['pickup_weekend'] = (df['pickup_weekday'] >= 5).astype(int)
    df['is_rush_hour'] = (((df['pickup_hour'] >= 7) & (df['pickup_hour'] <= 9)) |
                          ((df['pickup_hour'] >= 17) & (df['pickup_hour'] <= 19))).astype(int)
    df['is_night'] = ((df['pickup_hour'] >= 22) | (df['pickup_hour'] <= 5)).astype(int)
    
    # Không gian
    df['distance_km'] = haversine_array(
        df['pickup_latitude'], df['pickup_longitude'],
        df['dropoff_latitude'], df['dropoff_longitude']
    )
    
    dlon = df['dropoff_longitude'] - df['pickup_longitude']
    dlat = df['dropoff_latitude'] - df['pickup_latitude']
    df['direction'] = np.degrees(np.arctan2(dlat, dlon))
    
    df['center_latitude'] = (df['pickup_latitude'] + df['dropoff_latitude']) / 2
    df['center_longitude'] = (df['pickup_longitude'] + df['dropoff_longitude']) / 2
    
    # Flag
    df['store_and_fwd_flag'] = 1 if data_dict.get('store_and_fwd_flag') == 'Y' else 0
    
    return df

# --- 3. APP & ARTIFACTS ---
app = FastAPI()

try:
    with open('artifacts/model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('artifacts/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    print("✅ Đã load Model & Scaler!")
except:
    print("❌ Lỗi: Chưa có artifacts/model.pkl hoặc artifacts/scaler.pkl")

class TripInput(BaseModel):
    vendor_id: int
    pickup_datetime: str
    passenger_count: int
    pickup_longitude: float
    pickup_latitude: float
    dropoff_longitude: float
    dropoff_latitude: float
    store_and_fwd_flag: Optional[str] = "N"

@app.get("/", response_class=HTMLResponse)
async def home():
    p = Path("app/templates/index.html")
    return p.read_text(encoding="utf-8") if p.exists() else "Error"

@app.post("/predict")
async def predict(trip: TripInput):
    try:
        # 1. Feature Engineering
        df = transform_raw_data(trip.dict())
        
        # 2. Fill missing cols = 0
        for col in ALL_FEATURES:
            if col not in df.columns:
                df[col] = 0
        
        # 3. Sắp xếp cột & Tạo bản sao (FIX LỖI WARNING Ở ĐÂY)
        df_final = df[ALL_FEATURES].copy()
        
        # In ra khoảng cách thô để kiểm tra
        raw_dist = df_final['distance_km'].values[0]
        print(f"\n📏 Khoảng cách tính toán (Raw): {raw_dist:.4f} km")

        # 4. Scaling
        try:
            df_subset = df_final[SCALED_FEATURES]
            scaled_values = scaler.transform(df_subset)
            df_final[SCALED_FEATURES] = scaled_values
        except Exception as e:
            return {"success": False, "detail": f"Lỗi Scaler: {e}"}
        
        print(f"🔍 CHI TIẾT DỮ LIỆU ĐẦU VÀO MODEL ({len(df_final.columns)} cột)")
        print(f"{'index':<5} | {'feature name':<25} | {'value(scaled)'}")
        print("-" * 50)
        row_values = df_final.iloc[0]
        
        for i, col_name in enumerate(df_final.columns):
            val = row_values[col_name]
            val_str = f"{val:.4f}" if isinstance(val, (int, float)) else str(val)
            print(f"{i:<5} | {col_name:<25} | {val_str}")
            
        print("="*50 + "\n")
        # ----------------------------------------------------

        # 5. Dự đoán
        log_pred = model.predict(df_final)[0]
        seconds = np.expm1(log_pred)
        
        if seconds < 0: seconds = 0
        mins = int(seconds // 60)
        secs = int(seconds % 60)
        
        return {
            "success": True,
            "duration_text": f"{mins} phút {secs} giây",
            "distance_km": round(float(raw_dist), 2),
            "is_rush_hour": bool(df['is_rush_hour'].values[0])
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"success": False, "detail": str(e)}