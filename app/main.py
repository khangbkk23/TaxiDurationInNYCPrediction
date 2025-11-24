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

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))
from src.preprocessing import feature_engineering, clean_data_not_drop


# NUMERICAL_COLS = [
#     'vendor_id', 'passenger_count',
#     'pickup_longitude', 'pickup_latitude',
#     'dropoff_longitude', 'dropoff_latitude',
#     'pickup_hour', 'pickup_weekday', 'pickup_month',
#     'distance_km', 'direction', 'center_latitude', 'center_longitude'
# ]


app = FastAPI()

try:
    ARTIFACT_DIR = BASE_DIR / "artifacts"
    with open(ARTIFACT_DIR / "model.pkl", "rb") as f:
        model = pickle.load(f)
    with open(ARTIFACT_DIR / "scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open(ARTIFACT_DIR / "features.pkl", "rb") as f:
        feature_names = pickle.load(f)
except Exception as e:
    print("Lỗi: Tải lên thư mục lưu trữ không thành công")
    print(e)

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
        if model is None or scaler is None:
            raise HTTPException(status_code=500, detail="Model or scaler not loaded")
        
        data_dict = trip.dict()
        
        # 1️⃣ CHỈ CẦN feature engineering - KHÔNG clean_data
        df = feature_engineering(data_dict)
        # ❌ KHÔNG GỌI clean_data_not_drop ở đây!

        # 2️⃣ Fill các cột thiếu
        for col in feature_names:
            if col not in df.columns:
                df[col] = 0

        # 3️⃣ Sắp xếp theo feature_names TRƯỚC KHI scale
        df = df[feature_names]
        
        # DEBUG: In ra giá trị TRƯỚC khi scale
        print("\n=== TRƯỚC KHI SCALE ===")
        for i, col in enumerate(df.columns):
            print(f"{i:02d} | {col:22s} | {df[col].values[0]:.4f}")
        
        # 4️⃣ Scale numeric columns
        numeric_cols_in_df = [c for c in scaler.feature_names_in_ if c in df.columns]
        df[numeric_cols_in_df] = scaler.transform(df[numeric_cols_in_df])
        
        # DEBUG: In ra giá trị SAU khi scale
        print("\n=== SAU KHI SCALE ===")
        for i, col in enumerate(df.columns):
            print(f"{i:02d} | {col:22s} | {df[col].values[0]:.4f}")

        # 5️⃣ Predict
        log_pred = model.predict(df)[0]
        seconds = float(np.expm1(log_pred))
        if seconds < 0:
            seconds = 0.0

        mins = int(seconds // 60)
        secs = int(seconds % 60)

        print("\n=== PREDICTION ===")
        print(f"log_pred  : {log_pred:.4f}")
        print(f"seconds   : {seconds:.2f}")
        print(f"duration  : {mins} phút {secs} giây")
        print("==================")

        return {
            "success": True,
            "duration_text": f"{mins} phút {secs} giây",
            "distance_km": round(float(df['distance_km'].values[0]), 2),
            "is_rush_hour": bool(df['is_rush_hour'].values[0])
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"success": False, "detail": str(e)}