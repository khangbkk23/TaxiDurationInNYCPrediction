from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import Optional
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.preprocessing import preprocess_data

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

app = FastAPI()

print("Loading artifacts...")
try:
    with open('artifacts/best_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('artifacts/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    print("Đã load Model và Scaler thành công!")
except Exception as e:
    print(f"Lỗi load artifacts: {e}")
    model = None
    scaler = None

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
    return p.read_text(encoding="utf-8") if p.exists() else "<h1>File not found</h1>"

@app.post("/predict")
async def predict(trip: TripInput):
    if not model or not scaler:
        return {"success": False, "detail": "Model chưa được load"}

    try:
        data_dict = trip.dict()
        df_input = pd.DataFrame([data_dict])
        
        df = preprocess_data(df_input, is_train=False)
        for col in ALL_FEATURES:
            if col not in df.columns:
                df[col] = 0
        
        df = df[ALL_FEATURES]
        raw_debug = df.iloc[0].to_dict()
        try:
            # Lấy cột cần scale
            df_subset = df[SCALED_FEATURES]
            # Transform
            scaled_values = scaler.transform(df_subset)
            # Gán ngược lại
            df[SCALED_FEATURES] = scaled_values
        except Exception as e:
            return {"success": False, "detail": f"Lỗi Scaler: {str(e)}"}

        scaled_debug = df.values[0].tolist()
        log_pred = model.predict(df)[0]
        seconds = np.expm1(log_pred)
        
        if seconds < 0: seconds = 0
        mins = int(seconds // 60)
        secs = int(seconds % 60)

        return {
            "success": True,
            "duration_text": f"{mins} phút {secs} giây",
            "distance_km": round(float(df['distance_km'].values[0]), 2),
            "is_rush_hour": bool(raw_debug.get('is_rush_hour', False)),
            "debug_info": {
                "raw_features": raw_debug,
                "scaled_features": scaled_debug,
                "model_input_order": ALL_FEATURES
            }
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"success": False, "detail": str(e)}