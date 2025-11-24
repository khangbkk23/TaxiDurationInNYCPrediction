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
from src.preprocessing import feature_engineering


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
    print("Lỗi: Chưa có artifacts/model.pkl hoặc artifacts/scaler.pkl")
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
        
        df = feature_engineering(data_dict)
        
        df = df[feature_names].copy()

        NUMERICAL_COLS = [col for col in scaler.feature_names_in_ if col in feature_names]
        df_scaled = df.copy()
        df_scaled[NUMERICAL_COLS] = scaler.transform(df[NUMERICAL_COLS])
            
        print("\n=== DEBUG API vs CHECK ===")
        for i, col in enumerate(df_scaled.columns):
            print(f"{i:02d} | {col:22s} | {df_scaled[col].values[0]:.4f}")
        print("====================================")

        log_pred = model.predict(df_scaled)[0]
        seconds = float(np.expm1(log_pred))
        if seconds < 0:
            seconds = 0.0

        mins = int(seconds // 60)
        secs = int(seconds % 60)

        raw_dist = float(df['distance_km'].values[0])
        is_rush = bool(df['is_rush_hour'].values[0])

        print("=== DEBUG PREDICTION (API) ===")
        print(f"log_pred (API)    : {log_pred}")
        print(f"seconds (API)     : {seconds}")
        print(f"duration_text(API): {mins} phút {secs} giây")
        print("====================================")

        return {
            "success": True,
            "duration_text": f"{mins} phút {secs} giây",
            "distance_km": round(raw_dist, 2),
            "is_rush_hour": is_rush
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"success": False, "detail": str(e)}
