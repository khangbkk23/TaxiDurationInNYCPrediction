"""
FastAPI Application - ĐƠN GIẢN
"""
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from typing import Optional
import pickle
import numpy as np
import os
from pathlib import Path

from src.preprocessing import engineer_features

app = FastAPI(
    title="NYC Taxi Duration Prediction",
    description="Machine Learning",
    version="1.0.0"
)

print("Loading model...")
with open('artifacts/model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('artifacts/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('artifacts/feature_names.pkl', 'rb') as f:
    feature_names = pickle.load(f)

NUMERICAL_COLS = [
    'vendor_id', 'passenger_count', 'pickup_longitude', 'pickup_latitude',
    'dropoff_longitude', 'dropoff_latitude', 'pickup_hour', 'pickup_weekday',
    'pickup_month', 'distance_km', 'direction', 'center_latitude', 'center_longitude'
]

print("Model loaded!")

# Pydantic models
class TripInput(BaseModel):
    vendor_id: int = Field(..., ge=1, le=2)
    pickup_datetime: str = Field(..., example="2016-06-15 10:30:00")
    passenger_count: int = Field(..., ge=1, le=6)
    pickup_longitude: float
    pickup_latitude: float
    dropoff_longitude: float
    dropoff_latitude: float
    store_and_fwd_flag: Optional[str] = "N"

# Trang chủ - Serve HTML
@app.get("/", response_class=HTMLResponse)
async def home():
    html_file = Path("app/templates/index.html")
    if html_file.exists():
        return html_file.read_text(encoding='utf-8')
    return "<h1>API is running! Go to <a href='/docs'>/docs</a></h1>"

# API dự đoán
@app.post("/predict")
async def predict(trip: TripInput):
    try:
        # Chuyển sang dict
        data = trip.dict()

        # Feature engineering
        df = engineer_features(data)
        df = df[feature_names]

        df_scaled = df.copy()
        df_scaled[NUMERICAL_COLS] = scaler.transform(df[NUMERICAL_COLS])

        log_prediction = model.predict(df_scaled)[0]
        duration_seconds = np.expm1(log_prediction)
        duration_minutes = duration_seconds / 60

        mins = int(duration_minutes)
        secs = int(duration_seconds % 60)
        duration_text = f"{mins} phút {secs} giây"

        return {
            "success": True,
            "duration_seconds": round(float(duration_seconds), 2),
            "duration_minutes": round(float(duration_minutes), 2),
            "duration_text": duration_text,
            "distance_km": round(float(df['distance_km'].values[0]), 2),
            "is_rush_hour": bool(df['is_rush_hour'].values[0]),
            "is_weekend": bool(df['pickup_weekend'].values[0])
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e

# Health check
@app.get("/health")
async def health():
    """Kiểm tra API"""
    return {
        "status": "healthy",
        "model_loaded": model is not None
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)