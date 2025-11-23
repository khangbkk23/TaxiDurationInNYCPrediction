import pickle
import pandas as pd
import numpy as np
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.preprocessing import feature_engineering

print("\n1️⃣  LOAD MODEL ARTIFACTS")
with open('artifacts/features.pkl', 'rb') as f:
    feature_names = pickle.load(f)

with open('artifacts/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('artifacts/model.pkl', 'rb') as f:
    model = pickle.load(f)

print(f"Model type: {type(model).__name__}")
print(f"Feature names: {len(feature_names)} columns")

print("\n2️⃣  KIỂM TRA SCALER")
if hasattr(scaler, 'mean_'):
    print(f"Scaler đã được fit")
    print(f"Number of features in scaler: {len(scaler.mean_)}")
else:
    print("Scaler chưa được fit hoặc không phải StandardScaler")

print("\n3️⃣  TEST VỚI DỮ LIỆU MẪU (từ web form)")
sample_data = {
    "vendor_id": 2,
    "pickup_datetime": "2016-06-15 10:30:00",
    "passenger_count": 2,
    "pickup_longitude": -73.9776,
    "pickup_latitude": 40.7614,
    "dropoff_longitude": -73.9900,
    "dropoff_latitude": 40.7500,
    "store_and_fwd_flag": "N"
}

print("Input:")
for key, val in sample_data.items():
    print(f"  {key:20s} = {val}")

print("\n4️⃣  CHẠY FEATURE ENGINEERING")
df = feature_engineering(sample_data)

print(f"Output: {len(df.columns)} columns")
print(f"\nChi tiết từng feature (CHƯA SCALE):")
for i, col in enumerate(df.columns):
    val = df[col].values[0]
    print(f"  [{i:2d}] {col:20s} = {val:.4f}")

# 5. So sánh với feature_names
print("\n5️⃣  SO SÁNH VỚI MODEL")
print("-"*70)
if list(df.columns) == feature_names:
    print("Thứ tự columns không chính xác!")
else:
    print("Thứ tự columns sai!")
    print("\nExpected:")
    for i, col in enumerate(feature_names):
        print(f"  [{i:2d}] {col}")
    print("\nGot:")
    for i, col in enumerate(df.columns):
        print(f"  [{i:2d}] {col}")

# 6. Scaling
print("\n6️⃣  SCALING")
NUMERICAL_COLS = ['vendor_id', 'passenger_count', 'pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude', 'pickup_hour', 'pickup_weekday', 'pickup_month', 'distance_km', 'direction', 'center_latitude', 'center_longitude']

print(f"Sẽ scale {len(NUMERICAL_COLS)} cột:")
for col in NUMERICAL_COLS:
    print(f"  - {col}")

# Scale
df_scaled = df.copy()
df_scaled[NUMERICAL_COLS] = scaler.transform(df[NUMERICAL_COLS])

print(f"\nChi tiết sau khi SCALE:")
print("-"*70)
print(f"{'Index':<6} {'Feature':<22} {'Original':<12} {'Scaled':<12}")
print("-"*70)
for i, col in enumerate(df.columns):
    original = df[col].values[0]
    scaled = df_scaled[col].values[0]
    print(f"{i:<6} {col:<22} {original:<12.4f} {scaled:<12.4f}")

# 7. Prediction
print("\n7️⃣  PREDICTION")
print("-"*70)
try:
    log_pred = model.predict(df_scaled)[0]
    duration_seconds = np.expm1(log_pred)
    duration_minutes = duration_seconds / 60
    
    print(f"PREDICTION SUCCESS!")
    print(f"\nKết quả:")
    print(f"  Log prediction    : {log_pred:.4f}")
    print(f"  Duration (seconds): {duration_seconds:.2f}")
    print(f"  Duration (minutes): {duration_minutes:.2f}")
    print(f"  Duration (text)   : {int(duration_minutes)} phút {int(duration_seconds % 60)} giây")
    print(f"  Distance (km)     : {df['distance_km'].values[0]:.2f}")
    
except Exception as e:
    print(f"PREDICTION FAILED!")
    print(f"Error: {str(e)}")
    import traceback
    traceback.print_exc()

print("\n" + "="*70)