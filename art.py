import pickle
import pandas as pd
import numpy as np
from pathlib import Path

# --- Load artifacts ---
BASE_DIR = Path(__file__).resolve().parent
ARTIFACT_DIR = BASE_DIR / "artifacts"

with open(ARTIFACT_DIR / "model.pkl", "rb") as f:
    model = pickle.load(f)

with open(ARTIFACT_DIR / "scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# --- Dummy input (tương tự API) ---
data_dict = {
    "vendor_id": 2,
    "pickup_datetime": "2016-06-15 10:30:00",
    "passenger_count": 2,
    "pickup_longitude": -73.9776,
    "pickup_latitude": 40.7614,
    "dropoff_longitude": -73.99,
    "dropoff_latitude": 40.75,
    "store_and_fwd_flag": "N"
}

# --- Feature engineering giống main.py ---
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
    df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'])
    df['pickup_month'] = df['pickup_datetime'].dt.month
    df['pickup_day'] = df['pickup_datetime'].dt.day
    df['pickup_hour'] = df['pickup_datetime'].dt.hour
    df['pickup_minute'] = df['pickup_datetime'].dt.minute
    df['pickup_weekday'] = df['pickup_datetime'].dt.weekday
    df['pickup_yday'] = df['pickup_datetime'].dt.dayofyear
    df['pickup_weekend'] = (df['pickup_weekday'] >= 5).astype(int)
    df['is_rush_hour'] = (
        ((df['pickup_hour'] >= 7) & (df['pickup_hour'] <= 9)) |
        ((df['pickup_hour'] >= 17) & (df['pickup_hour'] <= 19))
    ).astype(int)
    df['is_night'] = ((df['pickup_hour'] >= 22) | (df['pickup_hour'] <= 5)).astype(int)
    df['distance_km'] = haversine_array(
        df['pickup_latitude'], df['pickup_longitude'],
        df['dropoff_latitude'], df['dropoff_longitude']
    )
    dlon = df['dropoff_longitude'] - df['pickup_longitude']
    dlat = df['dropoff_latitude'] - df['pickup_latitude']
    df['direction'] = np.degrees(np.arctan2(dlat, dlon))
    df['center_latitude'] = (df['pickup_latitude'] + df['dropoff_latitude']) / 2
    df['center_longitude'] = (df['pickup_longitude'] + df['dropoff_longitude']) / 2
    df['store_and_fwd_flag'] = 1 if data_dict.get('store_and_fwd_flag') == 'Y' else 0
    return df

ALL_FEATURES = [
    'vendor_id', 'passenger_count', 
    'pickup_longitude', 'pickup_latitude',
    'dropoff_longitude', 'dropoff_latitude', 'store_and_fwd_flag',
    'pickup_month', 'pickup_day', 'pickup_hour',
    'pickup_minute', 'pickup_weekday', 'pickup_yday', 'pickup_weekend',
    'is_rush_hour', 'is_night',
    'distance_km', 'direction',
    'center_latitude', 'center_longitude'
]

SCALED_FEATURES = [
    'vendor_id', 'passenger_count', 'pickup_longitude', 'pickup_latitude', 
    'dropoff_longitude', 'dropoff_latitude', 'pickup_hour', 
    'pickup_weekday', 'pickup_month', 'distance_km', 'direction', 
    'center_latitude', 'center_longitude'
]

df = transform_raw_data(data_dict)

# --- Check missing columns ---
for col in ALL_FEATURES:
    if col not in df.columns:
        df[col] = 0

df_final = df[ALL_FEATURES]

print("=== Debug Artifacts Check ===")
print("DF shape:", df_final.shape)
print("Model expects n_features_in_:", model.n_features_in_)
print("Scaler mean shape:", getattr(scaler, 'mean_', 'No mean found'))

# --- Print values before scaling ---
print("\nInput features before scaling:")
for i, col in enumerate(df_final.columns):
    val = df_final[col].values[0]
    print(f"{i:02d} | {col:<20} | {val}")

# --- Try scaling ---
try:
    scaled_values = scaler.transform(df_final[SCALED_FEATURES])
    print("\nScaled values:")
    for i, col in enumerate(SCALED_FEATURES):
        print(f"{i:02d} | {col:<20} | {scaled_values[0, i]:.4f}")
except Exception as e:
    print("Scaler error:", e)

# --- Try prediction ---
try:
    log_pred = model.predict(df_final)[0]
    seconds = np.expm1(log_pred)
    print(f"\nPrediction: log_pred={log_pred:.4f}, seconds={seconds:.2f}")
except Exception as e:
    print("Model prediction error:", e)
