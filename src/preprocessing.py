"""
Preprocessing functions
"""
import numpy as np
import pandas as pd
from datetime import datetime

def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6378.137
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    return R * (2 * np.arctan2(np.sqrt(a), np.sqrt(1-a)))

def engineer_features(data):
    df = pd.DataFrame([data]) if isinstance(data, dict) else data.copy()

    pickup_dt = pd.to_datetime(df['pickup_datetime'])

    # Time features
    df['pickup_year'] = pickup_dt.dt.year
    df['pickup_month'] = pickup_dt.dt.month
    df['pickup_day'] = pickup_dt.dt.day
    df['pickup_hour'] = pickup_dt.dt.hour
    df['pickup_minute'] = pickup_dt.dt.minute
    df['pickup_weekday'] = pickup_dt.dt.weekday
    df['pickup_yday'] = pickup_dt.dt.dayofyear
    df['pickup_weekend'] = (pickup_dt.dt.weekday >= 5).astype(int)

    # Time categories
    df['is_rush_hour'] = (
        ((df['pickup_hour'] >= 7) & (df['pickup_hour'] <= 9)) |
        ((df['pickup_hour'] >= 17) & (df['pickup_hour'] <= 19))
    ).astype(int)

    df['is_night'] = (
        (df['pickup_hour'] >= 22) | (df['pickup_hour'] <= 5)
    ).astype(int)

    df['distance_km'] = haversine_distance(
        df['pickup_latitude'], df['pickup_longitude'],
        df['dropoff_latitude'], df['dropoff_longitude']
    )

    df['direction'] = np.degrees(np.arctan2(
        df['dropoff_latitude'] - df['pickup_latitude'],
        df['dropoff_longitude'] - df['pickup_longitude']
    ))

    df['center_latitude'] = (df['pickup_latitude'] + df['dropoff_latitude']) / 2
    df['center_longitude'] = (df['pickup_longitude'] + df['dropoff_longitude']) / 2

    if 'store_and_fwd_flag' in df.columns:
        df['store_and_fwd_flag'] = (df['store_and_fwd_flag'] == 'Y').astype(int)
    else:
        df['store_and_fwd_flag'] = 0

    if 'pickup_datetime' in df.columns:
        df = df.drop(columns=['pickup_datetime'])

    return df