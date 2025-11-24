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

def feature_engineering(data):
    if isinstance(data, dict):
        df = pd.DataFrame([data])
    else:
        df = data.copy()

    pickup_datetime = pd.to_datetime(df['pickup_datetime'].iloc[0])
    df['pickup_month'] = pickup_datetime.month
    df['pickup_day'] = pickup_datetime.day
    df['pickup_hour'] = pickup_datetime.hour
    df['pickup_minute'] = pickup_datetime.minute
    df['pickup_weekday'] = pickup_datetime.weekday()
    df['pickup_yday'] = pickup_datetime.timetuple().tm_yday
    df['pickup_weekend'] = 1 if pickup_datetime.weekday() >= 5 else 0
    
    hour = pickup_datetime.hour
    df['is_rush_hour'] = 1 if ((hour >= 7 and hour <= 9) or (hour >= 17 and hour <= 19)) else 0
    df['is_night'] = 1 if (hour >= 22 or hour <= 5) else 0

    pickup_lat = df['pickup_latitude'].iloc[0]
    pickup_lon = df['pickup_longitude'].iloc[0]
    dropoff_lat = df['dropoff_latitude'].iloc[0]
    dropoff_lon = df['dropoff_longitude'].iloc[0]
    
    df['distance_km'] = haversine_distance(pickup_lat, pickup_lon, dropoff_lat, dropoff_lon)
    
    df['direction'] = np.degrees(np.arctan2(
        dropoff_lat - pickup_lat,
        dropoff_lon - pickup_lon
    ))
    
    df['center_latitude'] = (pickup_lat + dropoff_lat) / 2
    df['center_longitude'] = (pickup_lon + dropoff_lon) / 2
    
    # Binary flag
    if 'store_and_fwd_flag' in df.columns:
        df['store_and_fwd_flag'] = 1 if df['store_and_fwd_flag'].iloc[0] == 'Y' else 0
    else:
        df['store_and_fwd_flag'] = 0
    
    # Drop datetime
    if 'pickup_datetime' in df.columns:
        df = df.drop(columns=['pickup_datetime'])
    
    ordered_columns = [
        'vendor_id',
        'passenger_count',
        'pickup_longitude',
        'pickup_latitude',
        'dropoff_longitude',
        'dropoff_latitude',
        'store_and_fwd_flag',
        'pickup_month',
        'pickup_day',
        'pickup_hour',
        'pickup_minute',
        'pickup_weekday',
        'pickup_yday',
        'pickup_weekend',
        'is_rush_hour',
        'is_night',
        'distance_km',
        'direction',
        'center_latitude',
        'center_longitude'
    ]
    
    df = df[ordered_columns]
    
    return df

def clean_data_not_drop(df, is_train=True):
    df = df.copy()
    print(f"Initial shape: {df.shape}")

    if is_train and 'trip_duration' in df.columns:
        df = df[(df['trip_duration'] > 30) & (df['trip_duration'] < 3600 * 6)]
        print(f"After duration filter: {df.shape}")

    if is_train:
        df = df[df['distance_km'] > 0]
        print(f"After distance filter: {df.shape}")
    else:
        df.loc[df['distance_km'] <= 0, 'distance_km'] = 0.1

    bounds = {'min_lat': 40.5, 'max_lat': 41.0, 'min_lon': -74.3, 'max_lon': -73.7}
    if is_train:
        df = df[
            (df['pickup_latitude'].between(bounds['min_lat'], bounds['max_lat'])) &
            (df['pickup_longitude'].between(bounds['min_lon'], bounds['max_lon'])) &
            (df['dropoff_latitude'].between(bounds['min_lat'], bounds['max_lat'])) &
            (df['dropoff_longitude'].between(bounds['min_lon'], bounds['max_lon']))
        ]
        print(f"After geographic filter: {df.shape}")
    else:
        df['pickup_latitude'] = df['pickup_latitude'].clip(bounds['min_lat'], bounds['max_lat'])
        df['pickup_longitude'] = df['pickup_longitude'].clip(bounds['min_lon'], bounds['max_lon'])
        df['dropoff_latitude'] = df['dropoff_latitude'].clip(bounds['min_lat'], bounds['max_lat'])
        df['dropoff_longitude'] = df['dropoff_longitude'].clip(bounds['min_lon'], bounds['max_lon'])

    print(f"After cleaning ({'train' if is_train else 'test'}): {df.shape}")
    return df
