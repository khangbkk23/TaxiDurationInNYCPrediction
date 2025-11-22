import pandas as pd
import numpy as np

def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6378.137
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    return R * (2 * np.arctan2(np.sqrt(a), np.sqrt(1-a)))

def preprocess_data(df, is_train=True):
    data = df.copy()
    
    data['pickup_datetime'] = pd.to_datetime(data['pickup_datetime'])
    data['pickup_year'] = data['pickup_datetime'].dt.year
    data['pickup_month'] = data['pickup_datetime'].dt.month
    data['pickup_day'] = data['pickup_datetime'].dt.day
    data['pickup_hour'] = data['pickup_datetime'].dt.hour
    data['pickup_minute'] = data['pickup_datetime'].dt.minute
    data['pickup_weekday'] = data['pickup_datetime'].dt.weekday
    data['pickup_yday'] = data['pickup_datetime'].dt.dayofyear
    
    # Đặc trưng phát
    data['pickup_weekend'] = (data['pickup_weekday'] >= 5).astype(int)
    data['is_rush_hour'] = (((data['pickup_hour'] >= 7) & (data['pickup_hour'] <= 9)) |
                            ((data['pickup_hour'] >= 17) & (data['pickup_hour'] <= 19))).astype(int)
    data['is_night'] = ((data['pickup_hour'] >= 22) | (data['pickup_hour'] <= 5)).astype(int)
    
    # Khoảng cách
    data['distance_km'] = haversine_distance(
        data['pickup_latitude'], data['pickup_longitude'],
        data['dropoff_latitude'], data['dropoff_longitude']
    )
    
    data['direction'] = np.degrees(np.arctan2(
        data['dropoff_latitude'] - data['pickup_latitude'],
        data['dropoff_longitude'] - data['pickup_longitude']
    ))
    
    data['center_latitude'] = (data['pickup_latitude'] + data['dropoff_latitude']) / 2
    data['center_longitude'] = (data['pickup_longitude'] + data['dropoff_longitude']) / 2
    
    # Xử lý store_and_fwd_flag
    if 'store_and_fwd_flag' in data.columns:
        data['store_and_fwd_flag'] = (data['store_and_fwd_flag'] == 'Y').astype(int)
    if is_train:
        if 'trip_duration' in data.columns:
            data = data[(data['trip_duration'] > 30) & (data['trip_duration'] < 3600*6)]
            pass 
        
    cols_to_drop = ['pickup_datetime', 'dropoff_datetime']
    if is_train:
        cols_to_drop.extend(['id', 'trip_duration'])
        
    data = data.drop(columns=[c for c in cols_to_drop if c in data.columns], errors='ignore')
    
    return data