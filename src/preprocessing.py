import numpy as np
import pandas as pd
from datetime import datetime

def haversine_distance(lat1, lon1, lat2, lon2):
    """Tính khoảng cách Haversine"""
    R = 6378.137
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    return R * (2 * np.arctan2(np.sqrt(a), np.sqrt(1-a)))

def feature_engineering(data):
    """
    Feature engineering - TẠO ĐÚNG 20 FEATURES (KHÔNG CÓ pickup_year)
    
    OUTPUT (20 features theo thứ tự):
        0. vendor_id
        1. passenger_count
        2. pickup_longitude
        3. pickup_latitude
        4. dropoff_longitude
        5. dropoff_latitude
        6. store_and_fwd_flag
        7. pickup_month
        8. pickup_day
        9. pickup_hour
        10. pickup_minute
        11. pickup_weekday
        12. pickup_yday
        13. pickup_weekend
        14. is_rush_hour
        15. is_night
        16. distance_km
        17. direction
        18. center_latitude
        19. center_longitude
    """
    # Convert to DataFrame nếu là dict
    if isinstance(data, dict):
        df = pd.DataFrame([data])
    else:
        df = data.copy()
    
    # Parse datetime string thành datetime object
    pickup_datetime = pd.to_datetime(df['pickup_datetime'].iloc[0])
    
    # Time features (KHÔNG CÓ pickup_year)
    df['pickup_month'] = pickup_datetime.month
    df['pickup_day'] = pickup_datetime.day
    df['pickup_hour'] = pickup_datetime.hour
    df['pickup_minute'] = pickup_datetime.minute
    df['pickup_weekday'] = pickup_datetime.weekday()
    df['pickup_yday'] = pickup_datetime.timetuple().tm_yday
    df['pickup_weekend'] = 1 if pickup_datetime.weekday() >= 5 else 0
    
    # Time categories
    hour = pickup_datetime.hour
    df['is_rush_hour'] = 1 if ((hour >= 7 and hour <= 9) or (hour >= 17 and hour <= 19)) else 0
    df['is_night'] = 1 if (hour >= 22 or hour <= 5) else 0
    
    # Spatial features - lấy giá trị đầu tiên
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
    
    # Sắp xếp lại theo đúng thứ tự 20 cột
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
    
    # Đảm bảo có đủ và đúng thứ tự
    df = df[ordered_columns]
    
    return df