import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
features = [
    'vendor_id', 'passenger_count', 
    'pickup_longitude', 'pickup_latitude', 
    'dropoff_longitude', 'dropoff_latitude', 
    'store_and_fwd_flag', 'pickup_month', 'pickup_day', 'pickup_hour', 
    'pickup_minute', 'pickup_weekday', 'pickup_yday', 'pickup_weekend',
    'is_rush_hour', 'is_night', 
    'distance_km', 'direction', 
    'center_latitude', 'center_longitude'
]

numerical_cols = [
    'vendor_id', 'passenger_count',
    'pickup_longitude', 'pickup_latitude',
    'dropoff_longitude', 'dropoff_latitude',
    'pickup_hour', 'pickup_weekday', 'pickup_month',
    'distance_km', 'direction',
    'center_latitude', 'center_longitude'
]

def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6378.137 # Earth radius in kilometres
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0)**2
    return R * (2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a)))

def feature_engineering(df):
    df = df.copy()

    # Time
    pickup_datetime = pd.to_datetime(df['pickup_datetime'])
    df['pickup_year'] = pickup_datetime.dt.year
    df['pickup_month'] = pickup_datetime.dt.month
    df['pickup_day'] = pickup_datetime.dt.day
    df['pickup_hour'] = pickup_datetime.dt.hour
    df['pickup_minute'] = pickup_datetime.dt.minute
    df['pickup_weekday'] = pickup_datetime.dt.weekday
    df['pickup_yday'] = pickup_datetime.dt.dayofyear
    df['pickup_weekend'] = (pickup_datetime.dt.weekday >= 5).astype(int)

    # Time category
    df['is_rush_hour'] = (((df['pickup_hour'] >= 7) & (df['pickup_hour'] <= 9)) |
                          ((df['pickup_hour'] >= 17) & (df['pickup_hour'] <= 19))).astype(int)
    df['is_night'] = ((df['pickup_hour'] >= 22) | (df['pickup_hour'] <= 5)).astype(int)

    # Calculate the distance with Haversine fomula
    df['distance_km'] = haversine_distance(
        df['pickup_latitude'], df['pickup_longitude'],
        df['dropoff_latitude'], df['dropoff_longitude']
    )

    df['direction'] = np.degrees(np.arctan2(
        (df['dropoff_latitude'] - df['pickup_latitude']),
        (df['dropoff_longitude'] - df['pickup_longitude'])
    ))

    # Center coordinates
    df['center_latitude'] = (df['pickup_latitude'] + df['dropoff_latitude']) / 2
    df['center_longitude'] = (df['pickup_longitude'] + df['dropoff_longitude']) / 2

    # Binary categorical
    df['store_and_fwd_flag'] = (df['store_and_fwd_flag'] == 'Y').astype(int)

    return df

def clean_data(df):
    df = df.copy()

    print(f"Initial shape: {df.shape}")

    if 'trip_duration' in df.columns:
        df = df[(df['trip_duration'] > 30) & (df['trip_duration'] < 3600 * 6)]
        print(f"After duration filter: {df.shape}")

    # Remove zero-distance trips
    df = df[df['distance_km'] > 0]
    print(f"After distance filter: {df.shape}")

    # Geographic bounding box
    bounds = {'min_lat': 40.5, 'max_lat': 41.0, 'min_lon': -74.3, 'max_lon': -73.7}
    df = df[
        (df['pickup_latitude'].between(bounds['min_lat'], bounds['max_lat'])) &
        (df['pickup_longitude'].between(bounds['min_lon'], bounds['max_lon'])) &
        (df['dropoff_latitude'].between(bounds['min_lat'], bounds['max_lat'])) &
        (df['dropoff_longitude'].between(bounds['min_lon'], bounds['max_lon']))
    ]

    print(f"After geographic filter: {df.shape}")
    return df

def preprocessing(df: pd.DataFrame, scale=True, random_state=42) -> tuple:
    
    # Geospatial feature engineering
    df = feature_engineering(df)
    
    # Data cleaning
    df = clean_data(df)

    # Target transformation
    y = None
    if 'trip_duration' in df.columns:
        y = np.log1p(df['trip_duration'])

    X = df[features].copy()
    if y is not None:
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=random_state)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=random_state)
    else:
        return X, None, None, None, None, None, None

    # Feature scaling
    scaler = None
    if scale:
        # Only scale columns that exist
        scaler = StandardScaler()
        valid_cols = [c for c in numerical_cols if c in X_train.columns]
        
        X_train[valid_cols] = scaler.fit_transform(X_train[valid_cols])
        X_val[valid_cols] = scaler.transform(X_val[valid_cols])
        X_test[valid_cols] = scaler.transform(X_test[valid_cols])

    print("Preprocessing completed.")
    print(f"Features used: {len(features)}")

    return X_train, X_val, X_test, y_train, y_val, y_test, scaler