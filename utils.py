"""
utils.py - Data Preprocessing Utilities for Traffic Congestion Prediction

This module handles:
- Timestamp parsing and feature extraction
- Data normalization/denormalization
- Sequence creation for LSTM input
- Sample data generation
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Sequence length (number of past time steps to use for prediction)
SEQUENCE_LENGTH = 24

# Congestion thresholds (for classifying predictions)
CONGESTION_THRESHOLDS = {
    'low': 0.33,
    'medium': 0.66
}


def parse_timestamp(timestamp):
    """
    Extract hour and day features from timestamp string.
    
    Args:
        timestamp: String in format 'YYYY-MM-DD HH:MM:SS' or 'YYYY-MM-DD'
    
    Returns:
        Dictionary with hour (0-23) and day_of_week (0-6, Monday=0)
    """
    if isinstance(timestamp, str):
        dt = pd.to_datetime(timestamp)
    else:
        dt = timestamp
    
    return {
        'hour': dt.hour,
        'day_of_week': dt.dayofweek,
        'month': dt.month,
        'is_weekend': 1 if dt.dayofweek >= 5 else 0
    }


def extract_time_features(hour, day):
    """
    Extract cyclical features from time data.
    Cyclical encoding helps the model understand time relationships.
    
    Args:
        hour: Hour of day (0-23)
        day: Day of week (0-6)
    
    Returns:
        Array of cyclical features
    """
    hour_sin = np.sin(2 * np.pi * hour / 24)
    hour_cos = np.cos(2 * np.pi * hour / 24)
    day_sin = np.sin(2 * np.pi * day / 7)
    day_cos = np.cos(2 * np.pi * day / 7)
    
    return [hour_sin, hour_cos, day_sin, day_cos]


def normalize_data(data):
    """
    Normalize data to range [0, 1] using MinMaxScaler.
    
    Args:
        data: Array-like data
    
    Returns:
        Tuple of (scaled_data, scaler)
    """
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data.reshape(-1, 1))
    return scaled_data.flatten(), scaler


def denormalize_data(data, scaler):
    """
    Convert normalized data back to original scale.
    
    Args:
        data: Normalized array
        scaler: Fitted MinMaxScaler
    
    Returns:
        Data in original scale
    """
    return scaler.inverse_transform(data.reshape(-1, 1)).flatten()


def create_sequences(data, sequence_length):
    """
    Create input sequences and targets for LSTM training.
    
    LSTM requires sequences of data points to predict the next value.
    This function creates sliding windows over the data.
    
    Example with sequence_length=3:
    Input: [1, 2, 3, 4, 5]
    Sequences (X): [[1,2,3], [2,3,4]] -> shape: (2, 3, 1)
    Targets (y): [4, 5] -> shape: (2, 1)
    
    Args:
        data: Array of traffic values
        sequence_length: Number of past values to use
    
    Returns:
        Tuple of (X, y) arrays
    """
    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data[i:i + sequence_length])
        y.append(data[i + sequence_length])
    
    return np.array(X), np.array(y)


def generate_sample_data(num_days=30, samples_per_day=24):
    """
    Generate synthetic traffic data for demonstration.
    Realistic traffic patterns:
    - Rush hours (7-9 AM, 5-7 PM) have higher traffic
    - Weekends have different patterns
    - Night hours have lower traffic
    
    Args:
        num_days: Number of days to generate
        samples_per_day: Data points per day (default: hourly)
    
    Returns:
        DataFrame with timestamp and vehicle_count
    """
    np.random.seed(42)
    total_samples = num_days * samples_per_day
    
    timestamps = pd.date_range(
        start='2024-01-01',
        periods=total_samples,
        freq='h'
    )
    
    vehicle_counts = []
    
    for ts in timestamps:
        hour = ts.hour
        day_of_week = ts.dayofweek
        is_weekend = day_of_week >= 5
        
        # Base traffic varies by time of day
        if 7 <= hour <= 9:  # Morning rush
            base = 60 + np.random.normal(0, 10)
        elif 17 <= hour <= 19:  # Evening rush
            base = 70 + np.random.normal(0, 12)
        elif 22 <= hour or hour <= 5:  # Night
            base = 10 + np.random.normal(0, 5)
        else:  # Normal hours
            base = 35 + np.random.normal(0, 8)
        
        # Weekend adjustment
        if is_weekend:
            if 10 <= hour <= 16:  # Weekend afternoon
                base = 50 + np.random.normal(0, 10)
            else:
                base = base * 0.7
        
        # Add some daily variation
        base += np.random.normal(0, 5)
        
        # Ensure non-negative
        vehicle_counts.append(max(0, base))
    
    df = pd.DataFrame({
        'timestamp': timestamps,
        'vehicle_count': vehicle_counts
    })
    
    return df


def get_congestion_level(normalized_value):
    """
    Classify congestion level based on normalized traffic value.
    
    Args:
        normalized_value: Traffic value normalized to [0, 1]
    
    Returns:
        String indicating congestion level
    """
    if normalized_value < CONGESTION_THRESHOLDS['low']:
        return 'Low'
    elif normalized_value < CONGESTION_THRESHOLDS['medium']:
        return 'Medium'
    else:
        return 'High'


def prepare_prediction_input(hour, day, past_values, scaler):
    """
    Prepare input data for model prediction.
    
    Args:
        hour: Current hour (0-23)
        day: Day of week (0-6)
        past_values: List of recent traffic values
        scaler: Fitted MinMaxScaler
    
    Returns:
        Prepared input array for model
    """
    # Extract cyclical time features
    time_features = extract_time_features(hour, day)
    
    # Normalize past values
    normalized_past = scaler.transform(np.array(past_values).reshape(-1, 1)).flatten()
    
    # Combine features
    # Shape: (sequence_length + num_time_features,)
    combined = np.concatenate([normalized_past, time_features])
    
    return combined


def apply_heuristic_modifiers(base_pred, season, weather, road_type, special_factors):
    """
    Applies programmatic modifiers to the base LSTM prediction based on selected parameters.
    Returns the adjusted prediction and a highly reactive factor breakdown chart.
    """
    # Base weight scales
    w_time = 70.0
    w_weather = 5.0
    w_season = 5.0
    w_road = 10.0
    w_special = 2.0
    
    weather_mod = 0.0
    season_mod = 0.0
    road_mod = 0.0
    special_mod = 0.0

    # Weather Impacts
    if weather == 'Clear':
        w_weather = 5.0
    elif weather == 'Cloudy':
        weather_mod = 0.05
        w_weather = 12.0
    elif weather == 'Rain':
        weather_mod = 0.15
        w_weather = 25.0
    elif weather == 'Fog':
        weather_mod = 0.10
        w_weather = 18.0
    elif weather == 'Snow':
        weather_mod = 0.30
        w_weather = 35.0

    # Season Impacts
    if season == 'Spring':
        w_season = 5.0
    elif season == 'Summer':
        season_mod = -0.05
        w_season = 10.0 
    elif season == 'Autumn':
        season_mod = 0.02
        w_season = 12.0
    elif season == 'Winter':
        season_mod = 0.08
        w_season = 20.0

    # Road Type Impacts
    if road_type == 'Highway':
        w_road = 10.0
    elif road_type == 'Arterial':
        road_mod = 0.05
        w_road = 15.0
    elif road_type == 'Urban':
        road_mod = 0.15
        w_road = 28.0
    elif road_type == 'Residential':
        road_mod = -0.10
        w_road = 6.0

    # Special Factors Impacts
    if special_factors and len(special_factors) > 0:
        w_special = len(special_factors) * 15.0
        for factor in special_factors:
            if factor == 'Accident':
                special_mod += 0.40
                w_special += 20.0
            elif factor == 'Major event':
                special_mod += 0.25
                w_special += 15.0
            elif factor == 'Construction':
                special_mod += 0.20
                w_special += 10.0
            elif factor == 'School zone':
                special_mod += 0.10
                w_special += 5.0
            elif factor == 'Public holiday':
                special_mod -= 0.20
                w_special += 8.0

    # Time & Day natively fluctuates based on the baseline LSTM predictions.
    # High base_pred = Rush hour or busy workday -> Time/Day has massive attribution.
    # Low base_pred = 3 AM on Sunday -> Time/Day has lower attribution compared to anomalies.
    w_time = 20.0 + (base_pred * 80.0)
    
    # Still slightly push Time attribution out if severe anomalies exist
    severity_sum = w_weather + w_season + w_road + w_special
    w_time = max(10.0, w_time - (severity_sum * 0.4))

    # Assemble raw factor distribution
    factors = {
        'Time of day': w_time,
        'Weather': w_weather,
        'Season': w_season,
        'Road type': w_road,
        'Special factors': w_special
    }
    
    # Calculate final adjusted prediction (clamped to 0.05 - 0.99)
    adjusted_pred = base_pred + weather_mod + season_mod + road_mod + special_mod
    adjusted_pred = max(0.05, min(0.99, adjusted_pred))
    
    # Normalize to exactly 100%
    total = sum(factors.values())
    factors_normalized = {k: round((v / total) * 100) for k, v in factors.items()}
    
    # Clean up minor rounding mismatches to ensure exactly 100
    diff = 100 - sum(factors_normalized.values())
    if diff != 0:
        factors_normalized['Time of day'] += diff
    
    return adjusted_pred, factors_normalized


def calculate_metrics_from_prediction(adjusted_pred, base_free_flow_speed=65.0, base_travel_time=15.0):
    """
    Calculate user-friendly metrics based on the final prediction value [0, 1]
    """
    # Congestion Index is out of 100
    congestion_index = int(adjusted_pred * 100)
    
    # Speed exponentially drops as congestion goes up
    speed_factor = 1.0 - (adjusted_pred ** 1.5)
    speed_factor = max(0.1, speed_factor)  # Cannot go below 10% of speed limit
    avg_speed = int(base_free_flow_speed * speed_factor)
    
    # Travel time exponentially increases
    delay_factor = 1.0 / speed_factor
    travel_time = int(base_travel_time * delay_factor)
    
    delay = travel_time - int(base_travel_time)
    
    return {
        'congestion_index': congestion_index,
        'travel_time': travel_time,
        'avg_speed': avg_speed,
        'delay': delay
    }


def generate_hourly_forecast(base_pred, adjusted_pred):
    """
    Generates a 24-point realistic trajectory array starting low, peaking at rush hour, 
    and factoring in the current heuristic parameters.
    """
    # Standard daily curve (normalized 0-1)
    base_curve = [0.1, 0.05, 0.05, 0.05, 0.05, 0.08, 0.4, 0.85, 0.84, 0.85, 0.5, 0.5, 
                  0.5, 0.5, 0.5, 0.9, 0.89, 0.9, 0.88, 0.3, 0.28, 0.3, 0.2, 0.1]
    
    # We shift this curve up/down proportionally based on how much the prediction deviated from normal
    shift = adjusted_pred - 0.5  # Assume 0.5 is average
    
    forecast = []
    for val in base_curve:
        new_val = max(0.05, min(0.95, val + shift))
        forecast.append(int(new_val * 100))
        
    return forecast


def generate_weekly_pattern():
    """Generates standard weekly pattern data"""
    return [90, 90, 90, 90, 90, 55, 55]


# Standalone test
if __name__ == '__main__':
    # Test data generation
    df = generate_sample_data(num_days=7)
    print("Sample Data Generated:")
    print(df.head(10))
    print(f"\nTotal samples: {len(df)}")
    
    # Test sequence creation
    values = df['vehicle_count'].values
    scaler = MinMaxScaler()
    scaled_values = scaler.fit_transform(values.reshape(-1, 1)).flatten()
    
    X, y = create_sequences(scaled_values, SEQUENCE_LENGTH)
    print(f"\nSequence shape: X={X.shape}, y={y.shape}")
    
    # Test time features
    features = extract_time_features(9, 1)  # 9 AM on Tuesday
    print(f"\nTime features for 9 AM Tuesday: {features}")
    
    # Test congestion level
    print(f"\nCongestion levels:")
    print(f"  0.2 -> {get_congestion_level(0.2)}")
    print(f"  0.5 -> {get_congestion_level(0.5)}")
    print(f"  0.8 -> {get_congestion_level(0.8)}")
