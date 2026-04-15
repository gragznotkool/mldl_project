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
