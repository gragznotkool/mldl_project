"""
app.py - Flask Web Application for Traffic Congestion Prediction

This Flask application provides:
- Web interface for traffic prediction
- REST API endpoint for predictions
- Visualization of model results
"""

import os
import numpy as np
from flask import Flask, render_template, request, jsonify
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import io
import base64
from datetime import datetime, timedelta

from model import load_trained_model, build_lstm_model
from utils import (
    extract_time_features,
    get_congestion_level,
    generate_sample_data,
    SEQUENCE_LENGTH,
    apply_heuristic_modifiers,
    calculate_metrics_from_prediction,
    generate_hourly_forecast,
    generate_weekly_pattern
)

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'traffic-congestion-secret-key'

# Global variables
model = None
scaler = None
last_predictions = []


def load_model_and_scaler():
    """
    Load trained model and scaler on application startup.
    If model doesn't exist, create and train a new one.
    """
    global model, scaler
    
    model_path = os.path.join(os.path.dirname(__file__), 'traffic_model.h5')
    scaler_path = os.path.join(os.path.dirname(__file__), 'scaler_params.npy')
    
    # Check if model exists
    if os.path.exists(model_path):
        try:
            model = load_trained_model(model_path)
            print("Loaded existing model from traffic_model.h5")
        except Exception as e:
            print(f"Error loading model: {e}")
            model = None
    
    if os.path.exists(scaler_path):
        try:
            scaler_params = np.load(scaler_path, allow_pickle=True).item()
            scaler = MinMaxScaler()
            scaler.min_, scaler.scale_ = scaler_params['min'], scaler_params['scale']
            scaler.data_min_ = np.array([scaler_params['min']])
            scaler.data_max_ = np.array([scaler_params['max']])
            print("Loaded scaler parameters")
        except Exception as e:
            print(f"Error loading scaler: {e}")
            scaler = None
    
    # If model still None, create a basic model for demo
    if model is None:
        print("Creating new model for demonstration...")
        model = build_lstm_model()
        scaler = MinMaxScaler(feature_range=(0, 1))
        
        # Generate and fit sample data
        df = generate_sample_data(num_days=30)
        data = df['vehicle_count'].values.reshape(-1, 1)
        scaler.fit(data)


def prepare_prediction_data(hour, day, past_values):
    """
    Prepare input data for model prediction.
    
    Args:
        hour: Hour of day (0-23)
        day: Day of week (0-6)
        past_values: List of past traffic values
    
    Returns:
        Prepared input array for model
    """
    # Extract time features
    time_features = extract_time_features(hour, day)
    
    # Ensure we have enough past values
    if len(past_values) < SEQUENCE_LENGTH:
        # Pad with average if not enough values
        past_values = past_values + [np.mean(past_values)] * (SEQUENCE_LENGTH - len(past_values))
    
    # Take last SEQUENCE_LENGTH values
    past_values = past_values[-SEQUENCE_LENGTH:]
    
    # Normalize past values
    normalized_past = []
    for val in past_values:
        normalized_val = (val - scaler.data_min_[0]) / (scaler.data_max_[0] - scaler.data_min_[0])
        normalized_val = max(0, min(1, normalized_val))
        normalized_past.append(normalized_val)
    
    # Combine past values with time features
    combined = np.array(normalized_past + time_features)
    
    return combined


def make_prediction(input_data):
    """
    Make prediction using the model.
    
    Args:
        input_data: Prepared input array
    
    Returns:
        Predicted normalized value
    """
    # Reshape for model: (samples, timesteps, features)
    X = input_data.reshape(1, -1, 1)
    
    # Predict
    prediction = model.predict(X, verbose=0)
    
    return prediction.flatten()[0]


def denormalize_prediction(normalized_value):
    """
    Convert normalized prediction to actual vehicle count.
    
    Args:
        normalized_value: Value in [0, 1] range
    
    Returns:
        Approximate vehicle count
    """
    min_val = scaler.data_min_[0]
    max_val = scaler.data_max_[0]
    
    actual_value = normalized_value * (max_val - min_val) + min_val
    return int(round(actual_value))


def generate_prediction_plot():
    """
    Generate visualization of recent predictions.
    
    Returns:
        Base64 encoded image
    """
    global last_predictions
    
    if len(last_predictions) < 2:
        return None
    
    fig, ax = plt.subplots(figsize=(10, 5))
    
    # Plot predictions
    x = range(len(last_predictions))
    ax.plot(x, last_predictions, 'b-o', label='Predicted Values', markersize=4)
    
    ax.set_xlabel('Prediction Index')
    ax.set_ylabel('Normalized Traffic Value')
    ax.set_title('Recent Traffic Predictions')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Save to bytes
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    plt.close()
    
    # Encode to base64
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()
    
    return img_base64


# Routes
@app.route('/')
def index():
    """Render main page."""
    # Generate sample data info
    df = generate_sample_data(num_days=1)
    sample_hour = 9
    sample_day = 1
    sample_values = df['vehicle_count'].values[-SEQUENCE_LENGTH:].tolist()
    
    return render_template(
        'index.html',
        sample_hour=sample_hour,
        sample_day=sample_day,
        sample_values=sample_values
    )


@app.route('/predict', methods=['POST'])
def predict():
    global last_predictions
    
    try:
        data = request.get_json()
        
        # Extract input
        hour = int(data.get('hour', 9))
        day = int(data.get('day', 1))
        past_values = data.get('past_values', [])
        
        # Extract new advanced parameters
        season = data.get('season', 'Spring')
        weather = data.get('weather', 'Clear')
        road_type = data.get('road_type', 'Highway')
        special_factors = data.get('special_factors', [])
        
        # Validate input
        if hour < 0 or hour > 23:
            return jsonify({'success': False, 'error': 'Hour must be 0-23'}), 400
        if day < 0 or day > 6:
            return jsonify({'success': False, 'error': 'Day must be 0-6'}), 400
        if len(past_values) < 1:
            return jsonify({'success': False, 'error': 'At least one past value required'}), 400
        
        # Prepare data
        input_data = prepare_prediction_data(hour, day, past_values)
        
        # Make base model prediction
        raw_pred = make_prediction(input_data)
        
        # Apply programmatic modifier layer
        adjusted_pred, factors_breakdown = apply_heuristic_modifiers(
            raw_pred, season, weather, road_type, special_factors
        )
        
        # Get classical output
        congestion = get_congestion_level(adjusted_pred)
        vehicle_count = denormalize_prediction(adjusted_pred)
        
        # Calculate robust metrics
        target_metrics = calculate_metrics_from_prediction(adjusted_pred)
        
        # Generate chart arrays for frontend
        hourly_forecast = generate_hourly_forecast(raw_pred, adjusted_pred)
        weekly_pattern = generate_weekly_pattern()
        
        # Store for visualization (legacy/fallback)
        last_predictions.append(adjusted_pred)
        if len(last_predictions) > 50:
            last_predictions = last_predictions[-50:]
        
        return jsonify({
            'success': True,
            'prediction': float(adjusted_pred),
            'congestion_level': congestion,
            'vehicle_count': vehicle_count,
            'metrics': target_metrics,
            'charts': {
                'hourly_forecast': hourly_forecast,
                'factor_contribution': factors_breakdown,
                'weekly_pattern': weekly_pattern
            },
            'message': f'Predicted {congestion.lower()} traffic with approximately {vehicle_count} vehicles'
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/visualize', methods=['GET'])
def visualize():
    """
    Generate and return prediction visualization.
    
    Returns:
        PNG image of prediction history
    """
    img_base64 = generate_prediction_plot()
    
    if img_base64:
        return jsonify({
            'success': True,
            'image': img_base64
        })
    else:
        return jsonify({
            'success': False,
            'error': 'Not enough predictions yet'
        })


@app.route('/sample-data', methods=['GET'])
def sample_data():
    """
    Generate sample data for testing.
    
    Returns:
    {
        "hour": 9,
        "day": 1,
        "values": [35, 40, ...]
    }
    """
    num_days = int(request.args.get('days', 1))
    df = generate_sample_data(num_days=num_days)
    
    # Get values for a specific hour pattern
    hour = 9
    day = int(request.args.get('day', 1))
    
    values = df['vehicle_count'].values[-SEQUENCE_LENGTH:].tolist()
    
    return jsonify({
        'hour': hour,
        'day': day,
        'values': values
    })


@app.route('/about')
def about():
    """Render about page with LSTM explanation."""
    return render_template('about.html')


# Initialize on startup
load_model_and_scaler()


if __name__ == '__main__':
    print("\n" + "=" * 50)
    print("Traffic Congestion Prediction System")
    print("=" * 50)
    print("\nStarting Flask server...")
    print("Access the application at: http://127.0.0.1:5000")
    print("\nPress Ctrl+C to stop the server\n")
    
    app.run(debug=True, host='127.0.0.1', port=5000)
