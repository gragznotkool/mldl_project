# Time-Series Traffic Congestion Prediction using LSTM

A complete deep learning project that predicts traffic congestion levels using Long Short-Term Memory (LSTM) neural networks with a Flask web interface.

## Project Overview

This project demonstrates time-series prediction for traffic data, predicting congestion levels (Low/Medium/High) based on historical traffic patterns and temporal features.

## Features

- **LSTM Model**: Deep learning model for time-series prediction
- **Web Interface**: Clean, responsive UI for traffic prediction
- **Real-time Prediction**: Instant predictions via REST API
- **Visualization**: Charts showing prediction history
- **Sample Data Generation**: Built-in realistic traffic data generation

## Project Structure

```
traffic_congestion_project/
├── app.py              # Flask web application (MAIN FILE)
├── model.py            # LSTM model architecture
├── utils.py            # Data preprocessing utilities
├── train.py            # Model training script
├── requirements.txt    # Python dependencies
├── traffic_model.h5    # Trained model (after training)
├── scaler_params.npy  # Scaler parameters (after training)
├── traffic_data.csv    # Sample traffic data (after training)
├── templates/
│   ├── index.html      # Main prediction page
│   └── about.html      # LSTM explanation page
└── static/
    ├── style.css       # Styling
    ├── training_history.png
    └── predictions_plot.png
```

## Installation

1. **Create virtual environment** (recommended):
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

## Usage

### Step 1: Train the Model (First Time)

```bash
cd traffic_congestion_project
python train.py
```

This will:
- Generate 90 days of synthetic traffic data
- Train the LSTM model
- Save the model as `traffic_model.h5`
- Generate visualization plots
- Display training metrics

### Step 2: Run the Web Application

```bash
python app.py
```

Access the application at: **http://127.0.0.1:5000**

### Step 3: Make Predictions

1. Enter the hour of day (0-23)
2. Select the day of week (0-6)
3. Enter past 24 hours of traffic values (or use Auto-Fill)
4. Click "Predict Traffic"

## API Endpoints

### POST /predict

Predict traffic congestion for given input.

**Request:**
```json
{
    "hour": 9,
    "day": 1,
    "past_values": [35, 40, 45, 55, 60, ...]  // 24 values
}
```

**Response:**
```json
{
    "success": true,
    "prediction": 0.65,
    "congestion_level": "Medium",
    "vehicle_count": 58,
    "message": "Predicted medium traffic with approximately 58 vehicles"
}
```

### GET /sample-data

Get sample traffic data for testing.

**Response:**
```json
{
    "hour": 9,
    "day": 1,
    "values": [35, 40, 45, ...]
}
```

### GET /visualize

Get prediction history visualization.

## Understanding LSTM

### What is LSTM?

Long Short-Term Memory (LSTM) is a type of recurrent neural network (RNN) that can learn long-term dependencies in sequential data. Unlike traditional neural networks, LSTMs have "memory" that allows them to remember information from earlier in the sequence.

### Why LSTM for Traffic?

Traffic patterns are inherently temporal:
- Rush hours (7-9 AM, 5-7 PM) have higher traffic
- Weekends have different patterns than weekdays
- Previous hours' traffic affects current traffic

LSTMs excel at capturing these temporal dependencies.

### Sequence Windowing

LSTMs require sequences of data to make predictions. Sequence windowing creates overlapping windows of past values:

```
Data: [20, 35, 45, 60, 55, 70, 65, 80, ...]

Window 1: [20, 35, 45, 60] → Predict next: 55
Window 2: [35, 45, 60, 55] → Predict next: 70
Window 3: [45, 60, 55, 70] → Predict next: 65
```

This approach:
- Creates training examples from sequential data
- Captures local temporal patterns
- Increases effective training data size

## Model Architecture

```
Input Layer: 28 features (24 past values + 4 time features)
    ↓
LSTM Layer 1: 128 units, return_sequences=True
    ↓
Batch Normalization + Dropout (0.3)
    ↓
LSTM Layer 2: 64 units, return_sequences=True
    ↓
Batch Normalization + Dropout (0.3)
    ↓
LSTM Layer 3: 32 units
    ↓
Batch Normalization + Dropout (0.2)
    ↓
Dense Layer: 32 units, ReLU activation
    ↓
Dropout (0.2)
    ↓
Output Layer: 1 unit, Sigmoid activation
```

## Data Preprocessing

1. **Normalization**: Scale values to [0, 1] using MinMaxScaler
2. **Time Features**: Extract cyclical features using sin/cos encoding
   - hour_sin, hour_cos: Encode hour cyclically
   - day_sin, day_cos: Encode day of week cyclically
3. **Sequence Creation**: Create sliding windows of 24 values

## Congestion Classification

| Normalized Value | Congestion Level |
|-----------------|------------------|
| < 0.33          | Low              |
| 0.33 - 0.66     | Medium           |
| > 0.66          | High             |

## Requirements

- Python 3.8+
- TensorFlow 2.10+
- Flask 2.3+
- NumPy
- Pandas
- Scikit-learn
- Matplotlib

## Example Output

```
Training Progress:
Epoch 1/50 - loss: 0.0452 - val_loss: 0.0389
Epoch 2/50 - loss: 0.0321 - val_loss: 0.0298
...
Epoch 25/50 - loss: 0.0089 - val_loss: 0.0092

Model saved to traffic_model.h5
Test Performance: MAE = 0.0456

Flask server running at http://127.0.0.1:5000
```

## Tips for College Project

1. **Explain the Problem**: Start with why traffic prediction matters
2. **Show the Data**: Display sample traffic data patterns
3. **Visualize Training**: Show loss curves and prediction accuracy
4. **Demonstrate Real-time**: Use the web interface live
5. **Discuss Limitations**: Mention data quality, model accuracy, etc.

## Troubleshooting

**Model not found error:**
```bash
python train.py  # Train the model first
```

**Port already in use:**
```python
# In app.py, change port:
app.run(debug=True, host='127.0.0.1', port=5001)
```

**TensorFlow warnings:**
These are normal and can be ignored for the project demo.

## Future Improvements

- [ ] Add real-time traffic data integration
- [ ] Implement ensemble methods
- [ ] Add weather feature integration
- [ ] Deploy to cloud platform
- [ ] Add user authentication

## License

This project is for educational purposes (College Mini Project).

---

**Built with TensorFlow/Keras | Flask | Pure JavaScript**
