"""
model.py - LSTM Model Architecture for Traffic Congestion Prediction

This module contains:
- LSTM model definition
- Model training and evaluation
- Model saving and loading
- Prediction utilities
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

# Model configuration
SEQUENCE_LENGTH = 24
NUM_TIME_FEATURES = 4  # hour_sin, hour_cos, day_sin, day_cos
MODEL_PATH = 'traffic_model.h5'


def build_lstm_model(input_shape=(SEQUENCE_LENGTH + NUM_TIME_FEATURES, 1)):
    """
    Build LSTM model for time-series traffic prediction.
    
    Architecture:
    - Input layer accepts sequences with time features
    - LSTM layers learn temporal patterns
    - Dropout prevents overfitting
    - Dense layers for final prediction
    
    Args:
        input_shape: Shape of input data (sequence_length + features, 1)
    
    Returns:
        Compiled Keras model
    """
    model = Sequential([
        # Input layer
        Input(shape=input_shape),
        
        # First LSTM layer - return sequences for stacking
        LSTM(128, return_sequences=True),
        BatchNormalization(),
        Dropout(0.3),
        
        # Second LSTM layer
        LSTM(64, return_sequences=True),
        BatchNormalization(),
        Dropout(0.3),
        
        # Third LSTM layer
        LSTM(32, return_sequences=False),
        BatchNormalization(),
        Dropout(0.2),
        
        # Dense layers for processing
        Dense(32, activation='relu'),
        Dropout(0.2),
        
        # Output layer - single value prediction
        Dense(1, activation='sigmoid')
    ])
    
    # Compile with Adam optimizer and MSE loss
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )
    
    return model


def train_model(X_train, y_train, X_val=None, y_val=None, epochs=50, batch_size=32):
    """
    Train the LSTM model.
    
    Args:
        X_train: Training features
        y_train: Training targets
        X_val: Validation features (optional)
        y_val: Validation targets (optional)
        epochs: Maximum training epochs
        batch_size: Training batch size
    
    Returns:
        Trained model and training history
    """
    # Build fresh model
    model = build_lstm_model()
    
    # Define callbacks
    callbacks = [
        # Stop early if validation loss doesn't improve
        EarlyStopping(
            monitor='val_loss' if X_val is not None else 'loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        # Reduce learning rate if plateau
        ReduceLROnPlateau(
            monitor='val_loss' if X_val is not None else 'loss',
            factor=0.5,
            patience=5,
            min_lr=0.0001,
            verbose=1
        )
    ]
    
    # Prepare validation data if not provided
    validation_data = None
    if X_val is not None and y_val is not None:
        validation_data = (X_val, y_val)
    
    # Train model
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=validation_data,
        callbacks=callbacks,
        verbose=1
    )
    
    return model, history


def evaluate_model(model, X_test, y_test):
    """
    Evaluate model performance on test data.
    
    Args:
        model: Trained Keras model
        X_test: Test features
        y_test: Test targets
    
    Returns:
        Dictionary with evaluation metrics
    """
    loss, mae = model.evaluate(X_test, y_test, verbose=0)
    
    # Make predictions
    y_pred = model.predict(X_test, verbose=0).flatten()
    
    # Calculate additional metrics
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    
    metrics = {
        'loss': loss,
        'mae': mae,
        'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
        'predictions': y_pred,
        'actuals': y_test
    }
    
    return metrics


def predict(model, input_sequence):
    """
    Make prediction using the trained model.
    
    Args:
        model: Trained Keras model
        input_sequence: Prepared input data (normalized)
    
    Returns:
        Predicted normalized value
    """
    # Reshape for model input: (samples, timesteps, features)
    X = input_sequence.reshape(1, -1, 1)
    
    # Predict
    prediction = model.predict(X, verbose=0)
    
    return prediction.flatten()[0]


def save_model(model, path=MODEL_PATH):
    """
    Save trained model to disk.
    
    Args:
        model: Trained Keras model
        path: File path for saving
    """
    model.save(path)
    print(f"Model saved to {path}")


def load_trained_model(path=MODEL_PATH):
    """
    Load trained model from disk.
    
    Args:
        path: File path of saved model
    
    Returns:
        Loaded Keras model
    """
    model = load_model(path)
    print(f"Model loaded from {path}")
    return model


def get_model_summary():
    """
    Print model architecture summary.
    """
    model = build_lstm_model()
    model.summary()


# Standalone test
if __name__ == '__main__':
    print("LSTM Model Architecture Test")
    print("=" * 50)
    get_model_summary()
    
    # Test with dummy data
    print("\nTesting with dummy data...")
    dummy_input = np.random.rand(1, SEQUENCE_LENGTH + NUM_TIME_FEATURES, 1)
    model = build_lstm_model()
    prediction = model.predict(dummy_input, verbose=0)
    print(f"Dummy prediction shape: {prediction.shape}")
    print(f"Dummy prediction value: {prediction[0][0]:.4f}")
