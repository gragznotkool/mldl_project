"""
train.py - Training Script for Traffic Congestion Prediction Model

This script:
- Generates or loads traffic data
- Preprocesses data for LSTM
- Trains the model
- Evaluates and saves the model
- Generates visualization plots
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

from utils import (
    generate_sample_data,
    create_sequences,
    normalize_data,
    SEQUENCE_LENGTH
)
from model import train_model, evaluate_model, save_model, build_lstm_model


def prepare_training_data(df, test_size=0.2):
    """
    Prepare data for LSTM training.
    
    Steps:
    1. Extract vehicle counts
    2. Normalize the data
    3. Create sequences
    4. Split into train/test sets
    
    Args:
        df: DataFrame with 'vehicle_count' column
        test_size: Proportion of data for testing
    
    Returns:
        Tuple of (X_train, X_test, y_train, y_test, scaler)
    """
    # Get vehicle counts
    vehicle_counts = df['vehicle_count'].values
    
    # Normalize data to [0, 1]
    normalized_data, scaler = normalize_data(vehicle_counts)
    
    # Create sequences for LSTM
    X, y = create_sequences(normalized_data, SEQUENCE_LENGTH)
    
    # Reshape X for LSTM: (samples, timesteps, features)
    # Here features = 1 (just the vehicle count)
    X = X.reshape((X.shape[0], X.shape[1], 1))
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, shuffle=True, random_state=42
    )
    
    print(f"Training data: X_train={X_train.shape}, y_train={y_train.shape}")
    print(f"Testing data: X_test={X_test.shape}, y_test={y_test.shape}")
    
    return X_train, X_test, y_train, y_test, scaler


def plot_training_history(history):
    """
    Plot training history (loss and MAE over epochs).
    
    Args:
        history: Keras training history object
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot loss
    axes[0].plot(history.history['loss'], label='Training Loss', color='blue')
    if 'val_loss' in history.history:
        axes[0].plot(history.history['val_loss'], label='Validation Loss', color='red')
    axes[0].set_title('Model Loss', fontsize=14)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss (MSE)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot MAE
    axes[1].plot(history.history['mae'], label='Training MAE', color='blue')
    if 'val_mae' in history.history:
        axes[1].plot(history.history['val_mae'], label='Validation MAE', color='red')
    axes[1].set_title('Model MAE', fontsize=14)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Mean Absolute Error')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('static/training_history.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Training history plot saved to static/training_history.png")


def plot_predictions(y_actual, y_predicted, title='Actual vs Predicted Traffic'):
    """
    Plot actual vs predicted traffic values.
    
    Args:
        y_actual: Actual traffic values
        y_predicted: Model predictions
        title: Plot title
    """
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # Time series comparison
    axes[0].plot(y_actual, label='Actual', color='blue', alpha=0.7)
    axes[0].plot(y_predicted, label='Predicted', color='red', alpha=0.7)
    axes[0].set_title(title, fontsize=14)
    axes[0].set_xlabel('Time Step')
    axes[0].set_ylabel('Normalized Traffic')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Scatter plot
    axes[1].scatter(y_actual, y_predicted, alpha=0.5, color='purple')
    axes[1].plot([0, 1], [0, 1], 'r--', label='Perfect Prediction')
    axes[1].set_title('Prediction Accuracy', fontsize=14)
    axes[1].set_xlabel('Actual Values')
    axes[1].set_ylabel('Predicted Values')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('static/predictions_plot.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Predictions plot saved to static/predictions_plot.png")


def main():
    """
    Main training pipeline.
    """
    print("=" * 60)
    print("Traffic Congestion Prediction - LSTM Model Training")
    print("=" * 60)
    
    # Generate sample data
    print("\n1. Generating sample traffic data...")
    df = generate_sample_data(num_days=90)  # 90 days of hourly data
    print(f"   Generated {len(df)} data points")
    print(f"   Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    
    # Save sample data
    df.to_csv('traffic_data.csv', index=False)
    print("   Data saved to traffic_data.csv")
    
    # Prepare data
    print("\n2. Preparing data for LSTM training...")
    X_train, X_test, y_train, y_test, scaler = prepare_training_data(df)
    
    # Train model
    print("\n3. Training LSTM model...")
    model, history = train_model(
        X_train, y_train,
        X_test, y_test,
        epochs=50,
        batch_size=32
    )
    
    # Plot training history
    print("\n4. Generating training plots...")
    plot_training_history(history)
    
    # Evaluate model
    print("\n5. Evaluating model...")
    metrics = evaluate_model(model, X_test, y_test)
    print(f"   Test Loss (MSE): {metrics['loss']:.4f}")
    print(f"   Test MAE: {metrics['mae']:.4f}")
    print(f"   Test RMSE: {metrics['rmse']:.4f}")
    
    # Plot predictions
    plot_predictions(metrics['actuals'], metrics['predictions'])
    
    # Save model and scaler
    print("\n6. Saving model...")
    save_model(model, 'traffic_model.h5')
    
    # Save scaler parameters for later use
    scaler_params = {
        'min': scaler.data_min_[0],
        'max': scaler.data_max_[0],
        'scale': scaler.scale_[0]
    }
    np.save('scaler_params.npy', scaler_params)
    print("   Scaler parameters saved to scaler_params.npy")
    
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"\nModel saved as: traffic_model.h5")
    print(f"Test Performance: MAE = {metrics['mae']:.4f}")
    print("\nTo run the web interface, execute: python app.py")


if __name__ == '__main__':
    main()
