import tensorflow as tf
import joblib
import numpy as np
import pandas as pd
import json

def load_assets():
    """Load model, scaler, and configuration."""
    # Load the LSTM Autoencoder model
    model = tf.keras.models.load_model('assets/autoencoder_model.h5')
    
    # Load the StandardScaler
    scaler = joblib.load('assets/scaler.pkl')
    
    # Load configuration
    with open('config.json', 'r') as f:
        config = json.load(f)
        
    return model, scaler, config

def prepare_live_data(df, scaler, window_size=20):
    """
    Transforms raw MAV data into the 3D shape (1, 20, 4) 
    required by your LSTM model.
    """
    # 1. Scale the data using the NASA-trained scaler
    scaled_data = scaler.transform(df)
    
    # 2. Ensure we have enough data for a full window
    if len(scaled_data) < window_size:
        return None
        
    # 3. Take the latest window and reshape for the model (batch, steps, features)
    latest_window = scaled_data[-window_size:]
    return latest_window.reshape(1, window_size, 4)

def calculate_anomaly_score(model, processed_window):
    """Calculates the reconstruction error (MAE)."""
    reconstruction = model.predict(processed_window)
    # Mean Absolute Error between input and reconstruction
    mae_score = np.mean(np.abs(processed_window - reconstruction))
    return mae_score