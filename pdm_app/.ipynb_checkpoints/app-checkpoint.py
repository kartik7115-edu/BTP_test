import streamlit as st
import pandas as pd
import numpy as np
import time
from model_loader import load_assets, prepare_live_data, calculate_anomaly_score

# Page Configuration
st.set_page_config(page_title="PdM Dashboard", layout="wide")
st.title("⚙️ Predictive Maintenance Dashboard")

# 1. Load Assets (Model, Scaler, Config)
model, scaler, config = load_assets()

# 2. Sidebar for Controls
st.sidebar.header("System Settings")
st.sidebar.write(f"Threshold: {config['anomaly_threshold']}")

# 3. Simulate Live Data Stream 
# In a real scenario, replace this with a call to your sensor (e.g., MQTT or API)
def get_live_sensor_data():
    # Creating dummy data with 4 features to match your NASA model input
    return pd.DataFrame(np.random.rand(20, 4), columns=['B1', 'B2', 'B3', 'B4'])

# 4. Dashboard Layout
col1, col2 = st.columns(2)

placeholder = st.empty()

while True:
    with placeholder.container():
        # Get data
        raw_data = get_live_sensor_data()
        
        # Process and Predict
        processed = prepare_live_data(raw_data, scaler, config['window_size'])
        if processed is not None:
            score = calculate_anomaly_score(model, processed)
            
            # Display Metrics
            col1.metric("Current MAE Score", f"{score:.5f}")
            
            if score > config['anomaly_threshold']:
                st.error("⚠️ ANOMALY DETECTED: Maintenance Required!")
                col2.metric("Status", "CRITICAL")
            else:
                st.success("✅ System Status: Healthy")
                col2.metric("Status", "HEALTHY")
                
            # Visualization
            st.line_chart(raw_data)
            
    time.sleep(2) # Refresh every 2 seconds