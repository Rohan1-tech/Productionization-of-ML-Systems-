# ========================================
# 🚀 Travel Cost Prediction App (SVR Model)
# ========================================

import streamlit as st
import numpy as np
import joblib

# Load trained model and scalers
model = joblib.load("best_svr_model.joblib")
x_scaler = joblib.load("x_scaler.joblib")
y_scaler = joblib.load("y_scaler.joblib")

# Page setup
st.set_page_config(page_title="Travel Cost Predictor", page_icon="✈️", layout="centered")

st.title("✈️ Travel Cost Prediction App")
st.markdown("Use this app to predict **total travel cost** based on trip details. 💰")

# Sidebar info
st.sidebar.header("About App")
st.sidebar.info("""
This app uses a **Support Vector Regression (SVR)** model 
trained on flight, hotel, and travel data.  
Adjust the sliders below to get a predicted travel cost in ₹.
""")

# Collect inputs
st.subheader("Enter Trip Details:")

flight_price = st.number_input("✈️ Flight Price (₹)", min_value=300.0, max_value=5000.0, value=950.0)
hotel_price = st.number_input("🏨 Hotel Price (₹)", min_value=60.0, max_value=3000.0, value=220.0)
days = st.number_input("📅 Number of Days", min_value=1, max_value=30, value=4)
distance = st.number_input("🛣️ Distance (km)", min_value=100.0, max_value=2000.0, value=550.0)
time = st.number_input("⏱️ Flight Time (hours)", min_value=0.4, max_value=10.0, value=1.5)

# Derived features (match model training)
hotel_price_per_day = hotel_price / days
distance_per_day = distance / days
travel_intensity = distance * time

# Prepare data for prediction
input_data = np.array([[flight_price, hotel_price, days, distance,
                        hotel_price_per_day, distance_per_day, travel_intensity]])
scaled_input = x_scaler.transform(input_data)

# Predict button
if st.button("💡 Predict Travel Cost"):
    pred_scaled = model.predict(scaled_input)
    pred_real = y_scaler.inverse_transform(pred_scaled.reshape(-1, 1))
    predicted_cost = pred_real[0][0]
    
    st.success(f"💰 Predicted Total Travel Cost: ₹{predicted_cost:,.2f}")
    st.balloons()

# Footer
st.markdown("---")
st.caption("Developed by a Data Science Enthusiast using Streamlit + SVR 🧠")
