import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load the trained model
model = joblib.load('internet_model.pkl')

# App title
st.title("📶 Internet Usage Prediction App")
st.write("Predict a user's **monthly data usage (GB)** based on how many hours they spend online per day.")

# Sidebar for user input
st.sidebar.header("Enter your data:")
hours_per_day = st.sidebar.number_input(
    "Average Hours Spent Online per Day:",
    min_value=0.0, max_value=24.0, step=0.5
)

# Prediction button
if st.sidebar.button("Predict"):
    # Prepare input data for prediction
    input_data = np.array([[hours_per_day]])

    # Make prediction
    prediction = model.predict(input_data)[0][0]  # since output is 2D

    # Show result
    st.success(f"📊 Estimated Monthly Data Usage: **{prediction:.2f} GB**")

# Optional: show model info
st.write("---")
st.write("🧠 *This model was trained using Linear Regression on user internet usage data.*")
