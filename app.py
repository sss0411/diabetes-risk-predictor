import streamlit as st
import numpy as np
import joblib

st.set_page_config(page_title="Diabetes Risk Predictor")

# Load trained artifacts
model = joblib.load("diabetes_model.joblib")
scaler = joblib.load("scaler.joblib")

st.title("🩺 Diabetes Risk Predictor")
st.write("Predicting diabetes risk using machine learning")

# User inputs
age = st.number_input("Age", min_value=18, max_value=100, value=40)
bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0)
glucose = st.number_input("Fasting Glucose (mg/dL)", min_value=50, max_value=300, value=100)

if st.button("Predict Risk"):
    X = np.array([[age, bmi, glucose]])
    X_scaled = scaler.transform(X)
    risk = model.predict_proba(X_scaled)[0][1]
    st.success(f"Predicted diabetes risk: {risk:.2%}")

st.caption("For research and educational purposes only")
