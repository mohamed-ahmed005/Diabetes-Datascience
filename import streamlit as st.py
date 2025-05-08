import streamlit as st
import numpy as np
import joblib

# Load the trained model and scaler
model = joblib.load("random_forest_model.pkl")
scaler = joblib.load("scaler.pkl")

st.set_page_config(page_title="Diabetes Prediction", layout="centered")
st.title("🩺 Diabetes Prediction App")
st.markdown("Enter the following details to check if the patient is diabetic:")

# Input fields
pregnancies = st.number_input("Pregnancies", min_value=0, step=1)
glucose = st.number_input("Glucose", min_value=0)
blood_pressure = st.number_input("Blood Pressure", min_value=0)
skin_thickness = st.number_input("Skin Thickness", min_value=0)
bmi = st.number_input("BMI", min_value=0.0, format="%.1f")
dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, format="%.3f")
age = st.number_input("Age", min_value=0, step=1)

# Predict button
if st.button("Predict"):
    input_data = [pregnancies, glucose, blood_pressure, skin_thickness, bmi, dpf, age]
    input_array = np.array(input_data).reshape(1, -1)

    # Scale the input using the saved scaler
    input_scaled = scaler.transform(input_array)

    # Make prediction
    prediction = model.predict(input_scaled)

    # Show result
    if prediction[0] == 1:
        st.error("⚠️ The model predicts this person is **Diabetic**.")
    else:
        st.success("✅ The model predicts this person is **Non-Diabetic**.")
