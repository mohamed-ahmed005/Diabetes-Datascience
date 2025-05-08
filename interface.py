import streamlit as st
import numpy as np
import joblib
from transformers import pipeline
import pandas as pd

# Load the trained model and scaler
model = joblib.load("random_forest_model.pkl")
scaler = joblib.load("scaler.pkl")

# Initialize GPT-2 pipeline (only once)
generator = pipeline("text-generation", model="gpt2")

# Streamlit UI
st.set_page_config(page_title="Diabetes Predictor + Lifestyle Plan", layout="centered")
st.title("🩺 Diabetes Prediction with Personalized Suggestions")
st.markdown("Enter the following details to check diabetes risk and receive personalized advice.")

# Input fields
pregnancies = st.number_input("Pregnancies", min_value=0, step=1)
glucose = st.number_input("Glucose", min_value=0)
blood_pressure = st.number_input("Blood Pressure", min_value=0)
skin_thickness = st.number_input("Skin Thickness", min_value=0)
bmi = st.number_input("BMI", min_value=0.0, format="%.1f")
dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, format="%.3f")
age = st.number_input("Age", min_value=0, step=1)

# Predict button
if st.button("Predict and Suggest"):
    # Prepare input
    input_data = [pregnancies, glucose, blood_pressure, skin_thickness, bmi, dpf, age]
    input_array = np.array(input_data).reshape(1, -1)
    input_scaled = scaler.transform(input_array)

    # Prediction
    prediction = model.predict(input_scaled)

    if prediction[0] == 1:
        st.error("⚠️ The model predicts this person is **Diabetic**.")

        # Generate GPT-2 suggestion
        prompt = (
            f"A diabetic patient with the following features:\n"
            f"Age: {age}, Glucose: {glucose}, BMI: {bmi}, BloodPressure: {blood_pressure}.\n"
            f"Suggest a personalized diet, exercise, and lifestyle plan:"
        )
        result = generator(prompt, max_length=120, num_return_sequences=1)[0]['generated_text']
        suggestion = result.replace(prompt, "").strip()

        st.markdown("### 📝 Personalized Lifestyle Suggestion:")
        st.info(suggestion)

    else:
        st.success("✅ The model predicts this person is **Non-Diabetic**.")
#streamlit run f:/visualcode/Diabetes-Datascience/interface.py