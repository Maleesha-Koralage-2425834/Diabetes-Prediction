import streamlit as st
import pandas as pd
import joblib

model = joblib.load("training_model.pkl")

st.set_page_config(page_title="Diabetes Predictor")

st.title("🩺 Diabetes Prediction")

age = st.number_input("Age", 1, 120, 30)
bmi = st.number_input("BMI", 10.0, 60.0, 25.0)
hba1c = st.number_input("HbA1c", 3.0, 15.0, 5.5)

if st.button("Predict"):
    X = pd.DataFrame([[age, bmi, hba1c]], columns=["age","bmi","hba1c"])
    pred = model.predict(X)[0]
    prob = model.predict_proba(X)[0][1]

    st.success(f"Prediction: {pred}, Probability: {prob:.2f}")
