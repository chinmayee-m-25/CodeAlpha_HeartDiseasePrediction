
import streamlit as st
import numpy as np
import joblib

model  = joblib.load("best_model.pkl")
scaler = joblib.load("scaler.pkl")

st.set_page_config(page_title="Heart Disease Predictor", page_icon="🫀")
st.title("🫀 Heart Disease Prediction")
st.markdown("Fill in patient details and click Predict")
st.divider()

col1, col2, col3 = st.columns(3)

with col1:
    age      = st.number_input("Age",            min_value=1,   max_value=120, value=52)
    trestbps = st.number_input("Blood Pressure", min_value=80,  max_value=200, value=125)
    chol     = st.number_input("Cholesterol",    min_value=100, max_value=600, value=212)
    fbs      = st.selectbox("Fasting Blood Sugar > 120", [0, 1])
    restecg  = st.selectbox("Resting ECG", [0, 1, 2])

with col2:
    sex     = st.selectbox("Sex", ["Male", "Female"])
    cp      = st.selectbox("Chest Pain Type", [0, 1, 2, 3])
    thalach = st.number_input("Max Heart Rate", min_value=60, max_value=250, value=168)
    exang   = st.selectbox("Exercise Angina",  [0, 1])
    oldpeak = st.number_input("ST Depression", min_value=0.0, max_value=10.0, value=1.0)

with col3:
    slope = st.selectbox("Slope",          [0, 1, 2])
    ca    = st.selectbox("Major Vessels",  [0, 1, 2, 3, 4])
    thal  = st.selectbox("Thal",           [1, 2, 3])

st.divider()

if st.button("🔍 Predict", use_container_width=True):
    sex_val  = 1 if sex == "Male" else 0
    features = np.array([[age, sex_val, cp, trestbps, chol,
                          fbs, restecg, thalach, exang,
                          oldpeak, slope, ca, thal]])

    scaled      = scaler.transform(features)
    prediction  = model.predict(scaled)
    probability = model.predict_proba(scaled)[0]

    st.divider()
    if prediction[0] == 1:
        st.error("⚠️ Heart Disease DETECTED")
    else:
        st.success("✅ No Heart Disease Detected")

    c1, c2 = st.columns(2)
    c1.metric("🔴 Disease Probability", f"{probability[1]*100:.1f}%")
    c2.metric("🟢 Healthy Probability", f"{probability[0]*100:.1f}%")
    st.progress(float(probability[1]))
