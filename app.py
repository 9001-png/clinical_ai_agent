import streamlit as st
import pickle
import numpy as np

model = pickle.load(open("disease_model.pkl", "rb"))

st.title("🩺 Clinical Care & Diagnostics AI Agent")

st.write("Enter patient symptoms:")

fever = st.checkbox("Fever")
cough = st.checkbox("Cough")
fatigue = st.checkbox("Fatigue")
headache = st.checkbox("Headache")

if st.button("Predict Disease"):
    input_data = np.array([[fever, cough, fatigue, headache]])
    prediction = model.predict(input_data)

    st.success(f"Predicted Disease: {prediction[0]}")
    st.info("Please consult a doctor for confirmation.")