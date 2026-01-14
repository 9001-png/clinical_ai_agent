import streamlit as st
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier

st.title("🩺 Clinical Care & Diagnostics AI Agent")

# Load dataset
data = pd.read_csv("symptoms_dataset.csv")

X = data.drop("disease", axis=1)
y = data["disease"]

# Train model
model = DecisionTreeClassifier()
model.fit(X, y)

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
