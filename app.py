import streamlit as st
import pandas as pd
import pickle

# -----------------------------
# LOAD MODEL
# -----------------------------
try:
    with open("smart_model.pkl", "rb") as f:
        model, scaler, le, columns = pickle.load(f)
except Exception as e:
    st.error(f"Model loading failed: {e}")
    st.stop()

# -----------------------------
# TITLE
# -----------------------------
st.title("🌾 Smart IoT Prediction App")

st.write("Enter input values to get prediction")

# -----------------------------
# USER INPUT
# -----------------------------
user_input = []

for col in columns:
    val = st.number_input(f"{col}", value=0.0)
    user_input.append(val)

# -----------------------------
# PREDICTION
# -----------------------------
if st.button("Predict"):
    try:
        input_df = pd.DataFrame([user_input], columns=columns)

        # scale
        input_scaled = scaler.transform(input_df)

        # predict
        pred = model.predict(input_scaled)

        # decode
        result = le.inverse_transform(pred)

        st.success(f"Prediction: {result[0]}")

    except Exception as e:
        st.error(f"Prediction error: {e}")
