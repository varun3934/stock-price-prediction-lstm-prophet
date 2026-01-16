import streamlit as st
import requests
import matplotlib.pyplot as plt

API_URL = "https://stock-predictor-api-9flr.onrender.com/predict"


st.title("📈 Stock Price Predictor (LSTM vs Prophet)")

# --- User Inputs ---
stock = st.selectbox("Select Stock", ["TCS", "ADANI"])
days = st.slider("Prediction Days", 1, 30, 5)
model_choice = st.selectbox(
    "Select Model",
    ["lstm", "prophet", "compare"]
)

# --- Prediction Function ---
def get_prediction(stock, days, model):
    payload = {
        "stock_name": stock,
        "days": days,
        "model": model
    }
    response = requests.post(API_URL, json=payload)
    return response.json()

# --- Button ---
if st.button("Predict"):
    if model_choice != "compare":
        result = get_prediction(stock, days, model_choice)

        st.subheader(f"{model_choice.upper()} Predictions")
        st.write(result["predictions"])

        # Plot
        plt.figure()
        plt.plot(result["predictions"], marker="o")
        plt.title(f"{stock} - {model_choice.upper()} Forecast")
        plt.xlabel("Days Ahead")
        plt.ylabel("Price")
        st.pyplot(plt)

    else:
        # Compare LSTM vs Prophet
        lstm_result = get_prediction(stock, days, "lstm")
        prophet_result = get_prediction(stock, days, "prophet")

        st.subheader("LSTM vs Prophet Comparison")

        plt.figure()
        plt.plot(lstm_result["predictions"], label="LSTM", marker="o")
        plt.plot(prophet_result["predictions"], label="Prophet", marker="x")
        plt.legend()
        plt.xlabel("Days Ahead")
        plt.ylabel("Price")
        plt.title(f"{stock} Forecast Comparison")
        st.pyplot(plt)
