📈 Stock Price Prediction System (LSTM vs Prophet)

An end-to-end stock price forecasting application built using FastAPI, LSTM, Prophet, and Streamlit, focused on Indian NSE stocks such as TCS and Adani.

This project allows users to select a stock and model dynamically and visualize future price predictions along with LSTM vs Prophet comparisons.

🚀 Features

📊 Multi-stock support (TCS, ADANI)

🧠 Multiple models

LSTM (Deep Learning)

Prophet (Statistical Time Series)

🔄 Dynamic model selection

🖥️ FastAPI backend for inference

🎨 Streamlit frontend with charts

📈 LSTM vs Prophet comparison plot

🧩 Clean, modular, industry-standard architecture

🏗️ Architecture Overview
User
 ↓
Streamlit UI
 ↓
FastAPI Backend
 ↓
Data Fetching (yfinance)
 ↓
LSTM / Prophet Models
 ↓
Predictions + Visualization

🧠 Models Used
🔹 LSTM

Trained offline using 5 years of NSE data

Sliding window (60 days)

MinMax scaling

Best for short-term trend learning

🔹 Prophet

Trend + seasonality based forecasting

Interpretable and fast

Useful baseline for comparison

📁 Project Structure
stockproject/
│
├── app/                    # FastAPI backend
│   ├── api/                # API routes
│   ├── models/             # LSTM & Prophet loaders
│   ├── services/           # Data fetching logic
│   ├── schemas/            # Request validation
│   └── utils/              # Preprocessing & constants
│
├── trained_models/
│   ├── lstm/               # Saved LSTM models
│   └── prophet/            # Saved Prophet models
│
├── streamlit/              # Streamlit frontend
│   └── streamlit.py
│
├── notebooks/              # Training notebooks
├── requirements.txt
└── README.md

▶️ How to Run Locally
1️⃣ Start FastAPI backend
uvicorn app.main:app --reload


API Docs:

http://127.0.0.1:8000/docs

2️⃣ Start Streamlit frontend
streamlit run streamlit/streamlit.py

📊 Sample Output

LSTM predictions (smooth trend)

Prophet predictions (trend + seasonality)

Comparison chart for easy interpretation

⚠️ Disclaimer

This project is for educational purposes only.
Predictions should not be used for financial or trading decisions.

👨‍💻 Author

Varun Kumar M
Aspiring AI / ML Engineer

🧠 Skills Demonstrated

Machine Learning

Deep Learning (LSTM)

Time Series Forecasting

FastAPI

Streamlit

Model Deployment Readiness

Clean Software Architecture