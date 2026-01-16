import os
import requests
import joblib

# GitHub release base URL
GITHUB_RELEASE_BASE = (
    "https://github.com/varun3934/"
    "stock-price-prediction-lstm-prophet/"
    "releases/download/v1.0"
)

PROPHET_DIR = "trained_models/prophet"


def download_file(url, dest_path):
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)

    response = requests.get(url, stream=True)
    response.raise_for_status()

    with open(dest_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)


def ensure_prophet_model(stock_name: str):
    stock_name = stock_name.upper()
    model_filename = f"{stock_name}_prophet.pkl"
    model_path = os.path.join(PROPHET_DIR, model_filename)

    if not os.path.exists(model_path):
        download_url = f"{GITHUB_RELEASE_BASE}/{model_filename}"
        download_file(download_url, model_path)

    return model_path


def load_prophet_model(stock_name: str):
    model_path = ensure_prophet_model(stock_name)
    return joblib.load(model_path)


def predict_prophet(model, days: int):
    future = model.make_future_dataframe(periods=days)
    forecast = model.predict(future)
    return forecast.tail(days)["yhat"].tolist()
