import os
import requests
import numpy as np
from tensorflow.keras.models import load_model

# GitHub release base URL
GITHUB_RELEASE_BASE = (
    "https://github.com/varun3934/"
    "stock-price-prediction-lstm-prophet/"
    "releases/download/v1.0"
)

LSTM_DIR = "trained_models/lstm"


def download_file(url, dest_path):
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)

    response = requests.get(url, stream=True)
    response.raise_for_status()

    with open(dest_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)


def ensure_lstm_model(stock_name: str):
    model_filename = f"{stock_name}_model.h5"
    model_path = os.path.join(LSTM_DIR, model_filename)

    if not os.path.exists(model_path):
        download_url = f"{GITHUB_RELEASE_BASE}/{model_filename}"
        download_file(download_url, model_path)

    return model_path


def load_lstm_model(stock_name: str):
    model_path = ensure_lstm_model(stock_name)
    return load_model(model_path)


def predict_next_days(model, last_sequence, scaler, days: int):
    predictions = []
    current_seq = last_sequence.copy()

    for _ in range(days):
        pred = model.predict(
            current_seq.reshape(1, -1, 1),
            verbose=0
        )
        predictions.append(pred[0][0])
        current_seq = np.append(current_seq[1:], pred[0][0])

    predictions = scaler.inverse_transform(
        np.array(predictions).reshape(-1, 1)
    )

    return predictions.flatten().tolist()
