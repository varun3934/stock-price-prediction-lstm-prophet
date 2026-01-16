#

import numpy as np
from tensorflow.keras.models import load_model


def load_lstm_model(stock_name: str):
    model_path = f"trained_models/lstm/{stock_name}_model.h5"
    return load_model(model_path)


def predict_next_days(model, last_sequence, scaler, days: int):
    predictions = []
    current_seq = last_sequence.copy()

    for _ in range(days):
        pred = model.predict(current_seq.reshape(1, -1, 1), verbose=0)
        predictions.append(pred[0][0])
        current_seq = np.append(current_seq[1:], pred[0][0])

    predictions = scaler.inverse_transform(
        np.array(predictions).reshape(-1, 1)
    )

    return predictions.flatten().tolist()
