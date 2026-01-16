#preprocessing
import numpy as np
from sklearn.preprocessing import MinMaxScaler


def preprocess_for_lstm(data, lookback: int = 60):
    close_prices = data[['Close']].values

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(close_prices)

    X = []
    y = []

    for i in range(lookback, len(scaled_data)):
        X.append(scaled_data[i - lookback:i, 0])
        y.append(scaled_data[i, 0])

    X = np.array(X)
    y = np.array(y)

    X = X.reshape((X.shape[0], X.shape[1], 1))

    return X, y, scaler
