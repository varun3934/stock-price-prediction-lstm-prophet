#prophet models
import joblib

def load_prophet_model(stock_name: str):
    path = f"trained_models/prophet/{stock_name}_prophet.pkl"
    return joblib.load(path)

def predict_prophet(model, days: int):
    future = model.make_future_dataframe(periods=days)
    forecast = model.predict(future)
    return forecast.tail(days)["yhat"].tolist()
