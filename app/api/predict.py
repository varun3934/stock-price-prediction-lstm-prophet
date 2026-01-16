from fastapi import APIRouter, HTTPException
from app.schemas.stock_schema import StockRequest
from app.services.data_service import fetch_stock_data
from app.utils.preprocessing import preprocess_for_lstm
from app.models.lstm_model import load_lstm_model, predict_next_days
from app.models.prophet_model import load_prophet_model, predict_prophet


router = APIRouter()

@router.post("/predict")
def predict_stock(request: StockRequest):
    try:
        if request.model.lower() == "lstm":
        # 1. Fetch stock data
            data = fetch_stock_data(request.stock_name)

            # 2. Preprocess data
            X, y, scaler = preprocess_for_lstm(data)

            # 3. Take last sequence
            last_sequence = X[-1].reshape(-1)

            # 4. Load trained LSTM model
            model = load_lstm_model(request.stock_name)

            # 5. Predict future prices
            predictions = predict_next_days(
                model=model,
                last_sequence=last_sequence,
                scaler=scaler,
                days=request.days
            )

            return {
                "stock": request.stock_name,
                "model": "LSTM",
                "days": request.days,
                "predictions": predictions
            }
        elif request.model.lower() == "prophet":

            model = load_prophet_model(request.stock_name)
            predictions = predict_prophet(model, request.days)
            

            return {
                "stock": request.stock_name,
                "model": "PROPHET",
                "days": request.days,
                "predictions": predictions
            }

    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Model file not found")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
