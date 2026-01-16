#A clean function that fetches historical stock data for Indian stocks.
import yfinance as yf
from app.utils.constants import INDIAN_STOCKS

def fetch_stock_data(stock_name :str,period:str = "5y"):
    stock_name = stock_name.upper()

    if stock_name not in INDIAN_STOCKS:
        raise ValueError("Stock not supported")

    symbol = INDIAN_STOCKS[stock_name]

    data = yf.download(symbol, period=period)

    if data.empty:
        raise ValueError("No data found for the stock")

    return data
