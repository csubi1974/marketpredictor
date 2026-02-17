import yfinance as yf
from datetime import datetime
import pandas as pd

ticker = "^GSPC"
start = "2020-01-01"
end = "2024-01-01"

print(f"Testing yfinance {yf.__version__} for {ticker}...")
try:
    data = yf.download(ticker, start=start, end=end, progress=False)
    print("Success!")
    print(data.head())
except Exception as e:
    print(f"Failed: {e}")
