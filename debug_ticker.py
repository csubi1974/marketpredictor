import yfinance as yf
import json

def test_ticker(symbol):
    print(f"Testing {symbol}...")
    try:
        t = yf.Ticker(symbol)
        info = t.info
        if not info:
            print(f"Error: No info found for {symbol}")
            return
        
        print(f"Successfully fetched info for {symbol}")
        print(f"Price: {info.get('currentPrice')}")
        print(f"Sector: {info.get('sector')}")
    except Exception as e:
        print(f"Exception for {symbol}: {str(e)}")

if __name__ == "__main__":
    test_ticker("INTC")
    test_ticker("AAPL")
