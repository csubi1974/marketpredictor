import yfinance as yf
import json

ticker = "RIOT"
t = yf.Ticker(ticker)
info = t.info

stats = {
    "trailingPE": info.get("trailingPE"),
    "forwardPE": info.get("forwardPE"),
    "marketCap": info.get("marketCap"),
    "currentPrice": info.get("currentPrice"),
}

print(json.dumps(stats, indent=4))
