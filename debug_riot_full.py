import yfinance as yf
import json

ticker = "RIOT"
t = yf.Ticker(ticker)
info = t.info

keys_to_check = [
    "trailingPE", "forwardPE", "priceToSalesTrailing12Months", "priceToSales", 
    "priceToBook", "enterpriseToRevenue", "enterpriseToEbitda", "pegRatio"
]

stats = {k: info.get(k) for k in keys_to_check}
print(json.dumps(stats, indent=4))
