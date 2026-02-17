import yfinance as yf
ticker = "^GSPC"
data = yf.download(ticker, start="2023-01-01", progress=False, multi_level_index=False)
print(data.columns)
print(data.head(1))
