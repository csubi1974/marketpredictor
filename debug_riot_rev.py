import yfinance as yf
ticker = "RIOT"
t = yf.Ticker(ticker)
print(t.quarterly_financials.loc['Total Revenue'])
