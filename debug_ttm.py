import yfinance as yf
import pandas as pd

ticker = "RIOT"
t = yf.Ticker(ticker)

# Get current price
price = t.info.get('currentPrice')
market_cap = t.info.get('marketCap')

# Get TTM Revenue from quarterly financials
q_fin = t.quarterly_financials
if not q_fin.empty:
    ttm_revenue = q_fin.loc['Total Revenue'].iloc[:4].sum()
    manual_ps = market_cap / ttm_revenue if ttm_revenue > 0 else None
else:
    ttm_revenue = None
    manual_ps = None

# Get TTM EBITDA
if not q_fin.empty and 'EBITDA' in q_fin.index:
    ttm_ebitda = q_fin.loc['EBITDA'].iloc[:4].sum()
else:
    ttm_ebitda = None

ev = t.info.get('enterpriseValue')
manual_ev_ebitda = ev / ttm_ebitda if ttm_ebitda and ttm_ebitda > 0 else None

print(f"Current Price: {price}")
print(f"Market Cap: {market_cap}")
print(f"Enterprise Value: {ev}")
print(f"TTM Revenue (sum of 4 quarters): {ttm_revenue}")
print(f"Manual P/S: {manual_ps}")
print(f"yfinance info priceToSalesTrailing12Months: {t.info.get('priceToSalesTrailing12Months')}")
print(f"TTM EBITDA (sum of 4 quarters): {ttm_ebitda}")
print(f"Manual EV/EBITDA: {manual_ev_ebitda}")
print(f"yfinance info enterpriseToEbitda: {t.info.get('enterpriseToEbitda')}")
