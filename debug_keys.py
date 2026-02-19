import yfinance as yf

ticker = "RIOT"
t = yf.Ticker(ticker)

print("FINANCIALS_KEYS:")
for key in t.financials.index:
    print(f"- {key}")

print("\nBALANCE_SHEET_KEYS:")
for key in t.balance_sheet.index:
    print(f"- {key}")

print("\nCASHFLOW_KEYS:")
for key in t.cashflow.index:
    print(f"- {key}")
