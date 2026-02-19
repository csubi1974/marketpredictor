import yfinance as yf
import pandas as pd

ticker = "RIOT"
t = yf.Ticker(ticker)

print("--- FINANCIALS RIOT ---")
print(t.financials.iloc[:, 0:2]) # Ultime 2 colonne

print("\n--- BALANCE SHEET RIOT ---")
print(t.balance_sheet.iloc[:, 0:1])

print("\n--- CASH FLOW RIOT ---")
print(t.cashflow.iloc[:, 0:1])
