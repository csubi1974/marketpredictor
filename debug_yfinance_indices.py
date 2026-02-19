import yfinance as yf
import pandas as pd

ticker = "RIOT"
t = yf.Ticker(ticker)

print("--- FINANCIALS INDEX ---")
print(t.financials.index.tolist())
print("\n--- FINANCIALS REVENUE (LATEST) ---")
print(t.financials.iloc[:, 0])

print("\n--- BALANCE SHEET INDEX ---")
print(t.balance_sheet.index.tolist())
print("\n--- BALANCE SHEET (LATEST) ---")
print(t.balance_sheet.iloc[:, 0])
