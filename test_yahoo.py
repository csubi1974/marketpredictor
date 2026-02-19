import pandas as pd
import requests
import io

def test_scrape(current_day):
    url = f"https://finance.yahoo.com/calendar/economic?day={current_day}&region=US"
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
    response = requests.get(url, headers=headers, timeout=10)
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        try:
            tables = pd.read_html(io.StringIO(response.text))
            if tables:
                print(tables[0].head())
            else:
                print("No tables found")
        except Exception as e:
            print(f"Error parsing: {e}")

test_scrape("2026-02-23")
