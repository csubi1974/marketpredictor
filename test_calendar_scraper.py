import requests
import pandas as pd
import io
from datetime import datetime

def test_scraper():
    current_day = datetime.now().strftime("%Y-%m-%d")
    url = f"https://finance.yahoo.com/calendar/economic?day={current_day}&region=US"
    headers = {'User-Agent': 'Mozilla/5.0'}
    try:
        response = requests.get(url, headers=headers, timeout=10)
        print(f"Status Code: {response.status_code}")
        if response.status_code == 200:
            tables = pd.read_html(io.StringIO(response.text))
            if tables:
                print("Table found!")
                df = tables[0]
                print("Columns found:", df.columns.tolist())
                print(df.head())
            else:
                print("No tables found in HTML")
                # print(response.text[:500])
        else:
            print(f"Error: {response.status_code}")
    except Exception as e:
        print(f"Exception: {e}")

if __name__ == "__main__":
    test_scraper()
