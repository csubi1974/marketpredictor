import requests
from bs4 import BeautifulSoup
import pandas as pd

# URL del Put/Call Ratio diario en CBOE
url = "https://www.cboe.com/us/options/market_statistics/daily/"

# Realizar la solicitud GET a la página web
response = requests.get(url)

# Parsear el contenido HTML
soup = BeautifulSoup(response.content, 'html.parser')

# Extraer la tabla que contiene el Put/Call Ratio
table = soup.find('table')

# Convertir la tabla a un DataFrame de Pandas
df = pd.read_html(str(table))[0]

# Mostrar el Put/Call Ratio
print(df.head())
