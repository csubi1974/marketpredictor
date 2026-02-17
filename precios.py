import yfinance as yf
import streamlit as st
import pandas as pd

# Título de la aplicación
st.title("Precios históricos del S&P 500 y acciones importantes")

# Símbolo del S&P 500 en Yahoo Finanzas
spx = yf.Ticker("^GSPC")

# Lista de las 15 acciones más importantes (puedes modificar los símbolos según sea necesario)
acciones_importantes = [
  "AAPL",  # Apple
  "MSFT",  # Microsoft
  "AMZN",  # Amazon
  "GOOGL", # Alphabet (Google)
  "META",  # Meta (Facebook)
  "TSLA",  # Tesla
  "BRK-B", # Berkshire Hathaway
  "NVDA",  # NVIDIA
  "JPM",   # JPMorgan Chase
  "V",     # Visa
  "JNJ",   # Johnson & Johnson
  "PG",    # Procter & Gamble
  "UNH",   # UnitedHealth Group
  "HD",    # Home Depot
  "DIS",   # Walt Disney
  "^GSPC"  # S&P 500
]

# Crear un selector para elegir la acción
accion_seleccionada = st.selectbox("Selecciona una acción:", acciones_importantes)

# Obtener los datos históricos del último año
try:
  if accion_seleccionada == "^GSPC":
      datos = spx.history(period="1y")
  else:
      ticker = yf.Ticker(accion_seleccionada)
      datos = ticker.history(period="1y")
  
  if not datos.empty:
      # Ordenar los datos por fecha de manera descendente
      datos = datos.sort_index(ascending=False)

      # Crear una nueva columna para las flechas
      flechas = []
      for i in range(len(datos) - 1):
          if datos['Close'].iloc[i] > datos['Close'].iloc[i + 1]:
              flechas.append("📉")  # Bajó
          elif datos['Close'].iloc[i] < datos['Close'].iloc[i + 1]:
              flechas.append("📈")  # Subió
          else:
              flechas.append("➡️")  # Sin cambio

      # Agregar una flecha para el último día
      flechas.append("")  # Sin comparación para el último día

      # Añadir la columna de flechas al DataFrame
      datos['Flecha'] = flechas

      # Formatear los precios a 2 decimales
      datos['Open'] = datos['Open'].round(2)
      datos['Close'] = datos['Close'].round(2)
      datos['High'] = datos['High'].round(2)
      datos['Low'] = datos['Low'].round(2)

      # Resetear el índice para que la fecha no sea el índice del DataFrame
      datos.reset_index(inplace=True)

      # Mostrar los datos en una tabla
      st.write(f"**Precios del último año para {accion_seleccionada} (más reciente primero):**")
      st.dataframe(datos[['Date', 'Open', 'Close', 'High', 'Low', 'Flecha']])
      
      # Mostrar gráfico interactivo
      st.line_chart(datos.set_index('Date')[['Open', 'Close']])
  else:
      st.error("No se encontraron datos.")
except Exception as e:
  st.error(f"Error al obtener los datos: {e}")