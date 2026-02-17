import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import yfinance as yf
from datetime import datetime, timedelta
import streamlit as st
import joblib
import os
import plotly.graph_objects as go

def calculate_rsi(data, window=14):
  delta = data.diff()
  gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
  loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
  rs = gain / loss
  return 100 - (100 / (1 + rs))

@st.cache_data
def load_data(start_date, end_date):
  try:
      data = yf.download("^GSPC", start=start_date, end=end_date)
      data['Returns'] = data['Close'].pct_change()
      data['MA5'] = data['Close'].rolling(window=5).mean()
      data['MA20'] = data['Close'].rolling(window=20).mean()
      data['MA50'] = data['Close'].rolling(window=50).mean()
      data['RSI'] = calculate_rsi(data['Close'])
      data['Volatility'] = data['Returns'].rolling(window=20).std()
      data['Volume_Change'] = data['Volume'].pct_change()
      data['Target'] = (data['Returns'].shift(-1) > 0).astype(int)
      return data.dropna()
  except Exception as e:
      st.error(f"Error al cargar los datos: {e}")
      return None

def train_model(data):
  features = ['Returns', 'MA5', 'MA20', 'MA50', 'RSI', 'Volatility', 'Volume_Change']
  X = data[features]
  y = data['Target']
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
  
  param_grid = {
      'n_estimators': [100, 200, 300],
      'max_depth': [None, 10, 20, 30],
      'min_samples_split': [2, 5, 10],
      'min_samples_leaf': [1, 2, 4]
  }
  
  rf = RandomForestClassifier(random_state=42)
  grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, n_jobs=-1, verbose=1)
  grid_search.fit(X_train, y_train)
  
  best_model = grid_search.best_estimator_
  return best_model, X_test, y_test

def evaluate_model(model, X_test, y_test):
  y_pred = model.predict(X_test)
  accuracy = accuracy_score(y_test, y_pred)
  precision = precision_score(y_test, y_pred)
  recall = recall_score(y_test, y_pred)
  f1 = f1_score(y_test, y_pred)
  return accuracy, precision, recall, f1

def predict_next_day(model, data):
  features = ['Returns', 'MA5', 'MA20', 'MA50', 'RSI', 'Volatility', 'Volume_Change']
  last_data = data[features].iloc[-1].values.reshape(1, -1)
  prediction = model.predict(last_data)
  prediction_proba = model.predict_proba(last_data)
  next_day = data.index[-1] + pd.Timedelta(days=1)
  return prediction[0], prediction_proba[0], next_day

def save_model(model, filename='sp500_model.joblib'):
  joblib.dump(model, filename)
  st.success(f"Modelo guardado como {filename}")

def load_model(filename='sp500_model.joblib'):
  if os.path.exists(filename):
      return joblib.load(filename)
  return None

def perform_backtesting(model, data, days=60):
  features = ['Returns', 'MA5', 'MA20', 'MA50', 'RSI', 'Volatility', 'Volume_Change']
  
  # Obtener los últimos 60 días de datos
  test_data = data.iloc[-days:]
  
  predictions = []
  actual_values = []
  
  for i in range(len(test_data) - 1):
      X = test_data[features].iloc[i].values.reshape(1, -1)
      y_pred = model.predict(X)[0]
      y_actual = test_data['Target'].iloc[i]
      
      predictions.append(y_pred)
      actual_values.append(y_actual)
  
  accuracy = accuracy_score(actual_values, predictions)
  precision = precision_score(actual_values, predictions)
  recall = recall_score(actual_values, predictions)
  f1 = f1_score(actual_values, predictions)
  
  return accuracy, precision, recall, f1, predictions, actual_values

def main():
  st.title('Predicción del S&P 500')

  st.sidebar.header('Parámetros')
  end_date = datetime.now()
  start_date = st.sidebar.date_input('Fecha de inicio', value=end_date - timedelta(days=1460))
  end_date = st.sidebar.date_input('Fecha de fin', value=end_date)

  data = load_data(start_date, end_date)
  
  if data is None:
      st.error("No se pudieron cargar los datos. Por favor, intente de nuevo.")
      return

  # Obtener los precios de cierre y apertura del último día de negociación disponible
  last_trading_day = data.index[-1]
  last_close = data['Close'].iloc[-1]
  last_open = data['Open'].iloc[-1]
  previous_close = data['Close'].iloc[-2]  # Cierre del día anterior

  st.subheader('Último día de negociación disponible')
  col1, col2, col3, col4 = st.columns(4)
  col1.metric("Fecha", last_trading_day.strftime('%Y-%m-%d'))
  col2.metric("Precio de apertura", f"${last_open:.2f}")
  col3.metric("Precio de cierre", f"${last_close:.2f}", f"{(last_close - last_open):.2f}")
  col4.metric("Cambio desde el cierre anterior", f"{(last_close - previous_close):.2f}", f"{((last_close - previous_close) / previous_close * 100):.2f}%")

  model = load_model()

  if st.sidebar.button('Entrenar nuevo modelo'):
      with st.spinner('Cargando datos y entrenando modelo...'):
          model, X_test, y_test = train_model(data)
          save_model(model)

      st.success('Modelo entrenado con éxito!')

  if model:
      accuracy, precision, recall, f1 = evaluate_model(model, data[['Returns', 'MA5', 'MA20', 'MA50', 'RSI', 'Volatility', 'Volume_Change']], data['Target'])

      st.subheader('Métricas del modelo')
      col1, col2, col3, col4 = st.columns(4)
      col1.metric('Accuracy', f'{accuracy:.2f}')
      col2.metric('Precision', f'{precision:.2f}')
      col3.metric('Recall', f'{recall:.2f}')
      col4.metric('F1-score', f'{f1:.2f}')

      prediction, prediction_proba, next_day = predict_next_day(model, data)
      st.subheader(f'Predicción para {next_day.strftime("%Y-%m-%d")}')
      st.write(f"{'Subida' if prediction == 1 else 'Bajada'}")
      st.write(f"Probabilidad: Subida {prediction_proba[1]:.2f}, Bajada {prediction_proba[0]:.2f}")

      st.subheader('Importancia de las características')
      feature_importance = pd.DataFrame({
          'feature': ['Returns', 'MA5', 'MA20', 'MA50', 'RSI', 'Volatility', 'Volume_Change'],
          'importance': model.feature_importances_
      }).sort_values('importance', ascending=False)
      st.bar_chart(feature_importance.set_index('feature'))

      st.subheader('Backtesting (últimos 60 días)')
      accuracy_bt, precision_bt, recall_bt, f1_bt, predictions, actual_values = perform_backtesting(model, data)
      
      col1, col2, col3, col4 = st.columns(4)
      col1.metric('Accuracy', f'{accuracy_bt:.2f}')
      col2.metric('Precision', f'{precision_bt:.2f}')
      col3.metric('Recall', f'{recall_bt:.2f}')
      col4.metric('F1-score', f'{f1_bt:.2f}')
      
      # Visualización mejorada de predicciones vs valores reales
      bt_df = pd.DataFrame({
          'Fecha': data.index[-60:-1],
          'Predicción': predictions,
          'Valor Real': actual_values
      })
      
      # Crear una columna para el color de las barras
      bt_df['Color'] = ['green' if p == a else 'red' for p, a in zip(bt_df['Predicción'], bt_df['Valor Real'])]
      
      # Crear una columna para el texto de las predicciones
      bt_df['Texto'] = ['Subida' if p == 1 else 'Bajada' for p in bt_df['Predicción']]
      
      # Visualización con plotly
      fig = go.Figure()
      fig.add_trace(go.Bar(
          x=bt_df['Fecha'],
          y=bt_df['Predicción'],
          name='Predicción',
          marker_color=bt_df['Color'],
          text=bt_df['Texto'],
          textposition='auto'
      ))
      
      fig.update_layout(
          title='Predicciones del Backtesting',
          xaxis_title='Fecha',
          yaxis_title='Predicción (1: Subida, 0: Bajada)',
          yaxis=dict(tickmode='linear', tick0=0, dtick=1),
          height=500
      )
      
      st.plotly_chart(fig)
      
      # Calcular el porcentaje de predicciones correctas
      correct_predictions = sum([1 for p, a in zip(predictions, actual_values) if p == a])
      accuracy_percentage = (correct_predictions / len(predictions)) * 100
      st.write(f"Porcentaje de predicciones correctas: {accuracy_percentage:.2f}%")

      # Añadir leyenda
      st.write("Leyenda:")
      st.write("- Barra verde: Predicción correcta")
      st.write("- Barra roja: Predicción incorrecta")
      st.write("- 'Subida': El modelo predijo un aumento en el precio")
      st.write("- 'Bajada': El modelo predijo una disminución en el precio")

  st.subheader('Datos históricos')
  st.line_chart(data['Close'])

  st.warning('Este modelo es solo para fines educativos. No se recomienda su uso para tomar decisiones de inversión reales sin una evaluación adicional y asesoramiento profesional.')

if __name__ == '__main__':
  main()

# Created/Modified files during execution:
print("sp500_model.joblib")