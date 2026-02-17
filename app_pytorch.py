import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import yfinance as yf
from datetime import datetime, timedelta
import streamlit as st
import plotly.graph_objects as go

# Cargar datos de Yahoo Finance
@st.cache_data
def load_data(start_date, end_date):
  try:
      spx = yf.Ticker("^GSPC")
      datos = spx.history(start=start_date, end=end_date)

      if datos.empty:
          st.error("No se encontraron datos para el rango de fechas especificado.")
          return None
      
      datos['Returns'] = datos['Close'].pct_change()
      datos['MA5'] = datos['Close'].rolling(window=5).mean()
      datos['MA20'] = datos['Close'].rolling(window=20).mean()
      datos['MA50'] = datos['Close'].rolling(window=50).mean()
      datos['RSI'] = calculate_rsi(datos['Close'])
      datos['Volatility'] = datos['Returns'].rolling(window=20).std()
      datos['Volume_Change'] = datos['Volume'].pct_change()
      datos['Target'] = (datos['Returns'].shift(-1) > 0).astype(int)

      return datos.dropna()
  except Exception as e:
      st.error(f"Error al cargar los datos: {e}")
      return None

# Calcular RSI
def calculate_rsi(data, window=14):
  delta = data.diff()
  gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
  loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
  rs = gain / loss
  return 100 - (100 / (1 + rs))

# Crear conjunto de datos para PyTorch
class StockDataset(Dataset):
  def __init__(self, data, features, target):
      self.data = data
      self.features = features
      self.target = target

  def __len__(self):
      return len(self.data)

  def __getitem__(self, idx):
      features = self.data[self.features].iloc[idx].values
      target = self.data[self.target].iloc[idx]
      return {
          'features': torch.tensor(features, dtype=torch.float32),
          'target': torch.tensor(target, dtype=torch.long)
      }

# Crear modelo de PyTorch
class StockModel(nn.Module):
  def __init__(self, input_dim, hidden_dim, output_dim):
      super(StockModel, self).__init__()
      self.fc1 = nn.Linear(input_dim, hidden_dim)
      self.fc2 = nn.Linear(hidden_dim, output_dim)

  def forward(self, x):
      x = torch.relu(self.fc1(x))
      x = self.fc2(x)
      return x

# Entrenar modelo
def train_model(model, device, loader, optimizer, criterion):
  model.train()
  total_loss = 0
  for batch in loader:
      features = batch['features'].to(device)
      target = batch['target'].to(device)
      optimizer.zero_grad()
      output = model(features)
      loss = criterion(output, target)
      loss.backward()
      optimizer.step()
      total_loss += loss.item()
  return total_loss / len(loader)

# Evaluar modelo
def evaluate_model(model, device, loader, criterion):
  model.eval()
  total_loss = 0
  correct = 0
  with torch.no_grad():
      for batch in loader:
          features = batch['features'].to(device)
          target = batch['target'].to(device)
          output = model(features)
          loss = criterion(output, target)
          total_loss += loss.item()
          _, predicted = torch.max(output, 1)
          correct += (predicted == target).sum().item()
  accuracy = correct / len(loader.dataset)
  return total_loss / len(loader), accuracy

# Predecir próximo día hábil de mercado
def predict_next_day(model, device, data):
  features = data[['Returns', 'MA5', 'MA20', 'MA50', 'RSI', 'Volatility', 'Volume_Change']].iloc[-1].values
  features = torch.tensor(features, dtype=torch.float32).to(device)
  output = model(features)
  _, predicted = torch.max(output, 0)
  return predicted.item()

# Función principal de la app Streamlit
def main():
  st.title('Predicción del S&P 500')

  st.sidebar.header('Parámetros')
  end_date = datetime.now()
  start_date = st.sidebar.date_input('Fecha de inicio', value=end_date - timedelta(days=1460))
  end_date = st.sidebar.date_input('Fecha de fin', value=end_date)

  data = load_data(start_date, end_date)

  if data is None or data.empty:
      st.error("No se pudieron cargar los datos. Por favor, intente de nuevo.")
      return

  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

  model = StockModel(input_dim=7, hidden_dim=128, output_dim=2)
  model.to(device)

  criterion = nn.CrossEntropyLoss()
  optimizer = optim.Adam(model.parameters(), lr=0.001)

  dataset = StockDataset(data, ['Returns', 'MA5', 'MA20', 'MA50', 'RSI', 'Volatility', 'Volume_Change'], 'Target')
  loader = DataLoader(dataset, batch_size=32, shuffle=True)

  if st.sidebar.button('Entrenar modelo'):
      for epoch in range(100):
          loss = train_model(model, device, loader, optimizer, criterion)
          st.write(f'Epoch {epoch+1}, Loss: {loss:.4f}')

  if st.sidebar.button('Evaluar modelo'):
      loss, accuracy = evaluate_model(model, device, loader, criterion)
      st.write(f'Loss: {loss:.4f}, Accuracy: {accuracy:.4f}')

  if st.sidebar.button('Predecir próximo día hábil de mercado'):
      prediction = predict_next_day(model, device, data)
      st.write(f'Predicción: {prediction}')

if __name__ == '__main__':
  main()