import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import yfinance as yf
from datetime import datetime, timedelta
import streamlit as st

# Lista de las 20 acciones más importantes (puedes modificar esta lista según tus preferencias)
TOP_20_STOCKS = {
    'S&P 500 (SPX)': '^GSPC',
    'SPY ETF': 'SPY',
    'QQQ ETF': 'QQQ',
    'Apple (AAPL)': 'AAPL',
    'Microsoft (MSFT)': 'MSFT',
    'Amazon (AMZN)': 'AMZN',
    'Tesla (TSLA)': 'TSLA',
    'Alphabet (GOOGL)': 'GOOGL',
    'Meta (META)': 'META',
    'Nvidia (NVDA)': 'NVDA',
    'Berkshire Hathaway (BRK.B)': 'BRK-B',
    'JPMorgan Chase (JPM)': 'JPM',
    'Johnson & Johnson (JNJ)': 'JNJ',
    'Visa (V)': 'V',
    'Procter & Gamble (PG)': 'PG',
    'UnitedHealth (UNH)': 'UNH',
    'ExxonMobil (XOM)': 'XOM',
    'Walmart (WMT)': 'WMT',
    'Mastercard (MA)': 'MA',
    'Chevron (CVX)': 'CVX'
}

# Calcular RSI
def calculate_rsi(data, window=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

# Cargar datos de Yahoo Finance
@st.cache_data
def load_data(ticker, start_date, end_date):
    stock = yf.Ticker(ticker)
    datos = stock.history(start=start_date, end=end_date, auto_adjust=False)
    
    # Calcular cambios porcentuales y volatilidad
    datos['Returns'] = datos['Close'].pct_change()
    datos['Volatility'] = datos['Returns'].rolling(window=20).std()
    
    # Cambio en volumen
    datos['Volume_Change'] = datos['Volume'].pct_change()
    
    # Calcular medias móviles (MA5, MA20, MA50)
    datos['MA5'] = datos['Close'].rolling(window=5).mean()
    datos['MA20'] = datos['Close'].rolling(window=20).mean()
    datos['MA50'] = datos['Close'].rolling(window=50).mean()
    
    # Calcular el RSI (Índice de Fuerza Relativa)
    datos['RSI'] = calculate_rsi(datos['Close'])
    
    # Definir el target (si sube o baja el día siguiente)
    datos['Target'] = (datos['Close'].shift(-1) > datos['Close']).astype(int)

    return datos.dropna()

# Mostrar cálculos asociados al modelo de predicción
def show_model_calculations(data):
    st.subheader('1. Selección de Características (Features)')
    st.write("""
        Para el modelo de predicción, seleccionamos las siguientes características:
        - **Returns**: El retorno porcentual diario.
        - **Volatility**: La volatilidad calculada sobre los retornos en los últimos 20 días.
        - **Volume_Change**: El cambio porcentual en el volumen negociado.
        - **MA5, MA20, MA50**: Medias móviles de 5, 20 y 50 días.
        - **RSI**: Índice de Fuerza Relativa.
    """)
    st.dataframe(data[['Returns', 'Volatility', 'Volume_Change', 'MA5', 'MA20', 'MA50', 'RSI', 'Target']].head(10))

    st.subheader('2. División de Datos')
    st.write("""
        Dividimos los datos en conjunto de **entrenamiento** y **prueba** para entrenar el modelo y evaluar su rendimiento.
        El conjunto de entrenamiento representa el 80% de los datos, mientras que el conjunto de prueba es el 20% restante.
    """)
    
    # Selección de las características
    features = ['Returns', 'Volatility', 'Volume_Change', 'MA5', 'MA20', 'MA50', 'RSI']
    X = data[features]
    y = data['Target']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    st.write(f"Cantidad de datos de entrenamiento: {X_train.shape[0]}")
    st.write(f"Cantidad de datos de prueba: {X_test.shape[0]}")

    st.subheader('3. Entrenamiento del Modelo')
    st.write("""
        Utilizamos un **Random Forest Classifier** como modelo de predicción. Este modelo entrena múltiples árboles de decisión en diferentes subconjuntos del conjunto de datos y luego promedia los resultados para mejorar la precisión y controlar el sobreajuste.
    """)

    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [None, 10],
        'min_samples_split': [2, 5]
    }

    rf = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, n_jobs=-1, verbose=1)
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    st.success('Modelo entrenado con éxito!')

    st.subheader('4. Evaluación del Modelo')
    st.write("""
        Después del entrenamiento, evaluamos el modelo utilizando las métricas de **exactitud**, **precisión**, **recall** y **F1-score** en el conjunto de prueba.
    """)
    
    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    st.write(f"Exactitud (Accuracy): {accuracy:.2f}")
    st.write(f"Precisión (Precision): {precision:.2f}")
    st.write(f"Sensibilidad (Recall): {recall:.2f}")
    st.write(f"F1-score: {f1:.2f}")

    return best_model

# Generar predicciones para todo el conjunto de datos
def generate_predictions(data, model):
    features = ['Returns', 'Volatility', 'Volume_Change', 'MA5', 'MA20', 'MA50', 'RSI']
    data['Prediction'] = model.predict(data[features])
    return data

# Descargar datos con predicciones
def download_predictions(data, ticker):
    csv = data.to_csv(index=True)
    st.download_button(
        label="Descargar datos con predicciones",
        data=csv,
        file_name=f'{ticker}_predicciones.csv',
        mime='text/csv'
    )

# Función principal de la app Streamlit
def main():
    st.title('Predicción del Mercado de Valores con Explicación del Modelo')

    st.sidebar.header('Parámetros')
    selected_stock = st.sidebar.selectbox('Selecciona una acción o índice:', list(TOP_20_STOCKS.keys()))

    end_date = datetime.now()
    start_date = st.sidebar.date_input('Fecha de inicio', value=end_date - timedelta(days=1460))
    end_date = st.sidebar.date_input('Fecha de fin', value=end_date)

    ticker = TOP_20_STOCKS[selected_stock]
    data = load_data(ticker, start_date, end_date)

    if data is None or data.empty:
        st.error("No se pudieron cargar los datos. Por favor, intente de nuevo.")
        return

    st.subheader(f'Datos cargados para {selected_stock}')
    st.write(data.head())

    # Mostrar cálculos asociados al modelo de predicción
    best_model = show_model_calculations(data)

    # Generar predicciones para todo el conjunto de datos
    data_with_predictions = generate_predictions(data, best_model)

    st.subheader('Datos con Predicciones')
    st.write(data_with_predictions[['Close', 'Prediction']].head())

    # Descargar los datos con predicciones
    download_predictions(data_with_predictions, ticker)

    st.warning('Este análisis es solo para fines educativos y no debe ser utilizado como consejo financiero.')

if __name__ == '__main__':
    main()
