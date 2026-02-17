import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import yfinance as yf
import datetime
from datetime import datetime as dt, date as d_type, timedelta
import streamlit as st
import joblib
import os
import plotly.graph_objects as go
import time
import requests
import json
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

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
    # Evitar división por cero
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50) # Valor neutral si no hay datos suficientes

# Función para cargar datos desde Alpha Vantage
def load_data_alpha_vantage(symbol, api_key, outputsize='full'):
    """
    Carga datos históricos desde Alpha Vantage API
    """
    try:
        # Para el S&P 500, Alpha Vantage usa SPX como símbolo
        if symbol == '^GSPC':
            symbol = 'SPX'
        
        url = f'https://www.alphavantage.co/query'
        params = {
            'function': 'TIME_SERIES_DAILY',
            'symbol': symbol,
            'outputsize': outputsize,
            'apikey': api_key
        }
        
        response = requests.get(url, params=params)
        data = response.json()
        
        if 'Error Message' in data:
            st.error(f"Error de Alpha Vantage: {data['Error Message']}")
            return None
            
        if 'Note' in data:
            st.warning(f"Límite de API alcanzado: {data['Note']}")
            return None
            
        if 'Time Series (Daily)' not in data:
            st.error("No se encontraron datos en la respuesta de Alpha Vantage")
            return None
        
        # Convertir datos a DataFrame
        time_series = data['Time Series (Daily)']
        df = pd.DataFrame.from_dict(time_series, orient='index')
        
        # Renombrar columnas para que coincidan con yfinance
        df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        
        # Convertir a tipos numéricos
        for col in df.columns:
            df[col] = pd.to_numeric(df[col])
        
        # Convertir índice a datetime
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        
        return df
        
    except Exception as e:
        st.error(f"Error al cargar datos de Alpha Vantage: {str(e)}")
        return None

# Cargar datos con múltiples fuentes
@st.cache_data
def load_data(ticker, start_date, end_date, data_source='yahoo', alpha_vantage_key=None, max_retries=3):
    """
    Carga datos de múltiples fuentes: Yahoo Finance o Alpha Vantage
    """
    if data_source == 'alpha_vantage' and alpha_vantage_key:
        return load_data_with_alpha_vantage(ticker, start_date, end_date, alpha_vantage_key)
    else:
        return load_data_with_yahoo(ticker, start_date, end_date, max_retries)

def load_data_with_alpha_vantage(ticker, start_date, end_date, api_key):
    """
    Carga datos usando Alpha Vantage API
    """
    try:
        st.info(f"Cargando datos de Alpha Vantage para {ticker}...")
        
        # Cargar datos desde Alpha Vantage
        datos = load_data_alpha_vantage(ticker, api_key)
        
        if datos is None:
            return None
        
        # Filtrar por rango de fechas
        if isinstance(start_date, str):
            start_date = pd.to_datetime(start_date)
        if isinstance(end_date, str):
            end_date = pd.to_datetime(end_date)
            
        datos = datos[(datos.index >= start_date) & (datos.index <= end_date)]
        
        if datos.empty:
            st.error(f"No se encontraron datos para {ticker} en el rango de fechas especificado")
            return None
        
        st.success(f"✓ Datos cargados desde Alpha Vantage: {len(datos)} días de negociación")
        
        # Calcular indicadores técnicos
        datos['Returns'] = datos['Close'].pct_change()
        datos['MA5'] = datos['Close'].rolling(window=5).mean()
        datos['MA20'] = datos['Close'].rolling(window=20).mean()
        datos['MA50'] = datos['Close'].rolling(window=50).mean()
        datos['RSI'] = calculate_rsi(datos['Close'])
        datos['Volatility'] = datos['Returns'].rolling(window=20).std()
        datos['Volume_Change'] = datos['Volume'].pct_change()

        # Predecimos la subida o bajada del precio al día siguiente
        datos['Target'] = (datos['Close'].shift(-1) > datos['Close']).astype(int)

        # Limpiar valores infinitos y nulos
        datos = datos.replace([np.inf, -np.inf], np.nan)
        datos_limpios = datos.dropna()

        if len(datos_limpios) < 100:
            st.warning(f"⚠️ Solo se encontraron {len(datos_limpios)} días válidos. Se recomienda al menos 100 días para un modelo confiable.")

        return datos_limpios
        
    except Exception as e:
        st.error(f"Error al cargar datos con Alpha Vantage: {str(e)}")
        return None

def load_data_with_yahoo(ticker, start_date, end_date, max_retries=3):
    """
    Carga datos de Yahoo Finance usando yf.download con manejo de multi-index
    """
    for attempt in range(max_retries):
        try:
            print(f"DEBUG: Descargando {ticker} de Yahoo Finance... (Intento {attempt + 1})")
            st.info(f"Descargando {ticker} de Yahoo Finance... (Intento {attempt + 1})")

            # Limpieza y formato de fechas
            s_str = start_date.strftime('%Y-%m-%d') if hasattr(start_date, 'strftime') else str(start_date)
            e_str = (end_date + timedelta(days=1)).strftime('%Y-%m-%d') if hasattr(end_date, 'strftime') else str(end_date)

            print(f"DEBUG: Rango {s_str} a {e_str}")

            # Descargar datos
            datos = yf.download(ticker, start=s_str, end=e_str, progress=False, auto_adjust=False)

            if datos is None or datos.empty:
                print(f"DEBUG: Intento {attempt + 1}: No se obtuvieron datos")
                st.warning(f"Intento {attempt + 1}: No se obtuvieron datos para {ticker}")
                time.sleep(2)
                continue

            # MANEJO DE COLUMNAS (yfinance >= 0.2.x o cambios de API)
            print(f"DEBUG: Columnas recibidas: {datos.columns.tolist()}")

            # Si las columnas son multi-index, las aplanamos
            if isinstance(datos.columns, pd.MultiIndex):
                # Caso: ('Open', 'AAPL') -> 'Open'
                datos.columns = datos.columns.get_level_values(0)
            
            # Quitar espacios y estandarizar nombres
            datos.columns = [str(col).strip() for col in datos.columns]

            # Verificar que las columnas necesarias existan
            required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            found_cols = [col for col in required_cols if col in datos.columns]
            
            if len(found_cols) < 5:
                print(f"DEBUG: Faltan columnas. Encontradas: {found_cols}")
                st.error(f"Faltan columnas esenciales: {list(set(required_cols) - set(found_cols))}")
                return None

            st.success(f"✓ {len(datos)} días obtenidos correctamente.")

            # Asegurar tipos numéricos
            for col in required_cols:
                datos[col] = pd.to_numeric(datos[col], errors='coerce')
            
            # Cálculo de indicadores técnicos
            datos['Returns'] = datos['Close'].pct_change()
            datos['MA5'] = datos['Close'].rolling(window=5).mean()
            datos['MA20'] = datos['Close'].rolling(window=20).mean()
            datos['MA50'] = datos['Close'].rolling(window=50).mean()
            datos['RSI'] = calculate_rsi(datos['Close'])
            datos['Volatility'] = datos['Returns'].rolling(window=20).std()
            datos['Volume_Change'] = datos['Volume'].pct_change()
            datos['Target'] = (datos['Close'].shift(-1) > datos['Close']).astype(int)

            if datos.index.tz is not None:
                datos.index = datos.index.tz_convert('America/New_York')

            # Limpiar valores infinitos y nulos
            datos = datos.replace([np.inf, -np.inf], np.nan)
            datos_final = datos.dropna()
            
            print(f"DEBUG: Datos procesados correctamente. Filas final: {len(datos_final)}")
            return datos_final

        except Exception as e:
            print(f"DEBUG: Error en intento {attempt + 1}: {e}")
            st.warning(f"Error en intento {attempt + 1}: {e}")
            if attempt < max_retries - 1:
                time.sleep(2)
                continue
            st.error(f"Error definitivo tras {max_retries} intentos: {e}")
            return None
    return None

# Entrenar modelo
def train_model(data):
    features = ['Returns', 'MA5', 'MA20', 'MA50', 'RSI', 'Volatility', 'Volume_Change']
    
    # Limpieza final antes de entrenar
    df = data.copy()
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=features + ['Target'])
    
    X = df[features]
    y = df['Target']

    if len(df) < 50:
        raise ValueError("No hay suficientes datos limpios para entrenar (mínimo 50 días).")

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

# Evaluar modelo
def evaluate_model(model, X_test, y_test):
    # Asegurar que no haya nulos o inf en el set de prueba
    clean_idx = np.isfinite(X_test).all(axis=1)
    X_test_clean = X_test[clean_idx]
    y_test_clean = y_test[clean_idx]
    
    y_pred = model.predict(X_test_clean)
    accuracy = accuracy_score(y_test_clean, y_pred)
    precision = precision_score(y_test_clean, y_pred, zero_division=0)
    recall = recall_score(y_test_clean, y_pred, zero_division=0)
    f1 = f1_score(y_test_clean, y_pred, zero_division=0)
    return accuracy, precision, recall, f1

# Predecir el día siguiente al último día de negociación
def predict_next_day(model, data):
    features = ['Returns', 'MA5', 'MA20', 'MA50', 'RSI', 'Volatility', 'Volume_Change']

    # Seleccionamos los datos del último día de negociación
    last_trading_day = data.index[-1]
    last_data = data[features].iloc[-1].values.reshape(1, -1)
    
    last_data_df = pd.DataFrame(last_data, columns=features)
    prediction = model.predict(last_data_df)
    prediction_proba = model.predict_proba(last_data_df)
    
    # Calcular el próximo día hábil de negociación
    next_day = last_trading_day + timedelta(days=1)
    while next_day.weekday() >= 5:  # Si es sábado o domingo, saltamos al lunes
        next_day += timedelta(days=1)
    
    return prediction[0], prediction_proba[0], next_day

# Predecir desde una fecha seleccionada para hacer backtesting
def predict_from_date(model, data, selected_date):
    features = ['Returns', 'MA5', 'MA20', 'MA50', 'RSI', 'Volatility', 'Volume_Change']

    # Convertir a Timestamp y quitar zona horaria para evitar errores de comparación
    selected_ts = pd.Timestamp(selected_date).tz_localize(None)
    
    # Asegurar que el índice de los datos sea naive para la comparación
    data_naive = data.copy()
    if data_naive.index.tz is not None:
        data_naive.index = data_naive.index.tz_localize(None)

    # Filtrar los datos hasta la fecha seleccionada
    data_until_selected = data_naive.loc[:selected_ts]

    # Obtener los datos para la predicción del día siguiente
    last_data = data_until_selected[features].iloc[-1].values.reshape(1, -1)
    last_data_df = pd.DataFrame(last_data, columns=features)

    prediction = model.predict(last_data_df)
    prediction_proba = model.predict_proba(last_data_df)

    # Calcular el siguiente día de negociación
    last_trading_day = data_until_selected.index[-1]
    next_day = last_trading_day + timedelta(days=1)
    while next_day.weekday() >= 5:
        next_day += timedelta(days=1)

    return prediction[0], prediction_proba[0], next_day

# Crear gráfico de velas japonesas para 5 días antes y 5 días después de la fecha seleccionada, destacando el día seleccionado
# También eliminamos los sábados y domingos (días no hábiles)
def plot_candlestick_chart(data, selected_date):
    # Asegurar que ambos sean naive (sin zona horaria)
    selected_ts = pd.Timestamp(selected_date).tz_localize(None)
    data_naive = data.copy()
    if data_naive.index.tz is not None:
        data_naive.index = data_naive.index.tz_localize(None)

    # Definir el rango de 5 días antes y 5 días después
    start_range = selected_ts - timedelta(days=5)
    end_range = selected_ts + timedelta(days=5)

    # Filtrar los datos para el rango
    range_data = data_naive.loc[start_range:end_range]
    range_data = range_data[range_data.index.weekday < 5]

    # Convertir las fechas a categorías para eliminar huecos
    range_data['Date'] = range_data.index.astype(str)

    # Crear gráfico de velas japonesas con fechas categóricas
    fig = go.Figure(data=[go.Candlestick(x=range_data['Date'],
                                         open=range_data['Open'],
                                         high=range_data['High'],
                                         low=range_data['Low'],
                                         close=range_data['Close'],
                                         increasing_line_color='green',
                                         decreasing_line_color='red')])

    # Resaltar la vela del día seleccionado
    if selected_ts in range_data.index:
        selected_day = range_data.loc[selected_ts]
        fig.add_trace(go.Candlestick(x=[selected_day['Date']],
                                     open=[selected_day['Open']],
                                     high=[selected_day['High']],
                                     low=[selected_day['Low']],
                                     close=[selected_day['Close']],
                                     increasing_line_color='yellow',
                                     decreasing_line_color='yellow',
                                     line_width=2))

    fig.update_layout(title='Velas Japonesas - 5 Días Antes y Después (Destacando Día Seleccionado)',
                      xaxis_title='Fecha',
                      yaxis_title='Precio',
                      xaxis_rangeslider_visible=False)

    return fig

# Generar DataFrame con los resultados, predicción, valor real y si la predicción fue correcta o errónea
def generate_results(model, data):
    features = ['Returns', 'MA5', 'MA20', 'MA50', 'RSI', 'Volatility', 'Volume_Change']
    data['Prediction'] = model.predict(data[features])
    data['Real'] = data['Target']
    data['Correcto'] = np.where(data['Prediction'] == data['Real'], 'Correcto', 'Erroneo')
    results = data[['Open', 'Close', 'Prediction', 'Real', 'Correcto']]
    return results

# Opción de descarga de CSV con delimitador ';'
def download_csv(dataframe, ticker):
    csv = dataframe.to_csv(index=True, sep=';')  # Usar ';' como delimitador
    st.download_button(label="Descargar CSV con Predicciones",
                       data=csv,
                       file_name=f'{ticker}_predicciones.csv',
                       mime='text/csv')

# Guardar el modelo entrenado con el nombre del ticker
def save_model(model, ticker):
    # Limpiar el ticker para el nombre del archivo
    clean_ticker = ticker.replace('^', '').replace('-', '_')
    filename = f'model_{clean_ticker}.joblib'
    joblib.dump(model, filename)
    st.success(f"Modelo guardado como {filename}")

# Cargar un modelo existente basado en el ticker
def load_model(ticker):
    clean_ticker = ticker.replace('^', '').replace('-', '_')
    filename = f'model_{clean_ticker}.joblib'
    
    # Compatibilidad con el nombre de archivo anterior para el S&P 500
    if ticker == '^GSPC' and not os.path.exists(filename) and os.path.exists('sp500_model.joblib'):
        return joblib.load('sp500_model.joblib')
        
    if os.path.exists(filename):
        return joblib.load(filename)
    return None

# Función principal de la app Streamlit
def main():
    st.title('Predicción del Mercado de Valores')

    # Información inicial
    st.info("📊 Esta aplicación utiliza Machine Learning para predecir movimientos del mercado de valores")

    st.sidebar.header('Parámetros')

    selected_stock = st.sidebar.selectbox('Selecciona una acción o índice:', list(TOP_20_STOCKS.keys()))

    # Selección de fuente de datos - Yahoo Finance por defecto
    st.sidebar.subheader('🔗 Fuente de Datos')
    data_source = st.sidebar.radio(
        'Selecciona la fuente de datos:',
        ['Yahoo Finance', 'Alpha Vantage'],
        index=0,
        help='Yahoo Finance es gratuito y no requiere API key. Alpha Vantage es una alternativa si Yahoo falla.'
    )

    # Campo para API key de Alpha Vantage
    alpha_vantage_key = os.getenv('ALPHA_VANTAGE_API_KEY')
    if data_source == 'Alpha Vantage':
        alpha_vantage_key = st.sidebar.text_input(
            'API Key de Alpha Vantage:',
            value=alpha_vantage_key if alpha_vantage_key else 'Z4LODLV2DNPLO3ED',
            type='password',
            help='Obtén tu API key gratuita en https://www.alphavantage.co/support/#api-key'
        )
        if not alpha_vantage_key:
            st.sidebar.warning('⚠️ Se requiere API key para usar Alpha Vantage')
            st.sidebar.info('📝 Pasos para obtener API key:\n1. Visita https://www.alphavantage.co/support/#api-key\n2. Completa el formulario\n3. Copia la API key aquí')
        else:
            st.sidebar.success('✅ API Key configurada correctamente')

    # Usar un rango de 5 años por defecto
    end_date = dt.now().date()
    default_start = end_date - timedelta(days=1825)  # 5 años
    start_date = st.sidebar.date_input('Fecha de inicio', value=default_start, max_value=end_date)

    # Validar que la fecha de inicio no sea muy antigua
    max_days_back = 3650 # 10 años
    if (end_date - start_date).days > max_days_back:
        st.sidebar.warning(f"⚠️ El rango máximo es de {max_days_back} días (10 años)")
        start_date = end_date - timedelta(days=max_days_back)

    ticker = TOP_20_STOCKS[selected_stock]

    st.sidebar.info(f"📈 Ticker seleccionado: **{ticker}**")
    st.sidebar.info(f"📅 Rango: {start_date} a {end_date}")
    st.sidebar.info(f"📊 Días: {(end_date - start_date).days}")
    st.sidebar.info(f"🔗 Fuente: **{data_source}**")

    # Resetear métricas si cambia el ticker
    if 'last_ticker' not in st.session_state or st.session_state['last_ticker'] != ticker:
        st.session_state['last_ticker'] = ticker
        if 'metrics' in st.session_state:
            del st.session_state['metrics']

    # Cargar datos con mejor manejo de errores
    with st.spinner(f'Cargando datos de {selected_stock} desde {data_source}...'):
        if data_source == 'Alpha Vantage' and alpha_vantage_key:
            data = load_data(ticker, start_date, end_date, 'alpha_vantage', alpha_vantage_key)
        elif data_source == 'Yahoo Finance':
            data = load_data(ticker, start_date, end_date, 'yahoo')
        else:
            data = None
            if data_source == 'Alpha Vantage':
                st.error("❌ Se requiere API key para usar Alpha Vantage")

    if data is None or data.empty:
        st.error("❌ No se pudieron cargar los datos.")
        st.warning("💡 **Sugerencias:**")
        if data_source == 'Yahoo Finance':
            st.write("1. Intenta con otra acción del menú (por ejemplo: **Apple (AAPL)** o **Microsoft (MSFT)**)")
            st.write("2. Verifica tu conexión a internet")
            st.write("3. Intenta con un rango de fechas más corto")
            st.write("4. Si el problema persiste, prueba con **Alpha Vantage** como fuente alternativa")
        else:
            st.write("1. Verifica que tu API key de Alpha Vantage sea correcta")
            st.write("2. Asegúrate de no haber excedido el límite de 500 llamadas diarias")
            st.write("3. Intenta con **Yahoo Finance** como alternativa")
            st.write("4. Verifica tu conexión a internet")

        # Mostrar acciones alternativas recomendadas
        st.info("🔄 **Acciones recomendadas para probar:**")
        recommended = ['Apple (AAPL)', 'Microsoft (MSFT)', 'SPY ETF', 'QQQ ETF']
        for rec in recommended:
            st.write(f"   • {rec}")
        return

    last_trading_day = data.index[-1]
    last_close = data['Close'].iloc[-1]
    last_open = data['Open'].iloc[-1]
    previous_close = data['Close'].iloc[-2]



    st.subheader(f'Último día de negociación disponible para {selected_stock}')
    col1, col2, col3, col4 = st.columns(4)
    formatted_date = last_trading_day.strftime('%Y-%m-%d') 
    col1.metric("Fecha", formatted_date, "")
    col2.metric("Precio de apertura", f"${last_open:.2f}")
    col3.metric("Precio de cierre", f"${last_close:.2f}", f"{(last_close - last_open):.2f}")
    col4.metric("Cambio desde el cierre anterior", f"{(last_close - previous_close):.2f}", f"{((last_close - previous_close) / previous_close * 100):.2f}%")


    model = load_model(ticker)

    if st.sidebar.button('Entrenar nuevo modelo'):
        with st.spinner(f'Entrenando modelo para {ticker}...'):
            model, X_test, y_test = train_model(data)
            save_model(model, ticker)
            
            # Guardar métricas en el estado de la sesión para evitar pérdida al recargar
            st.session_state['metrics'] = evaluate_model(model, X_test, y_test)
            st.success('Modelo entrenado con éxito!')

    if model:
        # Si no acabamos de entrenar, evaluamos con los datos más recientes (pero con cuidado del leakage)
        if 'metrics' not in st.session_state:
            # Dividir datos para evaluación realista
            features = ['Returns', 'MA5', 'MA20', 'MA50', 'RSI', 'Volatility', 'Volume_Change']
            X = data[features]
            y = data['Target']
            _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            st.session_state['metrics'] = evaluate_model(model, X_test, y_test)

        accuracy, precision, recall, f1 = st.session_state['metrics']

        st.subheader(f'Métricas del modelo ({ticker})')
        col1, col2, col3, col4 = st.columns(4)
        col1.metric('Exactitud (Accuracy)', f'{accuracy:.2f}')
        col2.metric('Precisión (Precision)', f'{precision:.2f}')
        col3.metric('Sensibilidad (Recall)', f'{recall:.2f}')
        col4.metric('F1-score', f'{f1:.2f}')

        # Descripción de las métricas
        st.write(f"""
            **Exactitud (Accuracy)**: El modelo acertó en el {accuracy*100:.2f}% de los casos.
            **Precisión (Precision)**: El {precision*100:.2f}% de las veces que el modelo predijo "Subida", fue correcto.
            **Sensibilidad (Recall)**: El modelo identificó correctamente el {recall*100:.2f}% de los casos donde realmente subió.
            **F1-score**: Este equilibrio entre precisión y sensibilidad es de {f1:.2f}, lo que indica un rendimiento general robusto del modelo.
        """)

        # Predicción del siguiente día después del último día de negociación
        if st.sidebar.button('Predecir el siguiente día'):
            prediction, prediction_proba, next_day = predict_next_day(model, data)
            
            st.subheader(f'Predicción para el {next_day.strftime("%Y-%m-%d")}')
            
            # Resaltar más el resultado de la predicción
            if prediction == 1:
                st.markdown(f"<h2 style='color: green;'>Predicción: Subida</h2>", unsafe_allow_html=True)
            else:
                st.markdown(f"<h2 style='color: red;'>Predicción: Bajada</h2>", unsafe_allow_html=True)
            
            st.write(f"Probabilidad: Subida {prediction_proba[1]:.2f}, Bajada {prediction_proba[0]:.2f}")

        # Backtesting desde una fecha seleccionada
        st.sidebar.subheader('Seleccionar fecha para backtesting')
        selected_date = st.sidebar.date_input("Selecciona una fecha", value=data.index[-60].to_pydatetime(), min_value=data.index[0].to_pydatetime(), max_value=data.index[-2].to_pydatetime())

        if st.sidebar.button('Predecir desde la fecha seleccionada'):
            prediction, prediction_proba, next_day = predict_from_date(model, data, pd.Timestamp(selected_date))
            
            st.subheader(f'Predicción para el día {next_day.strftime("%Y-%m-%d")}')
            
            # Resaltar más el resultado de la predicción
            if prediction == 1:
                st.markdown(f"<h2 style='color: green;'>Predicción: Subida</h2>", unsafe_allow_html=True)
            else:
                st.markdown(f"<h2 style='color: red;'>Predicción: Bajada</h2>", unsafe_allow_html=True)

            st.write(f"Probabilidad: Subida {prediction_proba[1]:.2f}, Bajada {prediction_proba[0]:.2f}")

            st.subheader('Gráfico de velas japonesas')
            candlestick_fig = plot_candlestick_chart(data, pd.Timestamp(selected_date))
            st.plotly_chart(candlestick_fig)

        st.subheader('Descargar CSV con predicciones y resultados')
        results_df = generate_results(model, data)
        download_csv(results_df, ticker)

    st.subheader('Datos históricos')
    st.line_chart(data['Close'])

    st.warning('Este modelo es solo para fines educativos. No se recomienda su uso para tomar decisiones de inversión reales sin una evaluación adicional y asesoramiento profesional.')

if __name__ == '__main__':
    main()
