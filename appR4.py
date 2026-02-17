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

# Cargar datos de Yahoo Finance con precios no ajustados
@st.cache_data
def load_data(ticker, start_date, end_date):
    try:
        stock = yf.Ticker(ticker)
        # auto_adjust=False asegura que los precios no estén ajustados por dividendos o splits
        datos = stock.history(start=start_date, end=end_date, auto_adjust=False)

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

        # Predecimos la subida o bajada del precio al día siguiente
        datos['Target'] = (datos['Close'].shift(-1) > datos['Close']).astype(int)

        # Asegurarnos que las fechas en el índice no tengan información de zona horaria (naive)
        datos.index = datos.index.tz_convert('America/New_York')

        return datos.dropna()
    except Exception as e:
        st.error(f"Error al cargar los datos: {e}")
        return None


# Entrenar modelo
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

# Evaluar modelo
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
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

    if selected_date.tzinfo is None:
        selected_date = selected_date.tz_localize('America/New_York')
    else:
        selected_date = selected_date.tz_convert('America/New_York')

    # Filtrar los datos hasta la fecha seleccionada
    data_until_selected = data.loc[:selected_date]

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
    if selected_date.tzinfo is None:
        selected_date = selected_date.tz_localize('America/New_York')
    else:
        selected_date = selected_date.tz_convert('America/New_York')

    # Definir el rango de 5 días antes y 5 días después
    start_range = selected_date - timedelta(days=5)
    end_range = selected_date + timedelta(days=5)

    # Filtrar los datos para el rango de 5 días antes y después, eliminando los sábados y domingos
    range_data = data.loc[start_range:end_range]
    range_data = range_data[range_data.index.weekday < 5]  # Filtrar los sábados (5) y domingos (6)

    # Convertir las fechas a categorías para eliminar huecos en el gráfico
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
    if selected_date in range_data.index:
        selected_day = range_data.loc[selected_date]
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
    data['Correcto'] = np.where(data['Prediction'] == data['Real'], 'Correcto', 'Erróneo')
    results = data[['Open', 'Close', 'Prediction', 'Real', 'Correcto']]
    return results

# Opción de descarga de CSV
def download_csv(dataframe, ticker):
    csv = dataframe.to_csv(index=True)
    st.download_button(label="Descargar CSV con Predicciones",
                       data=csv,
                       file_name=f'{ticker}_predicciones.csv',
                       mime='text/csv')

# Guardar el modelo entrenado
def save_model(model, filename='sp500_model.joblib'):
    joblib.dump(model, filename)
    st.success(f"Modelo guardado como {filename}")

# Cargar un modelo existente
def load_model(filename='sp500_model.joblib'):
    if os.path.exists(filename):
        return joblib.load(filename)
    return None

# Función principal de la app Streamlit
def main():
    st.title('Predicción del Mercado de Valores')

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

    last_trading_day = data.index[-1]
    last_close = data['Close'].iloc[-1]
    last_open = data['Open'].iloc[-1]
    previous_close = data['Close'].iloc[-2]

    st.subheader(f'Último día de negociación disponible para {selected_stock}')
    col1, col2, col3, col4 = st.columns(4)
    formatted_date = last_trading_day.strftime('%Y-%m-%d')
    col1.metric("Fecha", formatted_date)
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
