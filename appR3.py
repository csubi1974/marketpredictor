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

# Cargar datos de Yahoo Finance
@st.cache_data
def load_data(ticker, start_date, end_date):
    try:
        stock = yf.Ticker(ticker)
        datos = stock.history(start=start_date, end=end_date)

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

# Encontrar el siguiente día de mercado abierto
def get_next_market_day(last_trading_day):
    next_day = last_trading_day + timedelta(days=1)
    
    while next_day.weekday() >= 5:  # 5 = Sábado, 6 = Domingo
        next_day += timedelta(days=1)
    
    return next_day

# Predecir próximo día hábil de mercado
def predict_next_day(model, data):
    features = ['Returns', 'MA5', 'MA20', 'MA50', 'RSI', 'Volatility', 'Volume_Change']
    
    last_data = data[features].iloc[-1].values.reshape(1, -1)
    last_data_df = pd.DataFrame(last_data, columns=features)

    prediction = model.predict(last_data_df)
    prediction_proba = model.predict_proba(last_data_df)
    
    last_trading_day = data.index[-1]
    next_day = get_next_market_day(last_trading_day)
    
    return prediction[0], prediction_proba[0], next_day

# Guardar el modelo entrenado
def save_model(model, filename='sp500_model.joblib'):
    joblib.dump(model, filename)
    st.success(f"Modelo guardado como {filename}")

# Cargar un modelo existente
def load_model(filename='sp500_model.joblib'):
    if os.path.exists(filename):
        return joblib.load(filename)
    return None

# Realizar backtesting
def perform_backtesting(model, data, days=60):
    features = ['Returns', 'MA5', 'MA20', 'MA50', 'RSI', 'Volatility', 'Volume_Change']
    
    original_dates = data.index[-days:]
    
    test_data = data.iloc[-days:]
    
    predictions = []
    actual_values = []
    
    for i in range(len(test_data)):
        X = test_data[features].iloc[i].values.reshape(1, -1)
        X_df = pd.DataFrame(X, columns=features)
        y_pred = model.predict(X_df)[0]
        y_actual = test_data['Target'].iloc[i]

        predictions.append(y_pred)
        actual_values.append(y_actual)
    
    backtest_df = pd.DataFrame({
        'Fecha': original_dates,
        'Predicción': predictions,
        'Valor Real': actual_values
    })
    
    full_dates = pd.date_range(start=backtest_df['Fecha'].min(), end=backtest_df['Fecha'].max())
    backtest_df = backtest_df.set_index('Fecha').reindex(full_dates).reset_index()
    backtest_df.columns = ['Fecha', 'Predicción', 'Valor Real']
    
    backtest_df['Color'] = backtest_df.apply(
        lambda row: 'green' if row['Predicción'] == row['Valor Real'] else 'red', axis=1
    )
    
    return backtest_df

# Generar gráfico de backtesting
def plot_backtesting(backtest_df):
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=backtest_df['Fecha'],
        y=backtest_df['Predicción'].fillna(0),
        name='Predicción',
        marker_color=backtest_df['Color'].fillna('gray'),
        text=['Subida' if p == 1 else 'Bajada' for p in backtest_df['Predicción'].fillna(-1)],
        textposition='auto'
    ))

    fig.update_layout(
        title='Predicciones del Backtesting',
        xaxis_title='Fecha',
        yaxis_title='Predicción (1: Subida, 0: Bajada)',
        yaxis=dict(tickmode='linear', tick0=0, dtick=1),
        height=500
    )
    
    return fig

# Función principal de la app Streamlit
def main():
    st.title('Predicción del Mercado de Valores')

    st.sidebar.header('Parámetros')

    # Lista desplegable para seleccionar la acción o índice
    selected_stock = st.sidebar.selectbox('Selecciona una acción o índice:', list(TOP_20_STOCKS.keys()))

    # Fecha de inicio y fin para los datos históricos
    end_date = datetime.now()
    start_date = st.sidebar.date_input('Fecha de inicio', value=end_date - timedelta(days=1460))
    end_date = st.sidebar.date_input('Fecha de fin', value=end_date)

    # Obtener el ticker seleccionado
    ticker = TOP_20_STOCKS[selected_stock]

    # Cargar los datos
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
        X_test = data[['Returns', 'MA5', 'MA20', 'MA50', 'RSI', 'Volatility', 'Volume_Change']].iloc[:-60]
        y_test = data['Target'].iloc[:-60]

        accuracy, precision, recall, f1 = evaluate_model(model, X_test, y_test)

        st.subheader('Métricas del modelo')
        col1, col2, col3, col4 = st.columns(4)
        col1.metric('Accuracy', f'{accuracy:.2f}')
        col2.metric('Precision', f'{precision:.2f}')
        col3.metric('Recall', f'{recall:.2f}')
        col4.metric('F1-score', f'{f1:.2f}')

        # Interpretación de las métricas
        st.subheader("Interpretación de las métricas")
        st.write(f"**Accuracy (Exactitud)**: El modelo acierta en el {accuracy*100:.2f}% de las predicciones.")
        st.write(f"**Precision (Precisión)**: El {precision*100:.2f}% de las predicciones positivas fueron correctas.")
        st.write(f"**Recall (Sensibilidad)**: El modelo identificó correctamente el {recall*100:.2f}% de los casos positivos.")
        st.write(f"**F1-score**: El equilibrio entre Precision y Recall es de {f1:.2f}, lo que indica un buen rendimiento global del modelo.")

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

        if len(data) >= 60:
            st.subheader('Backtesting (últimos 60 días)')
            backtest_df = perform_backtesting(model, data, days=60)
            fig = plot_backtesting(backtest_df)
            st.plotly_chart(fig)

            correct_predictions = sum([1 for p, a in zip(backtest_df['Predicción'], backtest_df['Valor Real']) if p == a])
            accuracy_percentage = (correct_predictions / len(backtest_df.dropna())) * 100
            st.write(f"Porcentaje de predicciones correctas: {accuracy_percentage:.2f}%")

            st.write("Leyenda:")
            st.write("- Barra verde: Predicción correcta")
            st.write("- Barra roja: Predicción incorrecta")
            st.write("- Barra gris: Día sin predicciones")

    st.subheader('Datos históricos')
    st.line_chart(data['Close'])

    st.warning('Este modelo es solo para fines educativos. No se recomienda su uso para tomar decisiones de inversión reales sin una evaluación adicional y asesoramiento profesional.')

if __name__ == '__main__':
    main()
