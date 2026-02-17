import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
import plotly.express as px
import json
from stock_qa_assistant import StockQAAssistant

# Configuración de la página
st.set_page_config(
    page_title="Consulta de Precios de Acciones - Alpha Vantage",
    page_icon="📈",
    layout="wide"
)

# API Key de Alpha Vantage
ALPHA_VANTAGE_API_KEY = "Z4LODLV2DNPLO3ED"
BASE_URL = "https://www.alphavantage.co/query"

# Inicializar el asistente de Q&A
qa_assistant = StockQAAssistant(ALPHA_VANTAGE_API_KEY)

# Función para obtener datos de Alpha Vantage
@st.cache_data(ttl=300)  # Cache por 5 minutos
def get_stock_data(symbol, outputsize="compact"):
    """
    Obtiene datos históricos de una acción usando Alpha Vantage API
    outputsize: 'compact' (últimos 100 días) o 'full' (20+ años)
    """
    params = {
        'function': 'TIME_SERIES_DAILY',
        'symbol': symbol,
        'outputsize': outputsize,
        'apikey': ALPHA_VANTAGE_API_KEY
    }
    
    try:
        response = requests.get(BASE_URL, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        # Verificar si hay errores en la respuesta
        if "Error Message" in data:
            return None, f"Error: {data['Error Message']}"
        
        if "Note" in data:
            return None, f"Límite de API alcanzado: {data['Note']}"
            
        if "Information" in data:
            return None, f"Información: {data['Information']}"
        
        # Extraer datos de series temporales
        time_series_key = "Time Series (Daily)"
        if time_series_key not in data:
            return None, "No se encontraron datos de series temporales"
        
        time_series = data[time_series_key]
        
        # Convertir a DataFrame
        df_data = []
        for date, values in time_series.items():
            df_data.append({
                'Date': pd.to_datetime(date),
                'Open': float(values['1. open']),
                'High': float(values['2. high']),
                'Low': float(values['3. low']),
                'Close': float(values['4. close']),
                'Volume': int(values['5. volume'])
            })
        
        df = pd.DataFrame(df_data)
        df = df.sort_values('Date')
        df.set_index('Date', inplace=True)
        
        return df, None
        
    except requests.exceptions.RequestException as e:
        return None, f"Error de conexión: {str(e)}"
    except json.JSONDecodeError as e:
        return None, f"Error al procesar respuesta JSON: {str(e)}"
    except Exception as e:
        return None, f"Error inesperado: {str(e)}"

# Función para obtener información de la empresa
@st.cache_data(ttl=3600)  # Cache por 1 hora
def get_company_overview(symbol):
    """
    Obtiene información general de la empresa
    """
    params = {
        'function': 'OVERVIEW',
        'symbol': symbol,
        'apikey': ALPHA_VANTAGE_API_KEY
    }
    
    try:
        response = requests.get(BASE_URL, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        if "Error Message" in data or not data or len(data) < 3:
            return {'Name': symbol, 'Sector': 'N/A', 'Industry': 'N/A'}
        
        return data
        
    except Exception as e:
        return {'Name': symbol, 'Sector': 'N/A', 'Industry': 'N/A'}

# Función para filtrar datos por período
def filter_data_by_period(df, period):
    """
    Filtra el DataFrame según el período seleccionado
    """
    if df.empty:
        return df
    
    end_date = df.index.max()
    
    if period == "1mo":
        start_date = end_date - timedelta(days=30)
    elif period == "3mo":
        start_date = end_date - timedelta(days=90)
    elif period == "6mo":
        start_date = end_date - timedelta(days=180)
    elif period == "1y":
        start_date = end_date - timedelta(days=365)
    elif period == "2y":
        start_date = end_date - timedelta(days=730)
    elif period == "5y":
        start_date = end_date - timedelta(days=1825)
    else:  # "full" o cualquier otro valor
        return df
    
    return df[df.index >= start_date]

# Título principal
st.title("📈 Consulta de Precios de Acciones - Alpha Vantage")

# Crear pestañas para diferentes funcionalidades
tab1, tab2 = st.tabs(["📊 Análisis de Acciones", "🤖 Asistente Q&A"])

with tab2:
    st.header("🤖 Asistente de Preguntas y Respuestas Financiero")
    st.markdown("Haz preguntas sobre acciones, empresas y mercados financieros en lenguaje natural.")
    
    # Ejemplos de preguntas
    with st.expander("💡 Ejemplos de preguntas que puedes hacer"):
        st.markdown("""
        **Preguntas sobre precios:**
        - ¿Cuál es el precio actual de AAPL?
        - ¿Cómo está TSLA hoy?
        - ¿Cuánto vale GOOGL?
        
        **Preguntas sobre empresas:**
        - Información sobre la empresa MSFT
        - ¿En qué sector está NVDA?
        - Datos de la compañía META
        
        **Preguntas sobre noticias:**
        - Últimas noticias de AMZN
        - Noticias recientes de TSLA
        
        **Análisis técnico:**
        - RSI de AAPL
        - Indicadores técnicos de GOOGL
        
        **Comparaciones:**
        - Comparar AAPL vs MSFT
        - ¿Cuál es mejor TSLA o NVDA?
        """)
    
    # Chat interface
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    # Mostrar historial de chat
    for i, (question, answer) in enumerate(st.session_state.chat_history):
        with st.container():
            st.markdown(f"**🙋 Pregunta {i+1}:** {question}")
            st.markdown(f"**🤖 Respuesta:** {answer}")
            st.markdown("---")
    
    # Input para nueva pregunta
    user_question = st.text_input(
        "Haz tu pregunta sobre finanzas:",
        placeholder="Ej: ¿Cuál es el precio actual de AAPL?",
        key="qa_input"
    )
    
    col1, col2 = st.columns([1, 4])
    
    with col1:
        if st.button("🚀 Preguntar", type="primary"):
            if user_question:
                with st.spinner("Procesando tu pregunta..."):
                    try:
                        answer = qa_assistant.answer_question(user_question)
                        st.session_state.chat_history.append((user_question, answer))
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error al procesar la pregunta: {str(e)}")
            else:
                st.warning("Por favor, escribe una pregunta.")
    
    with col2:
        if st.button("🗑️ Limpiar Chat"):
            st.session_state.chat_history = []
            st.rerun()

with tab1:
    st.markdown("---")

# Sidebar para controles
st.sidebar.header("Configuración")

# Cuadro de texto para el ticker
ticker = st.sidebar.text_input(
    "Ingresa el símbolo de la acción (ticker):",
    value="AAPL",
    placeholder="Ej: AAPL, GOOGL, MSFT, TSLA"
).upper()

# Lista desplegable para el período de tiempo
period_options = {
    "1 mes": "1mo",
    "3 meses": "3mo", 
    "6 meses": "6mo",
    "1 año": "1y",
    "2 años": "2y",
    "5 años": "5y",
    "Datos completos (20+ años)": "full"
}

selected_period = st.sidebar.selectbox(
    "Selecciona el período de tiempo:",
    options=list(period_options.keys()),
    index=3  # Por defecto "1 año"
)

# Selector de tamaño de datos
data_size = st.sidebar.radio(
    "Cantidad de datos:",
    options=["Compacto (últimos 100 días)", "Completo (20+ años)"],
    index=0
)

outputsize = "compact" if "Compacto" in data_size else "full"

# Botón para obtener datos
if st.sidebar.button("📊 Obtener Datos", type="primary"):
    if ticker:
        try:
            # Mostrar spinner mientras se cargan los datos
            with st.spinner(f'Obteniendo datos para {ticker} desde Alpha Vantage...'):
                # Obtener datos históricos
                hist_data, error = get_stock_data(ticker, outputsize)
                
                if error:
                    st.error(f"❌ {error}")
                    
                    # Proporcionar información específica sobre errores
                    if "Invalid API call" in error or "Error Message" in error:
                        st.info("🔍 **Este error generalmente ocurre cuando:**")
                        st.info("• El ticker no existe o está mal escrito")
                        st.info("• El símbolo no está disponible en Alpha Vantage")
                        
                    elif "API call frequency" in error or "límite" in error.lower():
                        st.info("⏰ **Límite de API alcanzado**")
                        st.info("• Alpha Vantage tiene límites de 5 llamadas por minuto")
                        st.info("• Espera un momento antes de hacer otra consulta")
                        
                    elif "timeout" in error.lower() or "conexión" in error.lower():
                        st.info("🌐 **Problema de conectividad**")
                        st.info("• Verifica tu conexión a internet")
                        st.info("• Intenta nuevamente en unos momentos")
                    
                    st.info("✅ **Ejemplos de tickers válidos para Alpha Vantage:**")
                    st.info("• AAPL, GOOGL, MSFT, TSLA, AMZN, META, NVDA")
                    st.stop()
                
                if hist_data.empty:
                    st.error(f"❌ No se encontraron datos para el ticker '{ticker}'")
                    st.info("💡 **Posibles soluciones:**")
                    st.info("• Verifica que el ticker sea correcto")
                    st.info("• Intenta con un ticker diferente")
                    st.info("• Asegúrate de usar símbolos de acciones estadounidenses")
                    st.stop()
                
                # Filtrar datos según el período seleccionado
                period_code = period_options[selected_period]
                if period_code != "full":
                    filtered_data = filter_data_by_period(hist_data, period_code)
                else:
                    filtered_data = hist_data
                
                # Obtener información de la empresa
                company_info = get_company_overview(ticker)
                
            # Mostrar información de la empresa
            col1, col2, col3 = st.columns(3)
            
            with col1:
                company_name = company_info.get('Name', ticker)
                st.metric(
                    label="Empresa",
                    value=company_name
                )
            
            with col2:
                current_price = filtered_data['Close'].iloc[-1]
                prev_price = filtered_data['Close'].iloc[-2] if len(filtered_data) > 1 else current_price
                price_change = current_price - prev_price
                st.metric(
                    label="Precio Actual",
                    value=f"${current_price:.2f}",
                    delta=f"{price_change:.2f}"
                )
            
            with col3:
                sector = company_info.get('Sector', 'N/A')
                st.metric(
                    label="Sector",
                    value=sector
                )
            
            st.markdown("---")
            
            # Gráfico de precios
            st.subheader(f"📈 Evolución del Precio - {ticker}")
            
            fig = go.Figure()
            
            # Agregar línea de precio de cierre
            fig.add_trace(go.Scatter(
                x=filtered_data.index,
                y=filtered_data['Close'],
                mode='lines',
                name='Precio de Cierre',
                line=dict(color='#1f77b4', width=2)
            ))
            
            # Configurar el layout del gráfico
            fig.update_layout(
                title=f'Precio de {ticker} - {selected_period}',
                xaxis_title='Fecha',
                yaxis_title='Precio (USD)',
                hovermode='x unified',
                showlegend=True,
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Estadísticas adicionales
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📊 Estadísticas del Período")
                stats_df = pd.DataFrame({
                    'Métrica': ['Precio Máximo', 'Precio Mínimo', 'Precio Promedio', 'Volatilidad (%)'],
                    'Valor': [
                        f"${filtered_data['Close'].max():.2f}",
                        f"${filtered_data['Close'].min():.2f}",
                        f"${filtered_data['Close'].mean():.2f}",
                        f"{filtered_data['Close'].std():.2f}"
                    ]
                })
                st.dataframe(stats_df, hide_index=True, use_container_width=True)
            
            with col2:
                st.subheader("📋 Datos Recientes")
                # Mostrar los últimos 10 días
                recent_data = filtered_data[['Open', 'High', 'Low', 'Close', 'Volume']].tail(10)
                recent_data = recent_data.round(2)
                st.dataframe(recent_data, use_container_width=True)
            
            # Gráfico de volumen
            st.subheader("📊 Volumen de Transacciones")
            
            fig_volume = px.bar(
                x=filtered_data.index,
                y=filtered_data['Volume'],
                title=f'Volumen de Transacciones - {ticker}',
                labels={'x': 'Fecha', 'y': 'Volumen'}
            )
            
            fig_volume.update_layout(height=300)
            st.plotly_chart(fig_volume, use_container_width=True)
            
            # Información adicional de la empresa
            if company_info and len(company_info) > 3:
                st.markdown("---")
                st.subheader("🏢 Información de la Empresa")
                
                info_col1, info_col2 = st.columns(2)
                
                with info_col1:
                    st.write(f"**Industria:** {company_info.get('Industry', 'N/A')}")
                    st.write(f"**País:** {company_info.get('Country', 'N/A')}")
                    st.write(f"**Capitalización de Mercado:** ${company_info.get('MarketCapitalization', 'N/A')}")
                
                with info_col2:
                    st.write(f"**P/E Ratio:** {company_info.get('PERatio', 'N/A')}")
                    st.write(f"**Dividend Yield:** {company_info.get('DividendYield', 'N/A')}")
                    st.write(f"**52 Week High:** ${company_info.get('52WeekHigh', 'N/A')}")
            
            # Opción para descargar datos
            st.markdown("---")
            st.subheader("💾 Descargar Datos")
            
            csv = filtered_data.to_csv()
            st.download_button(
                label="📥 Descargar datos como CSV",
                data=csv,
                file_name=f"{ticker}_{selected_period}_alpha_vantage.csv",
                mime="text/csv"
            )
            
        except Exception as e:
            st.error(f"❌ Error inesperado: {str(e)}")
            st.info("🔧 **Consejos:**")
            st.info("• Verifica tu conexión a internet")
            st.info("• Intenta con un ticker diferente")
            st.info("• Espera un momento antes de hacer otra consulta")
    else:
        st.warning("Por favor, ingresa un símbolo de acción (ticker).")

# Información adicional en el sidebar
st.sidebar.markdown("---")
st.sidebar.markdown("### ℹ️ Información")
st.sidebar.markdown("""
**Ejemplos de tickers populares:**
- AAPL (Apple)
- GOOGL (Google)
- MSFT (Microsoft)
- TSLA (Tesla)
- AMZN (Amazon)
- META (Meta/Facebook)
- NVDA (NVIDIA)
- SPY (S&P 500 ETF)
""")

st.sidebar.markdown("---")
st.sidebar.markdown("### ⚠️ Límites de API")
st.sidebar.markdown("""
**Alpha Vantage Free Tier:**
- 5 llamadas por minuto
- 500 llamadas por día
""")

st.sidebar.markdown("---")
st.sidebar.markdown("*Datos proporcionados por Alpha Vantage*")

# Mensaje inicial si no se han cargado datos
if 'ticker' not in st.session_state:
    st.info("👈 Ingresa un ticker en el panel lateral y haz clic en 'Obtener Datos' para comenzar.")
    st.info("🔑 **Usando Alpha Vantage API** - Datos financieros confiables y actualizados")