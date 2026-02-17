import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
import plotly.express as px

# Configuración de la página
st.set_page_config(
    page_title="Consulta de Precios de Acciones",
    page_icon="📈",
    layout="wide"
)

# Título principal
st.title("📈 Consulta de Precios de Acciones")
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
    "10 años": "10y",
    "Máximo disponible": "max"
}

selected_period = st.sidebar.selectbox(
    "Selecciona el período de tiempo:",
    options=list(period_options.keys()),
    index=3  # Por defecto "1 año"
)

# Botón para obtener datos
if st.sidebar.button("📊 Obtener Datos", type="primary"):
    if ticker:
        try:
            # Mostrar spinner mientras se cargan los datos
            with st.spinner(f'Obteniendo datos para {ticker}...'):
                # Crear objeto ticker
                stock = yf.Ticker(ticker)
                
                # Obtener datos históricos
                period_code = period_options[selected_period]
                hist_data = stock.history(period=period_code)
                
                # Verificar si se obtuvieron datos históricos
                if hist_data.empty:
                    st.error(f"❌ No se encontraron datos para el ticker '{ticker}'")
                    st.info("💡 **Posibles soluciones:**")
                    st.info("• Verifica que el ticker sea correcto (ej: AAPL, GOOGL)")
                    st.info("• Algunos tickers requieren sufijos (.MX para México, .L para Londres)")
                    st.info("• Intenta con un ticker diferente")
                    st.stop()
                
                # Obtener información de la empresa con manejo de errores
                try:
                    info = stock.info
                    # Verificar si la información está disponible
                    if not info or len(info) < 3:
                        info = {'longName': ticker, 'sector': 'N/A'}
                except Exception as info_error:
                    st.warning(f"⚠️ No se pudo obtener información detallada de la empresa: {str(info_error)}")
                    info = {'longName': ticker, 'sector': 'N/A'}
                
            if not hist_data.empty:
                # Información de la empresa
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric(
                        label="Empresa",
                        value=info.get('longName', ticker)
                    )
                
                with col2:
                    current_price = hist_data['Close'].iloc[-1]
                    prev_price = hist_data['Close'].iloc[-2] if len(hist_data) > 1 else current_price
                    price_change = current_price - prev_price
                    st.metric(
                        label="Precio Actual",
                        value=f"${current_price:.2f}",
                        delta=f"{price_change:.2f}"
                    )
                
                with col3:
                    sector = info.get('sector', 'N/A')
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
                    x=hist_data.index,
                    y=hist_data['Close'],
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
                            f"${hist_data['Close'].max():.2f}",
                            f"${hist_data['Close'].min():.2f}",
                            f"${hist_data['Close'].mean():.2f}",
                            f"{hist_data['Close'].std():.2f}"
                        ]
                    })
                    st.dataframe(stats_df, hide_index=True, use_container_width=True)
                
                with col2:
                    st.subheader("📋 Datos Recientes")
                    # Mostrar los últimos 10 días
                    recent_data = hist_data[['Open', 'High', 'Low', 'Close', 'Volume']].tail(10)
                    recent_data = recent_data.round(2)
                    st.dataframe(recent_data, use_container_width=True)
                
                # Gráfico de volumen
                st.subheader("📊 Volumen de Transacciones")
                
                fig_volume = px.bar(
                    x=hist_data.index,
                    y=hist_data['Volume'],
                    title=f'Volumen de Transacciones - {ticker}',
                    labels={'x': 'Fecha', 'y': 'Volumen'}
                )
                
                fig_volume.update_layout(height=300)
                st.plotly_chart(fig_volume, use_container_width=True)
                
                # Opción para descargar datos
                st.markdown("---")
                st.subheader("💾 Descargar Datos")
                
                csv = hist_data.to_csv()
                st.download_button(
                    label="📥 Descargar datos como CSV",
                    data=csv,
                    file_name=f"{ticker}_{selected_period}_datos.csv",
                    mime="text/csv"
                )
                
            else:
                st.error(f"No se pudieron obtener datos para el ticker '{ticker}'. Verifica que el símbolo sea correcto.")
                
        except Exception as e:
            error_msg = str(e)
            st.error(f"❌ Error al obtener los datos: {error_msg}")
            
            # Proporcionar información específica sobre diferentes tipos de errores
            if "Expecting value: line 1 column 1 (char 0)" in error_msg:
                st.info("🔍 **Este error generalmente ocurre cuando:**")
                st.info("• El ticker no existe o está mal escrito")
                st.info("• Yahoo Finance no tiene datos para este símbolo")
                st.info("• Hay problemas temporales de conectividad")
                st.info("• El ticker requiere un sufijo específico del mercado")
                
                st.info("💡 **Soluciones sugeridas:**")
                st.info("• Verifica la ortografía del ticker")
                st.info("• Prueba con tickers conocidos como: AAPL, GOOGL, MSFT")
                st.info("• Para acciones mexicanas agrega .MX (ej: WALMEX.MX)")
                st.info("• Para acciones europeas agrega el sufijo correspondiente")
                
            elif "No data found" in error_msg:
                st.info("📊 **No hay datos disponibles para este ticker en el período seleccionado**")
                st.info("• Intenta con un período de tiempo diferente")
                st.info("• Verifica que el ticker sea de una empresa que cotiza públicamente")
                
            elif "Connection" in error_msg or "timeout" in error_msg.lower():
                st.info("🌐 **Problema de conectividad**")
                st.info("• Verifica tu conexión a internet")
                st.info("• Intenta nuevamente en unos momentos")
                
            else:
                st.info("🔧 **Consejos generales:**")
                st.info("• Asegúrate de que el ticker sea válido")
                st.info("• Verifica tu conexión a internet")
                st.info("• Intenta con un ticker diferente")
                
            # Mostrar algunos ejemplos de tickers válidos
            st.info("✅ **Ejemplos de tickers válidos:**")
            examples_col1, examples_col2 = st.columns(2)
            with examples_col1:
                st.info("🇺🇸 **Estados Unidos:**\n• AAPL (Apple)\n• GOOGL (Google)\n• MSFT (Microsoft)\n• TSLA (Tesla)")
            with examples_col2:
                st.info("🌍 **Internacional:**\n• ASML (Países Bajos)\n• SAP (Alemania)\n• NESN.SW (Suiza)\n• 7203.T (Toyota - Japón)")
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
st.sidebar.markdown("*Datos proporcionados por Yahoo Finance*")

# Mensaje inicial si no se han cargado datos
if 'ticker' not in st.session_state:
    st.info("👈 Ingresa un ticker en el panel lateral y haz clic en 'Obtener Datos' para comenzar.")