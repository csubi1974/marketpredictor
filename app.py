import pandas as pd
import numpy as np
import re
import io
from sklearn.model_selection import train_test_split, GridSearchCV, TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from dotenv import load_dotenv
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
import sqlite3
from sqlite3 import Error
from datetime import timezone as tz, timedelta as td, time as t_time
from datetime import timezone as tz, timedelta as td, time as t_time
try:
    from dotenv import load_dotenv
    load_dotenv() # Carga variables del archivo .env
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

# Definición global de variables del modelo (Fuente única de verdad)
# Usamos solo variables estacionarias (porcentajes/ratios) para evitar el drift del precio
GLOBAL_FEATURES = [
    'Returns', 'Dist_MA20', 'Dist_MA50', 'Dist_MA200', 
    'RSI', 'Volatility', 'VIX_Change', 'Yield_Change', 
    'MACD_Rel', 'BB_Pos', 'Return_Lag1', 'Nikkei_Return', 'DAX_Return', 
    'Futures_Return', 'Volume_Ratio', 'Momentum_5d', 'DXY_Return', 'NYA_Return'
]

def get_ny_time():
    """Obtiene la hora oficial de Nueva York (EST/EDT)."""
    # Usamos UTC y aplicamos el offset de NY
    # Nota: Simplificamos a UTC-5 (EST) por ser Febrero
    return dt.now(tz(td(hours=-5)))

def check_market_status():
    """Determina si el mercado está abierto, en pre-market o cerrado."""
    ny_now = get_ny_time()
    current_time = ny_now.time()
    weekday = ny_now.weekday()
    
    # Horarios NY
    m_open = t_time(9, 30)
    m_close = t_time(16, 0)
    m_pre = t_time(8, 0)
    
    if weekday >= 5:
        return "🔴 Cerrado (Fin de Semana)", "closed"
    
    if current_time >= m_open and current_time < m_close:
        return "🟢 Mercado Abierto", "open"
    elif current_time >= m_pre and current_time < m_open:
        return "🟡 Pre-Mercado (¡Ventana de Oro!)", "pre"
    else:
        return "🔴 Mercado Cerrado", "closed"

# Calcular RSI
def calculate_rsi(data, window=14):
    if len(data) < window:
        return pd.Series(50, index=data.index)
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    # Evitar división por cero
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return pd.Series(rsi).fillna(50) # Valor neutral si no hay datos suficientes

# Calcular ATR
def calculate_atr(df, window=14):
    if len(df) < window + 1:
        return pd.Series(0, index=df.index)
    high_low = df['High'] - df['Low']
    high_close = (df['High'] - df['Close'].shift()).abs()
    low_close = (df['Low'] - df['Close'].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.rolling(window=window).mean()
    return atr

# Calcular ADX
def calculate_adx(df, window=14):
    if len(df) < window * 2:
        return pd.Series(0, index=df.index)
    plus_dm = df['High'].diff()
    minus_dm = df['Low'].diff().apply(lambda x: -x)
    plus_dm = np.where((plus_dm > minus_dm) & (plus_dm > 0), plus_dm, 0.0)
    minus_dm = np.where((minus_dm > plus_dm) & (minus_dm > 0), minus_dm, 0.0)
    
    high_low = df['High'] - df['Low']
    high_close = (df['High'] - df['Close'].shift()).abs()
    low_close = (df['Low'] - df['Close'].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.rolling(window=window).mean()
    
    plus_di = 100 * (pd.Series(plus_dm, index=df.index).rolling(window=window).mean() / atr.replace(0, np.nan))
    minus_di = 100 * (pd.Series(minus_dm, index=df.index).rolling(window=window).mean() / atr.replace(0, np.nan))
    denom = np.abs(plus_di + minus_di).replace(0, np.nan)
    dx = 100 * ( np.abs(plus_di - minus_di) / denom ).fillna(0)
    adx = dx.rolling(window=window).mean()
    return adx.fillna(0)

def calculate_crash_risk(df):
    """Analiza múltiples factores para determinar el riesgo de una caída sistémica."""
    try:
        last = df.iloc[-1]
        vix_chg = last.get('VIX_Change', 0)
        dist_200 = last.get('Dist_MA200', 0)
        vol = last.get('Volatility', 0)
        rsi = last.get('RSI', 50)
        
        risk_score = 0
        reasons = []

        # 1. Pánico en el VIX
        if vix_chg > 0.10: 
            risk_score += 2
            reasons.append("Pico de volatilidad en el VIX (>10%)")
        
        # 2. Debilidad de Tendencia (Precio bajo MA200)
        if dist_200 < -0.05:
            risk_score += 2
            reasons.append("Debilidad estructural (Bajo MA200)")
        
        # 3. Sobre-extensión Bullish (Burbuja inmediata)
        if dist_200 > 0.15:
            risk_score += 1
            reasons.append("Sobre-extensión alcista (Riesgo de reversión)")

        # 4. Volatilidad del Activo
        if vol > df['Volatility'].mean() * 1.5:
            risk_score += 1
            reasons.append("Volatilidad atípica detectada")

        # Determinar Nivel
        if risk_score >= 4:
            return "EXTREMO (Black Swan Alert)", "#dc3545", "🆘", reasons
        elif risk_score >= 2:
            return "ALTO (Protección de Capital)", "#fd7e14", "⚠️", reasons
        elif risk_score >= 1:
            return "MODERADO (Precaución)", "#ffc107", "🧐", reasons
        else:
            return "BAJO (Entorno Protegido)", "#28a745", "🛡️", ["Condiciones de mercado estables"]
    except:
        return "DESCONOCIDO", "#6c757d", "❓", ["Faltan datos de riesgo"]

# Función para calcular Momentum Intradía (Sniper Entry)
def get_intraday_momentum(ticker):
    """Descarga datos de 5m y calcula el momentum actual incluyendo Pre-market."""
    try:
        t_obj = yf.Ticker(ticker)
        # Pedimos 5 días para asegurar que saltamos el feriado de ayer (Presidents' Day)
        # include_prepost es crucial para ver el pre-market de hoy martes
        # auto_adjust=False para que el precio coincida con el nominal de la web (sin descontar dividendos)
        data_5m = t_obj.history(period="5d", interval="5m", prepost=True, auto_adjust=False)
        
        if data_5m.empty:
            return None
        
        # Aplanar columnas
        if isinstance(data_5m.columns, pd.MultiIndex):
            data_5m.columns = data_5m.columns.get_level_values(0)

        # 0. Descargar VIX Intradía (Latido del Miedo)
        try:
            v_obj = yf.Ticker("^VIX")
            vix_5m = v_obj.history(period="2d", interval="5m", prepost=True)
            if isinstance(vix_5m.columns, pd.MultiIndex): vix_5m.columns = vix_5m.columns.get_level_values(0)
        except:
            vix_5m = pd.DataFrame()

        # 1. Identificar la sesión más reciente disponible (Hoy martes o el último pre-market activo)
        latest_date = data_5m.index[-1].date()
        today_data = data_5m[data_5m.index.date == latest_date].copy()
        
        if today_data.empty: 
            return None

        # 1.5 Obtener Cierre Anterior para referencia (Gap analysis)
        try:
            prev_sessions = data_5m[data_5m.index.date < latest_date]
            prev_close = prev_sessions['Close'].iloc[-1] if not prev_sessions.empty else today_open
        except:
            prev_close = today_open

        # 2. Calcular VWAP (Manejar volumen cero del pre-market)
        v_p = today_data['Close'] * today_data['Volume']
        cum_vol = today_data['Volume'].cumsum()
        # Si no hay volumen (pre-market inicial), el VWAP es igual al Precio
        today_data['VWAP'] = (v_p.cumsum() / cum_vol).fillna(today_data['Close'])

        # 3. EMA 20
        today_data['EMA20'] = today_data['Close'].ewm(span=20, adjust=False).mean()
        
        last_price = today_data['Close'].iloc[-1]
        last_vwap = today_data['VWAP'].iloc[-1]
        last_ema = today_data['EMA20'].iloc[-1]
        today_open = today_data['Open'].iloc[0]
        
        # --- VALIDADOR DE TENDENCIA ---
        score = 0
        if last_price > today_open: score += 1
        if last_price > last_vwap: score += 1
        if last_price > last_ema: score += 1
        
        bear_score = 0
        if last_price < today_open: bear_score += 1
        if last_price < last_vwap: bear_score += 1
        if last_price < last_ema: bear_score += 1

        # Configuración visual
        if score == 3: status, color, icon = ("CONFIRMED BULLISH", "#28a745", "🚀")
        elif bear_score == 3: status, color, icon = ("CONFIRMED BEARISH", "#dc3545", "📉")
        elif score >= 2: status, color, icon = ("MODERATE BULLISH", "#4ade80", "�")
        elif bear_score >= 2: status, color, icon = ("MODERATE BEARISH", "#f87171", "📉")
        else: status, color, icon = ("SIDEWAYS (Rango)", "#94a3b8", "⏳")
            
        return {
            'price': last_price, 'vwap': last_vwap, 'ema': last_ema, 'open': today_open, 'prev_close': prev_close,
            'status': status, 'color': color, 'icon': icon, 'score': score, 'bear_score': bear_score,
            'force': (max(score, bear_score) / 3) * 100,
            'data': today_data, 'vix_data': vix_5m if not vix_5m.empty else None
        }
    except Exception as e:
        print(f"DEBUG Error Sniper: {e}")
        return None

def get_options_sentiment(ticker):
    """Descarga opciones, calcula P/C Ratio y detecta Muros de Gamma (GEX)."""
    try:
        stock = yf.Ticker(ticker)
        # Buscar expiración más cercana (esta semana o siguiente)
        expirations = stock.options
        if not expirations: return None
        
        # Usamos la expiración más cercana (Weekly) para tactica a corto plazo
        near_term = expirations[0]
        
        opt = stock.option_chain(near_term)
        calls = opt.calls
        puts = opt.puts
        
        # --- 1. Put/Call Ratio (Volumen) ---
        vol_calls = calls['volume'].sum()
        vol_puts = puts['volume'].sum()
        
        if vol_calls == 0: return None
        pc_ratio = vol_puts / vol_calls
        
        # --- 2. GEX / Muros (Open Interest) ---
        # Call Wall: Strike con mayor interés abierto en Calls (Resistencia)
        max_call_oi = calls.loc[calls['openInterest'].idxmax()] if not calls.empty else None
        call_wall = max_call_oi['strike'] if max_call_oi is not None else 0
        
        # Put Wall: Strike con mayor interés abierto en Puts (Soporte)
        max_put_oi = puts.loc[puts['openInterest'].idxmax()] if not puts.empty else None
        put_wall = max_put_oi['strike'] if max_put_oi is not None else 0
        
        # Sentimiento basado en Ratio
        if pc_ratio > 1.2:
            sent = "BEARISH (Miedo)"
            color = "#dc3545"
        elif pc_ratio < 0.7:
            sent = "BULLISH (Euforia)"
            color = "#28a745"
        else:
            sent = "NEUTRAL"
            color = "#ffc107"
            
        return {
            'ratio': pc_ratio,
            'sent': sent,
            'color': color,
            'vol_calls': vol_calls,
            'vol_puts': vol_puts,
            'exp': near_term,
            'call_wall': call_wall,
            'put_wall': put_wall
        }
    except Exception as e:
        print(f"Error Opciones: {e}")
        return None
    except Exception as e:
        print(f"Error en Intraday Monitor: {e}")
        return None

def get_llm_analysis(api_key, context_data):
    """Consulta a Groq (Llama 3) para un análisis táctico."""
    if not api_key: return "⚠️ Por favor ingresa tu API Key de Groq en el sidebar."
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    prompt = f"""
    Actúa como un Trader Institucional de Elite. Analiza estos datos del S&P 500 y dame una ESTRATEGIA EJECUTABLE (Breve y Directa).
    
    DATOS DEL MERCADO:
    - Predicción IA: {context_data.get('prediction', 'N/A')} (Confianza: {context_data.get('confidence', '0%')})
    - Riesgo Sistémico (Crash): {context_data.get('risk', 'N/A')}
    - Tendencia Intradía (Sniper): {context_data.get('sniper_status', 'N/A')} (Fuerza: {context_data.get('sniper_force', '0')}%)
    - Sentimiento Opciones: {context_data.get('options_sent', 'N/A')} (P/C Ratio: {context_data.get('pc_ratio', 'N/A')})
    - Call Wall: {context_data.get('call_wall', 'N/A')} | Put Wall: {context_data.get('put_wall', 'N/A')}
    - Contexto: {context_data.get('context_note', '')}
    
    TU MISIÓN:
    1. SINTETIZA: ¿Cuál es la "narrativa" real del mercado hoy? (Bull Trap, Subida Sana, Pánico, etc.)
    2. PLAN DE ATAQUE: ¿Qué debo hacer? (Long, Short, Esperar). Define zonas de entrada/salida si puedes inferirlas.
    3. ADVERTENCIA: ¿Cuál es el mayor peligro ahora mismo?
    
    Responde en formato Markdown, con estilo militar/profesional. Máximo 150 palabras.
    """
    
    payload = {
        "model": "llama-3.3-70b-versatile",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.5
    }
    
    try:
        response = requests.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=payload)
        if response.status_code == 200:
            return response.json()['choices'][0]['message']['content']
        else:
            return f"Error API: {response.text}"
    except Exception as e:
        return f"Error de conexión: {str(e)}"

def get_market_news(ticker):
    """Obtiene los últimos titulares con lógica de respaldo para índices."""
    try:
        # Tickers de respaldo para noticias generales si el principal falla o es un índice
        tickers_to_try = [ticker]
        if ticker.startswith('^') or ticker == 'SPY':
            tickers_to_try.extend(['SPY', 'QQQ', 'DIA'])
        else:
            tickers_to_try.append('SPY') # Siempre intentar SPY como backup

        headlines = []
        seen_titles = set()

        for t in tickers_to_try:
            if len(headlines) >= 8: break
            try:
                stock = yf.Ticker(t)
                news = stock.news
                if news:
                    for n in news:
                        title = n.get('title', '')
                        if title and title not in seen_titles:
                            headlines.append(f"- {title} ({n.get('publisher', 'Yahoo Finance')})")
                            seen_titles.add(title)
                        if len(headlines) >= 8: break
            except:
                continue
                
        return "\n".join(headlines) if headlines else "No se detectaron noticias urgentes en los canales de Yahoo Finance ahora mismo."
    except Exception as e:
        return f"Nota: Servicio de noticias temporalmente limitado. Enfoque en análisis técnico. ({str(e)})"

@st.cache_data(ttl=1800)
def get_economic_calendar():
    """Obtiene el calendario híbrido: Blueprint para la semana + Scraper para datos reales."""
    try:
        # 1. Cargar Blueprint (Estructura de la semana)
        blueprint_path = "macro_blueprint.json"
        if os.path.exists(blueprint_path):
            with open(blueprint_path, 'r') as f:
                bp_data = json.load(f)
            df = pd.DataFrame(bp_data['events'])
        else:
            # Fallback si no hay blueprint: usar scraper puro
            df = pd.DataFrame(columns=['Fecha', 'Hora', 'Evento', 'Actual', 'Previsto', 'Anterior'])

        # 2. Scrapear Yahoo para "Hoy" para rellenar los datos de "Actual"
        try:
            today_str = get_ny_time().strftime("%Y-%m-%d")
            url = f"https://finance.yahoo.com/calendar/economic?day={today_str}&region=US"
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
            
            response = requests.get(url, headers=headers, timeout=10)
            if response.status_code == 200:
                tables = pd.read_html(io.StringIO(response.text))
                if tables:
                    scrape_df = tables[0]
                    scrape_df.columns = [str(c).replace('\xa0', ' ') for c in scrape_df.columns]
                    
                    # Actualizar valores en nuestro DF principal si coinciden los nombres de eventos
                    for _, row in scrape_df.iterrows():
                        event_name = str(row.get('Event', '')).split('*')[0].strip()
                        actual_val = row.get('Actual', '-')
                        # Buscar en nuestro DF (Búsqueda difusa o parcial)
                        mask = df['Evento'].str.contains(event_name, case=False, na=False)
                        if mask.any():
                            df.loc[mask, 'Actual'] = str(actual_val)
        except Exception as e:
            print(f"Aviso Scraper: No se pudo actualizar datos reales ({e})")

        # Limpiar y ordenar
        df = df.replace('nan', '-').fillna('-')
        cols_to_show = ['Fecha', 'Hora', 'Evento', 'Actual', 'Previsto', 'Anterior']
        existing_cols = [c for c in cols_to_show if c in df.columns]
        return df[existing_cols]

    except Exception as e:
        return pd.DataFrame({"Error": [f"Error en calendario híbrido: {str(e)}"]})

def get_pre_market_briefing(api_key, context_data, news_text, calendar_text=""):
    """Genera un informe pre-mercado combinando técnico + noticias + macro."""
    if not api_key: return "⚠️ Error: Falta API Key de Groq."
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    prompt = f"""
    Eres el ESTRATEGA JEFE de un Fondo de Cobertura. Redacta el BRIEFING PRE-MERCADO para tus traders.
    
    1. CONTEXTO TÉCNICO:
    - Sentimiento IA: {context_data.get('prediction', 'N/A')}
    - Radar Opciones: {context_data.get('options_sent', 'N/A')} (P/C: {context_data.get('pc_ratio', 'N/A')})
    - Muros Clave: Call {context_data.get('call_wall', 'N/A')} / Put {context_data.get('put_wall', 'N/A')}
    - VIX/Riesgo: {context_data.get('risk', 'N/A')}
    
    2. ÚLTIMAS NOTICIAS:
    {news_text}
    
    3. CALENDARIO ECONÓMICO (Macro):
    {calendar_text}
    
    TU INFORME (Estilo Bloomberg, Max 200 palabras):
    - TITULAR DE IMPACTO: Resumen de una línea.
    - NARRATIVA: ¿Qué está moviendo el mercado hoy? (Cruza noticias, calendario y técnico).
    - ZONAS DE VIGILANCIA: Niveles de precio clave para hoy.
    - SENTENCIA: ¿Bullish, Bearish o Neutral?
    """
    
    payload = {
        "model": "llama-3.3-70b-versatile",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.6
    }
    
    try:
        response = requests.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=payload)
        if response.status_code == 200:
            return response.json()['choices'][0]['message']['content']
        else:
            return f"Error API: {response.text}"
    except Exception as e:
        return f"Error de conexión: {str(e)}"

# Alpha Vantage functions removed - using Yahoo Finance only

# --- SISTEMA DE BASE DE DATOS SQLITE ---
class MarketDB:
    def __init__(self, db_file="market_data.db"):
        self.db_file = db_file
        self.init_db()

    def init_db(self):
        try:
            conn = sqlite3.connect(self.db_file)
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS prices (
                    ticker TEXT,
                    date TEXT,
                    open REAL,
                    high REAL,
                    low REAL,
                    close REAL,
                    adj_close REAL,
                    volume INTEGER,
                    PRIMARY KEY (ticker, date)
                )
            ''')
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticker TEXT,
                    prediction_date TEXT,
                    execution_date TEXT,
                    prediction_value INTEGER,
                    prob_up REAL,
                    prob_down REAL,
                    is_backtest INTEGER
                )
            ''')
            conn.commit()
            conn.close()
        except Error as e:
            print(f"DB Error: {e}")

    def get_last_date(self, ticker):
        conn = sqlite3.connect(self.db_file)
        df = pd.read_sql_query(f"SELECT MAX(date) FROM prices WHERE ticker='{ticker}'", conn)
        conn.close()
        return df.iloc[0, 0]

    def save_data(self, df, ticker):
        if df.empty: return
        conn = sqlite3.connect(self.db_file)
        # Preparar para insertar: asegurar que el indice es la fecha y no tiene zona horaria
        temp_df = df.copy()
        if not isinstance(temp_df.index, pd.DatetimeIndex):
            temp_df.index = pd.to_datetime(temp_df.index)
        
        temp_df.index = temp_df.index.tz_localize(None)
        temp_df = temp_df.reset_index()
        temp_df['ticker'] = ticker
        temp_df['date'] = temp_df['Date'].dt.strftime('%Y-%m-%d')
        
        # Mapeo de columnas para que coincidan con la DB
        col_map = {
            'Open': 'open', 'High': 'high', 'Low': 'low', 
            'Close': 'close', 'Adj Close': 'adj_close', 'Volume': 'volume'
        }
        temp_df = temp_df.rename(columns=col_map)
        
        # Guardar (usando INSERT OR REPLACE para evitar duplicados en la PRIMARY KEY)
        for _, row in temp_df.iterrows():
            conn.execute('''
                INSERT OR REPLACE INTO prices (ticker, date, open, high, low, close, adj_close, volume)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (row['ticker'], row['date'], row['open'], row['high'], row['low'], row['close'], row.get('adj_close', row['close']), row['volume']))
        
        conn.commit()
        conn.close()

    def load_range(self, ticker, start_date, end_date):
        conn = sqlite3.connect(self.db_file)
        query = f"""
            SELECT * FROM prices 
            WHERE ticker = '{ticker}' 
            AND date BETWEEN '{start_date}' AND '{end_date}'
            ORDER BY date ASC
        """
        df = pd.read_sql_query(query, conn)
        conn.close()
        if not df.empty:
            df['date'] = pd.to_datetime(df['date'])
            df = df.set_index('date')
            # Volver a los nombres originales
            col_map_inv = {
                'open': 'Open', 'high': 'High', 'low': 'Low', 
                'close': 'Close', 'adj_close': 'Adj Close', 'volume': 'Volume'
            }
            df = df.rename(columns=col_map_inv)
            del df['ticker']
        return df

    def save_prediction(self, ticker, pred_date, pred_val, p_up, p_down, is_backtest=0):
        try:
            conn = sqlite3.connect(self.db_file)
            now = dt.now().strftime('%Y-%m-%d %H:%M:%S')
            p_date_str = pred_date.strftime('%Y-%m-%d') if hasattr(pred_date, 'strftime') else str(pred_date)
            conn.execute('''
                INSERT INTO predictions (ticker, prediction_date, execution_date, prediction_value, prob_up, prob_down, is_backtest)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (ticker, p_date_str, now, int(pred_val), float(p_up), float(p_down), int(is_backtest)))
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"Error saving prediction: {e}")

    def get_predictions(self, ticker, limit=10):
        conn = sqlite3.connect(self.db_file)
        query = f"SELECT * FROM predictions WHERE ticker='{ticker}' ORDER BY id DESC LIMIT {limit}"
        df = pd.read_sql_query(query, conn)
        conn.close()
        return df

market_db = MarketDB()

# Cargar datos con múltiples fuentes
@st.cache_data(show_spinner=False)
def load_data(ticker, start_date, end_date):
    """
    Carga datos usando Yahoo Finance únicamente.
    Versión Caché: 7.0 (Yahoo Only)
    """
    return load_data_with_yahoo(ticker, start_date, end_date)

def load_data_with_alpha_vantage(ticker, start_date, end_date, api_key):
    """
    Carga datos usando Alpha Vantage API y sincroniza con la base de datos local.
    """
    s_str = start_date.strftime('%Y-%m-%d') if hasattr(start_date, 'strftime') else str(start_date)
    e_str = end_date.strftime('%Y-%m-%d') if hasattr(end_date, 'strftime') else str(end_date)

    # Alpha Vantage sync logic removed - Yahoo Finance only
    
    if datos.empty: return None

    try:
        # Asegurar tipos numéricos y cálculo de indicadores
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in required_cols:
            datos[col] = pd.to_numeric(datos[col], errors='coerce')
        
        # Cálculo de Indicadores Relativos (Estacionarios)
        datos['Returns'] = datos['Close'].pct_change()
        if 'Returns' not in datos.columns:
            raise KeyError("Returns")
            
        datos['MA20'] = datos['Close'].rolling(window=20).mean()
        datos['MA50'] = datos['Close'].rolling(window=50).mean()
        datos['MA200'] = datos['Close'].rolling(window=200).mean()
        
        datos['Dist_MA20'] = (datos['Close'] / datos['MA20']) - 1
        datos['Dist_MA50'] = (datos['Close'] / datos['MA50']) - 1
        datos['Dist_MA200'] = (datos['Close'] / datos['MA200']) - 1
        
        datos['RSI'] = calculate_rsi(datos['Close'])
        datos['Volatility'] = datos['Returns'].rolling(window=20).std()
        datos['Return_Lag1'] = datos['Returns'].shift(1)
        
        # MACD Relativo
        exp1 = datos['Close'].ewm(span=12, adjust=False).mean()
        exp2 = datos['Close'].ewm(span=26, adjust=False).mean()
        datos['MACD_Rel'] = (exp1 - exp2) / datos['Close']
        
        datos['MA20_BB'] = datos['Close'].rolling(window=20).mean()
        datos['BB_Std'] = datos['Close'].rolling(window=20).std()
        datos['BB_Pos'] = (datos['Close'] - (datos['MA20_BB'] - datos['BB_Std'] * 2)) / (datos['BB_Std'] * 4).replace(0, np.nan)

        # NUEVAS VARIABLES: Volumen Relativo y Momentum 5d
        datos['Volume_MA20'] = datos['Volume'].rolling(window=20).mean()
        datos['Volume_Ratio'] = datos['Volume'] / datos['Volume_MA20']
        datos['Momentum_5d'] = datos['Close'].pct_change(5)

        # Mejoras de Inteligencia: ADX y ATR
        datos['ADX'] = calculate_adx(datos)
        datos['ATR'] = calculate_atr(datos)
        datos['ATR_Rel'] = datos['ATR'] / datos['Close']

        # ANÁLISIS INTERMARKET (Consistente con Yahoo)
        try:
            st.info("🌐 Analizando mercados globales (Oro + Petróleo + VIX)...")
            s_str_m = datos.index[0].strftime('%Y-%m-%d')
            e_str_m = datos.index[-1].strftime('%Y-%m-%d')
            
            vix_data = yf.download("^VIX", start=s_str_m, end=e_str_m, progress=False, auto_adjust=False)
            tnx_data = yf.download("^TNX", start=s_str_m, end=e_str_m, progress=False, auto_adjust=False)
            gold_data = yf.download("GC=F", start=s_str_m, end=e_str_m, progress=False, auto_adjust=False)
            oil_data = yf.download("CL=F", start=s_str_m, end=e_str_m, progress=False, auto_adjust=False)
            nikkei_data = yf.download("^N225", start=s_str_m, end=e_str_m, progress=False, auto_adjust=False)
            dax_data = yf.download("^GDAXI", start=s_str_m, end=e_str_m, progress=False, auto_adjust=False)
            futures_data = yf.download("ES=F", start=s_str_m, end=e_str_m, progress=False, auto_adjust=False)
            dxy_data = yf.download("DX-Y.NYB", start=s_str_m, end=e_str_m, progress=False, auto_adjust=False)
            nya_data = yf.download("^NYA", start=s_str_m, end=e_str_m, progress=False, auto_adjust=False)
            
            for df_market in [vix_data, tnx_data, gold_data, oil_data, nikkei_data, dax_data, futures_data, dxy_data, nya_data]:
                if isinstance(df_market.columns, pd.MultiIndex): df_market.columns = df_market.columns.get_level_values(0)
            
            datos['VIX_Change'] = vix_data['Close'].reindex(datos.index, method='ffill').pct_change().fillna(0)
            datos['Yield_Change'] = tnx_data['Close'].reindex(datos.index, method='ffill').pct_change().fillna(0)
            datos['Gold_Returns'] = gold_data['Close'].reindex(datos.index, method='ffill').pct_change().fillna(0)
            datos['Oil_Returns'] = oil_data['Close'].reindex(datos.index, method='ffill').pct_change().fillna(0)
            datos['Nikkei_Return'] = nikkei_data['Close'].reindex(datos.index, method='ffill').pct_change().fillna(0)
            datos['DAX_Return'] = dax_data['Close'].reindex(datos.index, method='ffill').pct_change().fillna(0)
            datos['Futures_Return'] = futures_data['Close'].reindex(datos.index, method='ffill').pct_change().fillna(0)
            datos['DXY_Return'] = dxy_data['Close'].reindex(datos.index, method='ffill').pct_change().fillna(0)
            datos['NYA_Return'] = nya_data['Close'].reindex(datos.index, method='ffill').pct_change().fillna(0)
        except Exception as e:
            print(f"DEBUG: Error cargando variables Macro: {e}")
            datos['VIX_Change'] = 0
            datos['Yield_Change'] = 0
            datos['Gold_Returns'] = 0
            datos['Oil_Returns'] = 0
            datos['Nikkei_Return'] = 0
            datos['DAX_Return'] = 0
            datos['Futures_Return'] = 0
            datos['DXY_Return'] = 0
            datos['NYA_Return'] = 0

        datos['Target'] = (datos['Close'].shift(-1) > datos['Close']).astype(int)

        # Limpiar valores infinitos y nulos
        datos = datos.replace([np.inf, -np.inf], np.nan)
        datos_final = datos.dropna().copy()
        
        return datos_final
        
    except Exception as e:
        st.error(f"Error al procesar datos de Alpha Vantage: {str(e)}")
        return None
        

def load_data_with_yahoo(ticker, start_date, end_date, max_retries=3):
    """
    Carga datos usando la DB como caché local y Yahoo para sincronizar.
    """
    s_str = start_date.strftime('%Y-%m-%d') if hasattr(start_date, 'strftime') else str(start_date)
    e_str = end_date.strftime('%Y-%m-%d') if hasattr(end_date, 'strftime') else str(end_date)
    
    # 1. Intentar cargar lo que ya tenemos en DB
    db_data = market_db.load_range(ticker, s_str, e_str)
    
    # 2. Verificar si hay "huecos" (especialmente el final para tener datos de hoy)
    last_in_db = market_db.get_last_date(ticker)
    today_str = dt.now().strftime('%Y-%m-%d')
    
    needs_update = False
    fetch_start = s_str
    
    if last_in_db:
        if last_in_db < today_str:
            needs_update = True
            fetch_start = last_in_db # Empezamos desde el último que tenemos
    else:
        needs_update = True

    if needs_update:
        for attempt in range(max_retries):
            try:
                print(f"DEBUG: Sincronizando {ticker} desde {fetch_start}...")
                new_data = yf.download(ticker, start=fetch_start, progress=False, auto_adjust=False)
                
                if not new_data.empty:
                    # Aplanar si es MultiIndex
                    if isinstance(new_data.columns, pd.MultiIndex):
                        new_data.columns = new_data.columns.get_level_values(0)
                    
                    market_db.save_data(new_data, ticker)
                    st.sidebar.success(f"✅ DB actualizada con últimos datos.")
                    break
            except Exception as e:
                print(f"Error sync: {e}")

    # 3. Cargar el rango final completo desde la DB
    datos = market_db.load_range(ticker, s_str, e_str)
    
    if datos.empty: 
        st.error(f"No hay datos disponibles para {ticker} en el rango seleccionado.")
        return None

    try:
        # Asegurar tipos numéricos
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in required_cols:
            datos[col] = pd.to_numeric(datos[col], errors='coerce')
        
        # Cálculo de Indicadores Relativos (Estacionarios)
        datos['Returns'] = datos['Close'].pct_change()
        if 'Returns' not in datos.columns:
            print("DEBUG: ERROR CRÍTICO - 'Returns' no se pudo crear")
            raise KeyError("Returns")
            
        datos['MA20'] = datos['Close'].rolling(window=20).mean()
        datos['MA50'] = datos['Close'].rolling(window=50).mean()
        datos['MA200'] = datos['Close'].rolling(window=200).mean()
        
        datos['Dist_MA20'] = (datos['Close'] / datos['MA20']) - 1
        datos['Dist_MA50'] = (datos['Close'] / datos['MA50']) - 1
        datos['Dist_MA200'] = (datos['Close'] / datos['MA200']) - 1
        
        datos['RSI'] = calculate_rsi(datos['Close'])
        datos['Volatility'] = datos['Returns'].rolling(window=20).std()
        datos['Return_Lag1'] = datos['Returns'].shift(1)
        
        # MACD Relativo
        exp1 = datos['Close'].ewm(span=12, adjust=False).mean()
        exp2 = datos['Close'].ewm(span=26, adjust=False).mean()
        datos['MACD_Rel'] = (exp1 - exp2) / datos['Close']
        
        # Bandas de Bollinger (Posición relativa)
        datos['MA20_BB'] = datos['Close'].rolling(window=20).mean()
        datos['BB_Std'] = datos['Close'].rolling(window=20).std()
        datos['BB_Pos'] = (datos['Close'] - (datos['MA20_BB'] - datos['BB_Std'] * 2)) / (datos['BB_Std'] * 4).replace(0, np.nan)

        # NUEVAS VARIABLES: Volumen Relativo y Momentum 5d
        datos['Volume_MA20'] = datos['Volume'].rolling(window=20).mean()
        datos['Volume_Ratio'] = datos['Volume'] / datos['Volume_MA20']
        datos['Momentum_5d'] = datos['Close'].pct_change(5)

        # Mejoras de Inteligencia: ADX y ATR
        datos['ADX'] = calculate_adx(datos)
        datos['ATR'] = calculate_atr(datos)
        datos['ATR_Rel'] = datos['ATR'] / datos['Close']

        # ANÁLISIS INTERMARKET (Oro + Petróleo + Macro)
        try:
            st.info("🌐 Analizando mercados globales (Oro + Petróleo + VIX)...")
            vix_data = yf.download("^VIX", start=s_str, end=e_str, progress=False, auto_adjust=False)
            tnx_data = yf.download("^TNX", start=s_str, end=e_str, progress=False, auto_adjust=False)
            gold_data = yf.download("GC=F", start=s_str, end=e_str, progress=False, auto_adjust=False)
            oil_data = yf.download("CL=F", start=s_str, end=e_str, progress=False, auto_adjust=False)
            nikkei_data = yf.download("^N225", start=s_str, end=e_str, progress=False, auto_adjust=False)
            dax_data = yf.download("^GDAXI", start=s_str, end=e_str, progress=False, auto_adjust=False)
            futures_data = yf.download("ES=F", start=s_str, end=e_str, progress=False, auto_adjust=False)
            dxy_data = yf.download("DX-Y.NYB", start=s_str, end=e_str, progress=False, auto_adjust=False)
            nya_data = yf.download("^NYA", start=s_str, end=e_str, progress=False, auto_adjust=False)
            
            for df_market in [vix_data, tnx_data, gold_data, oil_data, nikkei_data, dax_data, futures_data, dxy_data, nya_data]:
                if isinstance(df_market.columns, pd.MultiIndex): df_market.columns = df_market.columns.get_level_values(0)
            
            datos['VIX_Change'] = vix_data['Close'].reindex(datos.index, method='ffill').pct_change().fillna(0)
            datos['Yield_Change'] = tnx_data['Close'].reindex(datos.index, method='ffill').pct_change().fillna(0)
            datos['Gold_Returns'] = gold_data['Close'].reindex(datos.index, method='ffill').pct_change().fillna(0)
            datos['Oil_Returns'] = oil_data['Close'].reindex(datos.index, method='ffill').pct_change().fillna(0)
            datos['Nikkei_Return'] = nikkei_data['Close'].reindex(datos.index, method='ffill').pct_change().fillna(0)
            datos['DAX_Return'] = dax_data['Close'].reindex(datos.index, method='ffill').pct_change().fillna(0)
            datos['Futures_Return'] = futures_data['Close'].reindex(datos.index, method='ffill').pct_change().fillna(0)
            datos['DXY_Return'] = dxy_data['Close'].reindex(datos.index, method='ffill').pct_change().fillna(0)
            datos['NYA_Return'] = nya_data['Close'].reindex(datos.index, method='ffill').pct_change().fillna(0)
        except Exception as e:
            print(f"DEBUG: Error cargando variables Macro: {e}")
            datos['VIX_Change'] = 0
            datos['Yield_Change'] = 0
            datos['Gold_Returns'] = 0
            datos['Oil_Returns'] = 0
            datos['Nikkei_Return'] = 0
            datos['DAX_Return'] = 0
            datos['Futures_Return'] = 0
            datos['DXY_Return'] = 0
            datos['NYA_Return'] = 0

        datos['Target'] = (datos['Close'].shift(-1) > datos['Close']).astype(int)

        if datos.index.tz is not None:
            datos.index = datos.index.tz_convert('America/New_York')

        # Limpiar valores infinitos y nulos
        datos = datos.replace([np.inf, -np.inf], np.nan)
        datos_final = datos.dropna().copy()
        
        print(f"DEBUG: Datos procesados correctamente. Filas final: {len(datos_final)}")
        return datos_final

    except Exception as e:
        st.error(f"Error procesando indicadores técnicos: {str(e)}")
        return None

# Entrenar modelo avanzado con XGBoost - Versión de Alto Rendimiento 3.0
def train_model(data):
    features = GLOBAL_FEATURES
    
    # Limpieza final
    df = data.copy()
    df = df.replace([np.inf, -np.inf], np.nan)
    available_features = [f for f in features if f in df.columns]
    df = df.dropna(subset=available_features + ['Target'])
    
    if len(df) < 100:
        raise ValueError("No hay suficientes datos limpios.")

    X = df[available_features]
    y = df['Target']

    # 1. Ponderación Temporal (0.5 a 1.0) para dar más peso a lo reciente (más agresivo)
    sample_weights = np.linspace(0.5, 1.0, len(y))

    # Split cronológico
    split_idx = int(len(X) * 0.85)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    w_train = sample_weights[:split_idx]

    # Búsqueda Maestra de Parámetros (GridSearch) y Ensamble de Comité
    pos_weight_calc = (len(y_train) - sum(y_train)) / sum(y_train) if sum(y_train) > 0 else 1.0
    
    # 1. XGBoost
    xgb = XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.03, 
        scale_pos_weight=pos_weight_calc, random_state=42, eval_metric='logloss'
    )
    
    # 2. Random Forest
    rf = RandomForestClassifier(n_estimators=300, max_depth=6, random_state=42)
    
    # 3. Gradient Boosting (Sustituto de LightGBM por robustez)
    gb = GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, max_depth=4, random_state=42)

    # El Comité de Votación (Soft Voting usa las probabilidades para promediar)
    ensemble = VotingClassifier(
        estimators=[('xgb', xgb), ('rf', rf), ('gb', gb)],
        voting='soft'
    )
    
    ensemble.fit(X_train, y_train, sample_weight=w_train)

    return ensemble, X_test, y_test

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
    features = GLOBAL_FEATURES
    stat_text, stat_code = check_market_status()

    # 1. Obtener indicadores del último cierre
    last_row = data.iloc[-1].copy()
    last_trading_day = data.index[-1]
    
    # 2. Si es Pre-Mercado o Abierto, intentar inyectar datos REAL-TIME
    if stat_code in ['pre', 'open']:
        try:
            # Sincronizar Futuros, Nikkei, DAX, DXY y NYA del momento exacto
            tickers_intl = {
                "Nikkei_Return": "^N225", 
                "DAX_Return": "^GDAXI", 
                "Futures_Return": "ES=F",
                "DXY_Return": "DX-Y.NYB",
                "NYA_Return": "^NYA"
            }
            for feat, symb in tickers_intl.items():
                intl_df = yf.download(symb, period="5d", progress=False, auto_adjust=False)
                if len(intl_df) >= 2:
                    current_price = intl_df['Close'].iloc[-1]
                    prev_close = intl_df['Close'].iloc[-2]
                    last_row[feat] = (current_price / prev_close) - 1
        except Exception as e:
            print(f"DEBUG: Fallo en Live Sync: {e}")

    # --- BLINDAJE ANTI-MISMATCH AVANZADO ---
    # Detectamos qué columnas espera el modelo realmente
    model_features = features
    try:
        if hasattr(model, 'feature_names_in_'):
            model_features = list(model.feature_names_in_)
        elif hasattr(model, 'estimators_') and hasattr(model.estimators_[0], 'feature_names_in_'):
            model_features = list(model.estimators_[0].feature_names_in_)
    except:
        pass
    
    # Preparamos el DataFrame con las columnas exactas que el modelo conoce
    input_df = pd.DataFrame([last_row.reindex(model_features).fillna(0.0)], columns=model_features)
    
    try:
        prediction_proba = model.predict_proba(input_df)[0]
    except Exception as e:
        print(f"Error crítico en predict_proba: {e}")
        return -1, [0.5, 0.5], last_trading_day, {}, "⚠️ Error de sincronización de modelo. Por favor, re-entrena desde el sidebar."
    
    # FILTRO DE SEGURIDAD DINÁMICO
    prob_up = prediction_proba[1]
    prob_down = prediction_proba[0]
    adx_val = last_row.get('ADX', 0)
    
    # Si no hay tendencia (ADX bajo), exigimos más confianza para evitar ruido
    threshold = 0.72 if adx_val < 20 else 0.65
    
    if prob_up > threshold:
        final_prediction = 1
    elif prob_down > threshold:
        final_prediction = 0
    else:
        final_prediction = -1 # Neutral / Esperar
    
    # Determinar fecha objetivo
    if stat_code in ['pre', 'open']:
        target_date = get_ny_time()
    else:
        target_date = last_trading_day + timedelta(days=1)
        while target_date.weekday() >= 5:
            target_date += timedelta(days=1)
    
    # Generar Nota de Sesgo Matinal
    bias_note = ""
    f_ret = last_row.get('Futures_Return', 0)
    d_ret = last_row.get('DXY_Return', 0)
    v_chg = last_row.get('VIX_Change', 0)
    adx_val = last_row.get('ADX', 0)
    
    if stat_code in ['pre', 'open']:
        direction = "Alcista" if final_prediction == 1 else "Bajista" if final_prediction == 0 else "Neutral"
        intensity = "Fuerte" if max(prob_up, prob_down) > 0.75 else "Moderado"
        
        note = f"**BIAS Matinal:** {direction} {intensity}. "
        
        # Inyectar inteligencia de tendencia
        if adx_val < 20:
            note += "⚠️ Mercado en rango/lateral (ADX bajo). "
        elif adx_val > 35:
            note += "🔥 Tendencia con alta convicción (ADX fuerte). "

        if f_ret > 0.002: note += "Los Futuros muestran optimismo. "
        elif f_ret < -0.002: note += "Los Futuros indican presión vendedora. "
        
        if d_ret > 0.001: note += "El Dólar (DXY) está fuerte (presión bajista). "
        if v_chg > 0.02: note += "VIX al alza: Volatilidad elevada."
        
        # --- GATILLO DE SHORT POR RIESGO DE CRASH ---
        # Calculamos el riesgo en tiempo real
        risk_lvl, _, _, _ = calculate_crash_risk(data)
        if "ALTO" in risk_lvl or "EXTREMO" in risk_lvl:
            if final_prediction == 0: # Si la IA ya dice bajada
                note += "\n\n🔥 **OPORTUNIDAD DE SHORT:** Riesgo sistémico elevado + Sesgo bajista. Considera proteger o buscar entradas cortas."
        
        bias_note = note
    else:
        direction = "Alcista" if final_prediction == 1 else "Bajista" if final_prediction == 0 else "Neutral"
        bias_note = f"**Sesgo de Cierre:** {direction}. Basado en la configuración histórica."

    # Analizar alineación para el panel visual
    market_breadth = {
        'sp500_ret': f_ret if stat_code in ['pre', 'open'] else last_row.get('Returns', 0),
        'nya_ret': last_row.get('NYA_Return', 0)
    }
    
    return final_prediction, prediction_proba, target_date, market_breadth, bias_note

# Predecir desde una fecha seleccionada para hacer backtesting
def predict_from_date(model, data, selected_date):
    features = GLOBAL_FEATURES

    # Convertir a Timestamp y quitar zona horaria para evitar errores de comparación
    selected_ts = pd.Timestamp(selected_date).tz_localize(None)
    
    # Asegurar que el índice de los datos sea naive para la comparación
    data_naive = data.copy()
    if data_naive.index.tz is not None:
        data_naive.index = data_naive.index.tz_localize(None)

    # Filtrar los datos hasta la fecha seleccionada
    data_until_selected = data_naive.loc[:selected_ts]

    # --- BLINDAJE ANTI-MISMATCH ---
    # Detectamos qué columnas espera el modelo realmente
    model_features = features
    if hasattr(model, 'feature_names_in_'):
        model_features = list(model.feature_names_in_)
        
    last_row = data_until_selected.iloc[-1]
    last_data_df = pd.DataFrame([last_row.reindex(model_features).fillna(0.0)], columns=model_features)

    prediction = model.predict(last_data_df)
    prediction_proba = model.predict_proba(last_data_df)

    # Calcular el siguiente día de negociación
    last_trading_day = data_until_selected.index[-1]
    next_day = last_trading_day + timedelta(days=1)
    while next_day.weekday() >= 5:
        next_day += timedelta(days=1)

    return prediction[0], prediction_proba[0], next_day

def plot_feature_importance(model, features):
    import plotly.express as px
    import pandas as pd
    import numpy as np
    
    try:
        # 1. Caso Ensamble (VotingClassifier)
        if hasattr(model, 'estimators_'):
            all_imps = []
            for est in model.estimators_:
                # En XGBoost/RF de scikit-learn, feature_importances_ es estándar
                if hasattr(est, 'feature_importances_'):
                    all_imps.append(est.feature_importances_)
            
            if all_imps:
                # Promediamos las opiniones de todo el comité
                importances = np.mean(all_imps, axis=0)
            else:
                return None
        # 2. Caso Modelo Individual
        elif hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        else:
            return None

        # Asegurar nombres de columnas
        if hasattr(model, 'feature_names_in_'):
            names = list(model.feature_names_in_)
        else:
            names = features[:len(importances)]

        # Crear DataFrame para el gráfico
        df_imp = pd.DataFrame({
            'Variable': names, 
            'Importancia': importances
        }).sort_values(by='Importancia', ascending=True)

        # Crear gráfica con Plotly
        fig = px.bar(
            df_imp, 
            x='Importancia', 
            y='Variable', 
            orientation='h',
            title='🎯 Relevancia de los Indicadores (Comité de Modelos)',
            color='Importancia',
            color_continuous_scale='Viridis',
            labels={'Importancia': 'Peso en el Consenso'}
        )
        
        fig.update_layout(
            height=500,
            margin=dict(l=20, r=20, t=40, b=20),
            showlegend=False
        )
        return fig
        
    except Exception as e:
        print(f"DEBUG: Error generando gráfica: {e}")
        return None

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
    # Detectar qué columnas espera el modelo realmente
    if model is None:
        return pd.DataFrame()
        
    features = GLOBAL_FEATURES
    try:
        if hasattr(model, 'feature_names_in_'):
            features = list(model.feature_names_in_)
        elif hasattr(model, 'estimators_') and hasattr(model.estimators_[0], 'feature_names_in_'):
             features = list(model.estimators_[0].feature_names_in_)
    except:
        pass
    
    # Trabajar sobre una copia limpia
    df = data.copy()
    
    # Asegurar que todas las features requeridas existan en el DF (rellenar con 0 si faltan)
    for f in features:
        if f not in df.columns:
            df[f] = 0.0
            
    # Limpieza final
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=['Target'])
    
    if df.empty:
        return pd.DataFrame()

    # Realizar predicciones con las columnas exactas
    df['Prediction'] = model.predict(df[features])
    df['Real'] = df['Target']
    df['Correcto'] = np.where(df['Prediction'] == df['Real'], 'Correcto', 'Erroneo')
    
    results = df[['Open', 'Close', 'Prediction', 'Real', 'Correcto']]
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
    st.set_page_config(
        page_title="Market Predictor | AI Control Tower",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    st.title('Predicción del Mercado de Valores')

    # Información inicial
    st.info("📊 Esta aplicación utiliza Machine Learning para predecir movimientos del mercado de valores")

    st.sidebar.header('Parámetros')
    
    # Reloj de NY y Estado del Mercado
    status_text, status_code = check_market_status()
    ny_now = get_ny_time()
    st.sidebar.markdown(f"**🕒 NY Time:** {ny_now.strftime('%H:%M:%S')}")
    st.sidebar.markdown(f"**Estado:** {status_text}")
    st.sidebar.markdown("---")

    selected_stock = st.sidebar.selectbox('Selecciona una acción o índice:', list(TOP_20_STOCKS.keys()))

    # Fuente de Datos: Yahoo Finance (Única opción)
    # Alpha Vantage removed.


    # Usar un rango de 5 años por defecto
    end_date = dt.now().date()
    default_start = end_date - timedelta(days=1825)  # 5 años
    start_date = st.sidebar.date_input('Fecha de inicio', value=default_start, max_value=end_date)

    # Validar rango de fecha de inicio (Limitar a 30 años para estabilidad)
    max_days_back = 10950 # ~30 años
    if (end_date - start_date).days > max_days_back:
        st.sidebar.warning(f"⚠️ El rango máximo es de {max_days_back // 365} años para optimizar el rendimiento.")
        start_date = end_date - timedelta(days=max_days_back)

    ticker = TOP_20_STOCKS[selected_stock]

    st.sidebar.info(f"📈 Ticker seleccionado: **{ticker}**")
    st.sidebar.info(f"📅 Rango: {start_date} a {end_date}")
    st.sidebar.info(f"📊 Días: {(end_date - start_date).days}")
    st.sidebar.info(f"🔗 Fuente: **Yahoo Finance**")

    # Resetear métricas si cambia el ticker
    if 'last_ticker' not in st.session_state or st.session_state['last_ticker'] != ticker:
        st.session_state['last_ticker'] = ticker
        if 'metrics' in st.session_state:
            del st.session_state['metrics']
            
    # --- INTEGRACIÓN LLM (Copiloto) ---
    # --- INTEGRACIÓN LLM (Copiloto) ---
    st.sidebar.markdown("---")
    
    # Cargar API Key de Groq silenciosamente
    groq_api_key = os.getenv('GROQ_API_KEY')
    
    if groq_api_key:
        st.sidebar.success("✅ Copiloto IA (Groq): Activo")
    else:
        st.sidebar.warning("⚠️ Copiloto IA: Inactivo")
        st.sidebar.info("Para activar la IA, añade GROQ_API_KEY en el archivo .env")

    # Cargar datos con mejor manejo de errores
    with st.spinner(f'Cargando datos de {selected_stock} desde Yahoo Finance...'):
        data = load_data(ticker, start_date, end_date)

    if data is None or data.empty:
        st.error("❌ No se pudieron cargar los datos.")
        st.warning("💡 **Sugerencias:**")
        st.write("1. Intenta con otra acción del menú (por ejemplo: **Apple (AAPL)** o **Microsoft (MSFT)**)")
        st.write("2. Verifica tu conexión a internet")


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


    # --- CARGA DE INTELIGENCIA ---
    model = load_model(ticker)

    # --- ESTRUCTURA DE PANTALLA: TORRE DE CONTROL ---
    tab_market, tab_stats, tab_brain, tab_calendar, tab_history = st.tabs([
        "📟 Market Desk", "📊 Estadísticas", "🧠 Inteligencia IA", "📅 Agenda Económica", "📜 Historial"
    ])

    with tab_market:
        # Reloj y Estado (Header Prominente)
        status_text, status_code = check_market_status()
        status_color = "#28a745" if status_code == 'open' else "#ffc107" if status_code == 'pre' else "#dc3545"
        
        st.markdown(f"""
        <div style="padding:20px; border-radius:15px; background: rgba(0,0,0,0.3); border-left: 10px solid {status_color}; margin-bottom: 20px;">
            <h1 style="margin:0; font-size: 1.5em;">{ticker} | Torre de Control</h1>
            <p style="margin:0; color:{status_color}; font-weight: bold; font-size: 1.1em;">{status_text}</p>
        </div>
        """, unsafe_allow_html=True)

        # Fila 1: Estado del Mercado
        m_col1, m_col2, m_col3 = st.columns(3)
        
        last_close = data['Close'].iloc[-1]
        previous_close = data['Close'].iloc[-2]
        change_pts = last_close - previous_close
        change_pct = (change_pts / previous_close) * 100
        
        m_col1.metric("Precio Actual", f"${last_close:.2f}", f"{change_pts:+.2f} pts")
        m_col2.metric("Variación %", f"{change_pct:+.2f}%")
        m_col3.metric("Volumen", f"{data['Volume'].iloc[-1]:,.0f}")

        # --- INDICADOR DE RIESGO DE CRASH ---
        risk_lvl, risk_color, risk_icon, risk_reasons = calculate_crash_risk(data)
        st.markdown(f"""
        <div style="padding:15px; border-radius:10px; background: rgba(0,0,0,0.2); border: 2px solid {risk_color}; margin: 20px 0;">
            <div style="display:flex; align-items:center; gap:15px;">
                <span style="font-size:2em;">{risk_icon}</span>
                <div>
                    <h4 style="margin:0; color:{risk_color};">RIESGO SISTÉMICO: {risk_lvl}</h4>
                    <p style="margin:5px 0 0 0; font-size:0.9em; color:#bbb;">Razones: {', '.join(risk_reasons)}</p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Fila 2: Contexto Global (Intermarket)
        st.markdown("##### 🌍 Contexto Global (Pre-Mercado)")
        c_col1, c_col2, c_col3, c_col4 = st.columns(4)
        
        # VIX
        if 'VIX_Change' in data.columns:
            last_vix_ch = data['VIX_Change'].iloc[-1] * 100
            c_col1.metric("Volatilidad (VIX)", f"{last_vix_ch:+.2f}%")
        else:
            c_col1.metric("VIX", "N/D")
    
        # Nikkei
        if 'Nikkei_Return' in data.columns:
            last_nikkei = data['Nikkei_Return'].iloc[-1] * 100
            c_col2.metric("Asia (Nikkei)", f"{last_nikkei:+.2f}%")
        else:
            c_col2.metric("Asia (Nikkei)", "N/A")
    
        # DAX
        if 'DAX_Return' in data.columns:
            last_dax = data['DAX_Return'].iloc[-1] * 100
            c_col3.metric("Europa (DAX)", f"{last_dax:+.2f}%")
        else:
            c_col3.metric("Europa (DAX)", "N/A")
    
        # Futuros
        if 'Futures_Return' in data.columns:
            last_futures = data['Futures_Return'].iloc[-1] * 100
            c_col4.metric("Futuros (ES=F)", f"{last_futures:+.2f}%")
        else:
            c_col4.metric("Futuros", "N/A")

    # --- NUEVO APARTADO: MÉTRICAS DEL ACTIVO ---
    st.markdown("---")
    with tab_stats:
        st.subheader(f"📊 Análisis de Movimientos: {selected_stock}")
        
        # Cálculos de métricas históricas
        col_date1, col_date2 = st.columns(2)
        m_start = col_date1.date_input("Inicio Análisis", value=data.index[-30].date() if len(data) > 30 else data.index[0].date(), min_value=data.index[0].date(), max_value=data.index[-1].date(), key="m_start")
        m_end = col_date2.date_input("Fin Análisis", value=data.index[-1].date(), min_value=data.index[0].date(), max_value=data.index[-1].date(), key="m_end")
        
        if st.button("📊 Generar Reporte de Movimientos", use_container_width=True):
            mask = (data.index.date >= m_start) & (data.index.date <= m_end)
            df_metrics = data.loc[mask].copy()
            
            if not df_metrics.empty:
                # Cálculos (Simplificado para el chunk)
                df_metrics['Diff_Points'] = df_metrics['Close'].diff()
                df_metrics['Intraday_Diff'] = df_metrics['Close'] - df_metrics['Open']
                
                e_col1, e_col2 = st.columns(2)
                e_col1.metric("Máxima Subida", f"+{df_metrics['Returns'].max()*100:.2f}%")
                e_col2.metric("Máxima Caída", f"{df_metrics['Returns'].min()*100:.2f}%")
                
                st.markdown("#### Movimiento Intradía Promedio")
                st.info(f"El activo se mueve en promedio **{df_metrics['Intraday_Diff'].abs().mean():.2f} puntos** entre apertura y cierre.")
                
            st.line_chart(data['Close'])
        
        st.markdown("---")
        st.subheader("🧪 Terminal de Backtesting")
        
        # Inicializar Racha en session_state
        if 'backtest_streak' not in st.session_state:
            st.session_state['backtest_streak'] = 0

        back_col1, back_col2 = st.columns([1, 2])
        with back_col1:
            test_date = st.date_input("Fecha de Simulación", value=data.index[-2].date())
            
            # Mostrar Racha Actual
            streak = st.session_state['backtest_streak']
            if streak > 0:
                st.markdown(f"🔥 **Racha Actual: {streak} Aciertos**")
            
            if st.button("🚀 Ejecutar Simulación"):
                if model:
                    p, prob, next_d = predict_from_date(model, data, pd.Timestamp(test_date))
                    st.write(f"Resultado IA: **{'Subida 🟢' if p==1 else 'Bajada 🔴'}**")
                    
                    # Verificar si ya tenemos el resultado real para esa fecha
                    if next_d in data.index:
                        real_return = data.loc[next_d, 'Returns']
                        real_direction = 1 if real_return > 0 else 0
                        
                        if p == real_direction:
                            st.session_state['backtest_streak'] += 1
                            st.success(f"🎯 **ACERTADO** (El mercado {'subió' if real_direction==1 else 'bajó'} un {real_return*100:.2f}%)")
                            if st.session_state['backtest_streak'] >= 3:
                                st.balloons()
                        else:
                            st.session_state['backtest_streak'] = 0
                            st.error(f"❌ **FALLO** (El mercado {'subió' if real_direction==1 else 'bajó'} un {real_return*100:.2f}%)")
                    else:
                        st.info("⌛ Resultado real pendiente.")
                        
                    market_db.save_prediction(ticker, next_d, p, prob[1], prob[0], 1)
        with back_col2:
            st.plotly_chart(plot_candlestick_chart(data, pd.Timestamp(test_date)), use_container_width=True)

    st.markdown("---")


    with tab_brain:
        st.subheader("🧠 Auditoría de IA (Confidence & Metrics)")
        
        if model:
            if 'metrics' in st.session_state:
                accuracy, precision, recall, f1 = st.session_state['metrics']
                m_col1, m_col2, m_col3, m_col4 = st.columns(4)
                m_col1.metric('Exactitud', f'{accuracy:.2f}')
                m_col2.metric('Precisión', f'{precision:.2f}')
                m_col3.metric('Sensibilidad', f'{recall:.2f}')
                m_col4.metric('F1-score', f'{f1:.2f}')
            
            st.markdown("---")
            st.write("📊 **Importancia de Variables (Comité de Modelos)**")
            importance_fig = plot_feature_importance(model, GLOBAL_FEATURES)
            if importance_fig:
                st.plotly_chart(importance_fig, use_container_width=True)
            
            # Descargar resultados
            try:
                results_df = generate_results(model, data)
                if not results_df.empty:
                    download_csv(results_df, ticker)
            except Exception as e:
                st.error(f"🚨 Error al generar historial: {e}. Se recomienda re-entrenar el modelo.")
        else:
            st.info("No hay un modelo activo. Usa el sidebar para entrenar uno nuevo.")

    with tab_calendar:
        st.subheader("📅 Calendario Económico Semanal (EE.UU.)")
        st.info("💡 Resaltado en verde los eventos de HOY. Los datos pasados ayudan a entender el contexto de la semana.")
        
        cal_df = get_economic_calendar()
        
        if "Error" in cal_df.columns:
            st.error(cal_df["Error"].iloc[0])
        elif "Info" in cal_df.columns:
            st.warning(cal_df["Info"].iloc[0])
        else:
            # Resaltar filas de hoy
            today_str = get_ny_time().strftime("%b %d, %Y")
            
            def highlight_today(row):
                return ['background-color: rgba(40, 167, 69, 0.3)' if row['Fecha'] == today_str else '' for _ in row]

            st.dataframe(cal_df.style.apply(highlight_today, axis=1), use_container_width=True, hide_index=True)
            
            # IA Context: Solo le enviamos hoy y futuro cercano
            summary_cal = []
            for _, row in cal_df.head(15).iterrows():
                summary_cal.append(f"- {row.get('Fecha', '')} {row.get('Hora', '')} | {row.get('Evento', 'N/D')} | Act: {row.get('Actual', '-')} Prev: {row.get('Previsto', '-')}")
            st.session_state['calendar_text'] = "\n".join(summary_cal)

    with tab_market:
        st.sidebar.markdown("---")
        if st.sidebar.button('🚆 Entrenar / Sincronizar Modelo'):
            with st.spinner('Consolidando inteligencia del comité...'):
                model, X_test, y_test = train_model(data)
                save_model(model, ticker)
                st.session_state['metrics'] = evaluate_model(model, X_test, y_test)
                st.rerun()

        # Operación Maestra
        st.markdown("### 🚀 Señal de Trading")
        btn_label = "🎯 OBTENER SESGO DE APERTURA" if status_code == 'pre' else "🔮 CONSULTAR PREDICCIÓN"
        
        if st.button(btn_label, use_container_width=True):
            if model:
                try:
                    res = predict_next_day(model, data)
                    st.session_state['last_pred'] = {
                        'prediction': res[0],
                        'proba': res[1],
                        'date': res[2],
                        'breadth': res[3],
                        'note': res[4]
                    }
                    market_db.save_prediction(ticker, res[2], res[0], res[1][1], res[1][0], 0)
                except Exception as e:
                    st.error(f"🚨 Error de inteligencia: {e}. Por favor, pulsa 'Entrenar' en el sidebar para sincronizar.")

        # Mostrar Predicción Persistente
        if 'last_pred' in st.session_state:
            lp = st.session_state['last_pred']
            st.info(f"📝 {lp['note']}")
            
            p_col1, p_col2 = st.columns([2, 1])
            with p_col1:
                if lp['prediction'] == 1:
                    st.success(f"### Predicción: SUBIDA (🟢 {lp['proba'][1]*100:.1f}%)")
                elif lp['prediction'] == 0:
                    st.error(f"### Predicción: BAJADA (🔴 {lp['proba'][0]*100:.1f}%)")
                else:
                    st.warning("### Predicción: NEUTRAL / ESPERAR")
            with p_col2:
                st.write(f"Vence: **{lp['date'].strftime('%d/%m/%Y')}**")

            # Alineación
            st.markdown("##### 📊 Alineación S&P vs NYSE")
            sp_ret, nya_ret = lp['breadth']['sp500_ret'], lp['breadth']['nya_ret']
            color = "green" if (sp_ret * nya_ret > 0) else "red"
            st.markdown(f"""
            <div style="padding:10px; border-radius:5px; background: rgba(0,0,0,0.2); border-left: 5px solid {color};">
                <b>Estado:</b> {"✅ Sincronizado" if color=="green" else "⚠️ Divergente"}<br>
                S&P: {sp_ret*100:+.2f}% | NYSE: {nya_ret*100:+.2f}%
            </div>
            """, unsafe_allow_html=True)

            # --- OPCIONES SENTIMENT (NUEVO) ---
            opt_data = get_options_sentiment(ticker)
            if opt_data:
                st.markdown("##### 🎰 Radar de Opciones (Smart Money)")
                st.markdown(f"""
                <div style="padding:15px; border-radius:10px; border: 1px solid {opt_data['color']}; background: rgba(0,0,0,0.2);">
                    <div style="display:flex; justify-content:space-between; align-items:center;">
                        <h4 style="margin:0; color:{opt_data['color']};">{opt_data['sent']}</h4>
                        <span style="font-size:0.8em; color:#fff; background:{opt_data['color']}; padding:2px 8px; border-radius:10px;">P/C: {opt_data['ratio']:.2f}</span>
                    </div>
                    <div style="margin-top:10px; padding:10px; background:rgba(255,255,255,0.05); border-radius:5px;">
                        <div style="display:flex; justify-content:space-between; margin-bottom:5px;">
                            <span style="color:#ff6b6b; font-weight:bold;">🧱 Call Wall (Resistencia):</span>
                            <span style="color:white;">${opt_data['call_wall']:,.0f}</span>
                        </div>
                        <div style="display:flex; justify-content:space-between;">
                            <span style="color:#51cf66; font-weight:bold;">🛡️ Put Wall (Soporte):</span>
                            <span style="color:white;">${opt_data['put_wall']:,.0f}</span>
                        </div>
                    </div>
                    <div style="display:flex; justify-content:space-between; font-size:0.75em; margin-top:8px; color:#999;">
                        <span>Vol. Calls: {int(opt_data['vol_calls']):,}</span>
                        <span>Vol. Puts: {int(opt_data['vol_puts']):,}</span>
                        <span>Exp: {opt_data['exp']}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            # --- MONITOR SNIPER LIVE ---
            st.markdown("---")
            head_col1, head_col2 = st.columns([2, 1])
            head_col1.subheader("🎯 Monitor SNIPER")
            live_tracking = head_col2.toggle("📡 Rastreo en Vivo (60s)", value=False)

            snip = get_intraday_momentum(ticker)
            
            # --- COPILOTO ESTRATÉGICO (LLM) ---
            with st.expander("🤖 Copiloto Estratégico (IA)", expanded=True):
                # Botones de Acción
                c_col1, c_col2 = st.columns(2)
                
                with c_col1:
                    btn_tactical = st.button("🧠 INFORME TÁCTICO (Técnico)", use_container_width=True)
                with c_col2:
                    btn_briefing = st.button("🌅 BRIEFING PRE-MERCADO (Noticias)", use_container_width=True)

                if 'last_pred' in st.session_state:
                     # Recopilar contexto común
                    ctx = {
                        'prediction': "ALCISTA" if st.session_state['last_pred']['prediction'] == 1 else "BAJISTA",
                        'confidence': f"{max(st.session_state['last_pred']['proba'])*100:.1f}%",
                        'risk': risk_lvl,
                        'sniper_status': snip['status'] if snip else "Esperando datos...",
                        'sniper_force': int(snip['force']) if snip else 0,
                        'options_sent': opt_data['sent'] if opt_data else "Sin datos",
                        'pc_ratio': f"{opt_data['ratio']:.2f}" if opt_data else "N/A",
                        'call_wall': f"${opt_data['call_wall']:,.0f}" if opt_data else "N/A",
                        'put_wall': f"${opt_data['put_wall']:,.0f}" if opt_data else "N/A",
                        'context_note': st.session_state['last_pred']['note']
                    }
                    
                    if btn_tactical:
                        with st.spinner("Analizando estructura de mercado..."):
                            analysis = get_llm_analysis(groq_api_key, ctx)
                            st.info("### 🛡️ Informe Táctico (Intradía)")
                            st.markdown(analysis)
                            
                    if btn_briefing:
                        with st.spinner("Leyendo noticias y cruzando datos..."):
                            news = get_market_news(ticker)
                            cal_txt = st.session_state.get('calendar_text', "Sin eventos macro reportados.")
                            briefing = get_pre_market_briefing(groq_api_key, ctx, news, cal_txt)
                            st.success("### 🌅 Briefing Pre-Mercado (Macro + Técnico)")
                            st.markdown(briefing)
                            st.markdown("---")
                            st.caption("📰 Titulares Fuente:")
                            st.text(news)
                else:
                    if btn_tactical or btn_briefing:
                        st.warning("Primero debes obtener una predicción del modelo.")

            if snip:
                s_col1, s_col2 = st.columns([1, 2])
                with s_col1:
                    st.markdown(f"""
                    <div style="padding:15px; border-radius:10px; background:{snip['color']}; color:white; text-align:center; border: 2px solid white;">
                        <h1 style="margin:0;">{snip['icon']}</h1>
                        <p style="margin:5px 0; font-weight:bold;">{snip['status']}</p>
                        <p style="margin:0; font-size:0.8em;">
                            <b>Precio (Live):</b> ${snip['price']:.2f}<br>
                            <b>Cierre Anterior:</b> ${snip['prev_close']:.2f}<br>
                            <b>Apertura (4am):</b> ${snip['open']:.2f}<br>
                            <b>VWAP:</b> ${snip['vwap']:.2f}
                        </p>
                        <p style="margin:5px 0 0 0; font-size:0.7em; color:#ddd;">Actualizado (NY): {get_ny_time().strftime('%H:%M:%S')}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown(f"**Fuerza del Mercado: {snip['force']:.0f}%**")
                    st.progress(snip['force']/100)

                    # --- VALIDADOR DE BIAS VS TENDENCIA REAL ---
                    if lp['prediction'] == 1: # IA dice Subida
                        if snip['score'] == 3:
                            st.success("✅ **VALIDADO:** Tendencia e IA sincronizadas.")
                        elif snip['bear_score'] >= 2:
                            st.error("🚨 **INVALIDADO:** La tendencia real es bajista. ¡Cuidado con la compra!")
                        else:
                            st.warning("⏳ **ESPERANDO CONFLUENCIA:** Tendencia mixta.")
                    
                    elif lp['prediction'] == 0: # IA dice Bajada
                        if snip['bear_score'] == 3:
                            st.success("✅ **VALIDADO:** Tendencia e IA sincronizadas.")
                        elif snip['score'] >= 2:
                            st.error("🚨 **INVALIDADO:** El mercado está rebotando. ¡Cuidado con la venta!")
                        else:
                            st.warning("⏳ **ESPERANDO CONFLUENCIA:** Tendencia mixta.")

                with s_col2:
                    fig_snip = go.Figure()
                    fig_snip.add_trace(go.Scatter(x=snip['data'].index, y=snip['data']['Close'], name='Precio', line=dict(color='white', width=2)))
                    fig_snip.add_trace(go.Scatter(x=snip['data'].index, y=snip['data']['VWAP'], name='VWAP', line=dict(color='cyan', dash='dash')))
                    fig_snip.add_trace(go.Scatter(x=snip['data'].index, y=snip['data']['EMA20'], name='EMA 20', line=dict(color='magenta', width=1)))
                    fig_snip.update_layout(height=260, margin=dict(l=0, r=0, t=0, b=0), showlegend=True, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                    st.plotly_chart(fig_snip, use_container_width=True)
                    
                    # Latido del Miedo (VIX Intradía)
                    if snip['vix_data'] is not None:
                        vix_cur = snip['vix_data']['Close'].iloc[-1]
                        vix_open = snip['vix_data']['Open'].iloc[0]
                        vix_pct = ((vix_cur - vix_open) / vix_open) * 100
                        
                        vix_col = "red" if vix_pct > 0 else "green"
                        st.caption(f"📉 **Latido del Miedo (VIX 5m):** {vix_cur:.2f} ({vix_pct:+.2f}%)")
                        
                        fig_vix = go.Figure()
                        fig_vix.add_trace(go.Scatter(x=snip['vix_data'].index, y=snip['vix_data']['Close'], line=dict(color=vix_col, width=1), fill='tozeroy'))
                        fig_vix.update_layout(height=100, margin=dict(l=0, r=0, t=0, b=0), showlegend=False, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', yaxis=dict(showgrid=False))
                        st.plotly_chart(fig_vix, use_container_width=True)

            else:
                st.info("Esperando datos intradía para rastreo.")

            # Lógica de Auto-refresco
            if live_tracking:
                time.sleep(60)
                st.rerun()

    with tab_history:
        st.subheader("📝 Bitácora del Trader")
        history_df = market_db.get_predictions(ticker, limit=20)
        if not history_df.empty:
            history_df['Dirección'] = history_df['prediction_value'].map({1: 'Subida 🟢', 0: 'Bajada 🔴', -1: 'Neutral 🟡'})
            st.dataframe(history_df[['execution_date', 'prediction_date', 'Dirección', 'prob_up', 'prob_down']], use_container_width=True)
        else:
            st.info("No hay registros en la bitácora aún.")



    st.markdown("---")
    st.caption("⚠️ Advertencia: Plataforma de análisis algorítmico. No constituye asesoría financiera directa.")

if __name__ == '__main__':
    main()
