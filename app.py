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

# --- DEEP DIVE ANALYSIS MODULE ---

def get_deep_financials(ticker):
    """Extrae datos financieros profundos: Balance, Income Statement, Cash Flow."""
    try:
        t = yf.Ticker(ticker)
        info = t.info
        
        # --- DATOS GENERALES ---
        price = info.get('currentPrice', info.get('regularMarketPrice', 0))
        high_52 = info.get('fiftyTwoWeekHigh', 0)
        low_52 = info.get('fiftyTwoWeekLow', 0)
        dist_52h = ((price / high_52) - 1) * 100 if high_52 > 0 else 0
        dist_52l = ((price / low_52) - 1) * 100 if low_52 > 0 else 0
        
        general = {
            'shortName': info.get('shortName', ticker),
            'sector': info.get('sector', 'N/D'),
            'industry': info.get('industry', 'N/D'),
            'country': info.get('country', 'N/D'),
            'employees': info.get('fullTimeEmployees', 0),
            'website': info.get('website', ''),
            'summary': info.get('longBusinessSummary', 'Sin descripción disponible.'),
            'price': price,
            'high_52': high_52,
            'low_52': low_52,
            'dist_52h': round(dist_52h, 1),
            'dist_52l': round(dist_52l, 1),
            'marketCap': info.get('marketCap', 0),
            'enterpriseValue': info.get('enterpriseValue', 0),
        }
        
        # --- VALUACIÓN ---
        valuation = {
            'trailingPE': info.get('trailingPE', None),
            'forwardPE': info.get('forwardPE', None),
            'pegRatio': info.get('pegRatio', None),
            'priceToBook': info.get('priceToBook', None),
            'priceToSales': info.get('priceToSalesTrailing12Months', None),
            'evToRevenue': info.get('enterpriseToRevenue', None),
            'evToEbitda': info.get('enterpriseToEbitda', None),
        }
        
        # --- RENTABILIDAD ---
        profitability = {
            'profitMargin': info.get('profitMargins', None),
            'operatingMargin': info.get('operatingMargins', None),
            'grossMargin': info.get('grossMargins', None),
            'returnOnEquity': info.get('returnOnEquity', None),
            'returnOnAssets': info.get('returnOnAssets', None),
        }
        
        # --- CRECIMIENTO ---
        growth = {
            'revenueGrowth': info.get('revenueGrowth', None),
            'earningsGrowth': info.get('earningsGrowth', None),
            'earningsQuarterlyGrowth': info.get('earningsQuarterlyGrowth', None),
            'revenueQuarterlyGrowth': info.get('revenueQuarterlyGrowth', None),
        }
        
        # --- SOLVENCIA Y DEUDA ---
        solvency = {
            'totalDebt': info.get('totalDebt', 0),
            'totalCash': info.get('totalCash', 0),
            'debtToEquity': info.get('debtToEquity', None),
            'currentRatio': info.get('currentRatio', None),
            'quickRatio': info.get('quickRatio', None),
            'freeCashflow': info.get('freeCashflow', 0),
            'operatingCashflow': info.get('operatingCashflow', 0),
        }
        
        # --- DIVIDENDOS ---
        dividends = {
            'dividendYield': info.get('dividendYield', None),
            'dividendRate': info.get('dividendRate', None),
            'payoutRatio': info.get('payoutRatio', None),
            'exDividendDate': info.get('exDividendDate', None),
            'fiveYearAvgDividendYield': info.get('fiveYearAvgDividendYield', None),
        }
        
        # --- RIESGO Y VOLATILIDAD ---
        risk = {
            'beta': info.get('beta', None),
            'shortRatio': info.get('shortRatio', None),
            'shortPercentOfFloat': info.get('shortPercentOfFloat', None),
            'heldPercentInsiders': info.get('heldPercentInsiders', None),
            'heldPercentInstitutions': info.get('heldPercentInstitutions', None),
        }
        
        # --- EPS Y TARGETS ANALISTAS ---
        analyst = {
            'trailingEps': info.get('trailingEps', None),
            'forwardEps': info.get('forwardEps', None),
            'targetHighPrice': info.get('targetHighPrice', None),
            'targetLowPrice': info.get('targetLowPrice', None),
            'targetMeanPrice': info.get('targetMeanPrice', None),
            'targetMedianPrice': info.get('targetMedianPrice', None),
            'recommendationKey': info.get('recommendationKey', None),
            'recommendationMean': info.get('recommendationMean', None),
            'numberOfAnalystOpinions': info.get('numberOfAnalystOpinions', 0),
        }
        
        # --- INGRESOS HISTÓRICOS (Income Statement) ---
        income_history = []
        try:
            inc = t.financials
            if inc is not None and not inc.empty:
                for col in inc.columns:
                    year_data = {}
                    year_data['period'] = col.strftime('%Y') if hasattr(col, 'strftime') else str(col)
                    year_data['totalRevenue'] = float(inc.loc['Total Revenue', col]) if 'Total Revenue' in inc.index else 0
                    year_data['grossProfit'] = float(inc.loc['Gross Profit', col]) if 'Gross Profit' in inc.index else 0
                    year_data['operatingIncome'] = float(inc.loc['Operating Income', col]) if 'Operating Income' in inc.index else 0
                    year_data['netIncome'] = float(inc.loc['Net Income', col]) if 'Net Income' in inc.index else 0
                    year_data['ebitda'] = float(inc.loc['EBITDA', col]) if 'EBITDA' in inc.index else 0
                    income_history.append(year_data)
        except:
            pass

        # --- BALANCE (Balance Sheet) ---
        balance_data = {}
        try:
            bs = t.balance_sheet
            if bs is not None and not bs.empty:
                latest = bs.iloc[:, 0]
                balance_data['totalAssets'] = float(latest.get('Total Assets', 0))
                balance_data['totalLiabilities'] = float(latest.get('Total Liabilities Net Minority Interest', latest.get('Total Debt', 0)))
                balance_data['totalEquity'] = float(latest.get('Stockholders Equity', latest.get('Total Equity Gross Minority Interest', 0)))
                balance_data['cash'] = float(latest.get('Cash And Cash Equivalents', 0))
                balance_data['totalDebt'] = float(latest.get('Total Debt', 0))
        except:
            pass

        # --- CASH FLOW ---
        cashflow_data = {}
        try:
            cf = t.cashflow
            if cf is not None and not cf.empty:
                latest = cf.iloc[:, 0]
                cashflow_data['operatingCashflow'] = float(latest.get('Operating Cash Flow', latest.get('Total Cash From Operating Activities', 0)))
                cashflow_data['capitalExpenditure'] = float(latest.get('Capital Expenditure', 0))
                cashflow_data['freeCashflow'] = cashflow_data['operatingCashflow'] + cashflow_data['capitalExpenditure']
                cashflow_data['dividendsPaid'] = float(latest.get('Common Stock Dividend Paid', latest.get('Cash Dividends Paid', 0)))
                cashflow_data['shareRepurchase'] = float(latest.get('Repurchase Of Capital Stock', 0))
        except:
            pass
        
        return {
            'general': general,
            'valuation': valuation,
            'profitability': profitability,
            'growth': growth,
            'solvency': solvency,
            'dividends': dividends,
            'risk': risk,
            'analyst': analyst,
            'income_history': income_history,
            'balance': balance_data,
            'cashflow': cashflow_data,
            'success': True
        }
    except Exception as e:
        return {'success': False, 'error': str(e)}


def calculate_financial_health_score(fin_data):
    """Calcula un score de salud financiera (0-100) basado en 6 dimensiones."""
    scores = {}
    details = {}
    
    # 1. VALUACIÓN (¿Está cara o barata?) - 20 pts máx
    val_score = 10  # Neutral base
    val_notes = []
    pe = fin_data['valuation'].get('forwardPE') or fin_data['valuation'].get('trailingPE')
    peg = fin_data['valuation'].get('pegRatio')
    pb = fin_data['valuation'].get('priceToBook')
    
    if pe is not None:
        if pe < 0: val_score = 3; val_notes.append("P/E negativo (pérdidas)")
        elif pe < 15: val_score = 18; val_notes.append("Valuación atractiva")
        elif pe < 25: val_score = 14; val_notes.append("Valuación razonable")
        elif pe < 40: val_score = 8; val_notes.append("Valuación elevada")
        else: val_score = 4; val_notes.append("Valuación extrema")
    
    if peg is not None and peg > 0:
        if peg < 1.0: val_score = min(val_score + 4, 20); val_notes.append("PEG < 1 (subvaluada vs crecimiento)")
        elif peg > 2.5: val_score = max(val_score - 3, 0); val_notes.append("PEG alto (cara vs crecimiento)")
    
    scores['Valuación'] = min(val_score, 20)
    details['Valuación'] = val_notes
    
    # 2. RENTABILIDAD (¿Es un buen negocio?) - 20 pts máx
    prof_score = 0
    prof_notes = []
    roe = fin_data['profitability'].get('returnOnEquity')
    margin = fin_data['profitability'].get('profitMargin')
    op_margin = fin_data['profitability'].get('operatingMargin')
    
    if roe is not None:
        if roe > 0.25: prof_score += 8; prof_notes.append("ROE excelente (>25%)")
        elif roe > 0.15: prof_score += 6; prof_notes.append("ROE bueno (>15%)")
        elif roe > 0.08: prof_score += 4; prof_notes.append("ROE aceptable")
        elif roe > 0: prof_score += 2; prof_notes.append("ROE bajo")
        else: prof_notes.append("ROE negativo")
    
    if margin is not None:
        if margin > 0.20: prof_score += 7; prof_notes.append("Margen neto robusto (>20%)")
        elif margin > 0.10: prof_score += 5; prof_notes.append("Margen neto saludable")
        elif margin > 0.05: prof_score += 3; prof_notes.append("Margen neto ajustado")
        elif margin > 0: prof_score += 1; prof_notes.append("Margen neto mínimo")
        else: prof_notes.append("Margen negativo (pérdidas)")
    
    if op_margin is not None and op_margin > 0.15: prof_score += 5
    
    scores['Rentabilidad'] = min(prof_score, 20)
    details['Rentabilidad'] = prof_notes
    
    # 3. CRECIMIENTO (¿Tiene futuro?) - 20 pts máx
    growth_score = 0
    growth_notes = []
    rev_g = fin_data['growth'].get('revenueGrowth')
    earn_g = fin_data['growth'].get('earningsGrowth')
    
    if rev_g is not None:
        if rev_g > 0.25: growth_score += 10; growth_notes.append("Crecimiento de ingresos explosivo (>25%)")
        elif rev_g > 0.10: growth_score += 7; growth_notes.append("Crecimiento de ingresos sólido")
        elif rev_g > 0: growth_score += 4; growth_notes.append("Crecimiento de ingresos moderado")
        else: growth_score += 1; growth_notes.append("Ingresos en contracción")
    
    if earn_g is not None:
        if earn_g > 0.20: growth_score += 10; growth_notes.append("Beneficios creciendo rápidamente")
        elif earn_g > 0.05: growth_score += 6; growth_notes.append("Beneficios creciendo")
        elif earn_g > 0: growth_score += 3; growth_notes.append("Beneficios estables")
        else: growth_score += 0; growth_notes.append("Beneficios cayendo")
    
    scores['Crecimiento'] = min(growth_score, 20)
    details['Crecimiento'] = growth_notes
    
    # 4. SOLVENCIA (¿Puede sobrevivir una crisis?) - 20 pts máx
    solv_score = 10  # Neutral
    solv_notes = []
    de_ratio = fin_data['solvency'].get('debtToEquity')
    current = fin_data['solvency'].get('currentRatio')
    fcf = fin_data['solvency'].get('freeCashflow', 0)
    
    if de_ratio is not None:
        if de_ratio < 30: solv_score = 18; solv_notes.append("Deuda mínima (fortress balance)")
        elif de_ratio < 80: solv_score = 14; solv_notes.append("Deuda controlada")
        elif de_ratio < 150: solv_score = 8; solv_notes.append("Deuda considerable")
        else: solv_score = 4; solv_notes.append("Alto apalancamiento")
    
    if current is not None:
        if current > 2.0: solv_score = min(solv_score + 4, 20); solv_notes.append("Liquidez excelente")
        elif current < 1.0: solv_score = max(solv_score - 4, 0); solv_notes.append("Riesgo de liquidez")
    
    if fcf and fcf > 0: solv_notes.append("Generación de caja positiva")
    elif fcf and fcf < 0: solv_score = max(solv_score - 3, 0); solv_notes.append("Cash flow negativo")
    
    scores['Solvencia'] = min(solv_score, 20)
    details['Solvencia'] = solv_notes
    
    # 5. MOMENTUM Y CONSENSO ANALISTAS - 10 pts máx
    mom_score = 5
    mom_notes = []
    rec = fin_data['analyst'].get('recommendationMean')
    target_mean = fin_data['analyst'].get('targetMeanPrice')
    price = fin_data['general'].get('price', 0)
    
    if rec is not None:
        if rec <= 1.5: mom_score = 10; mom_notes.append("Consenso: Strong Buy")
        elif rec <= 2.2: mom_score = 8; mom_notes.append("Consenso: Buy")
        elif rec <= 3.0: mom_score = 5; mom_notes.append("Consenso: Hold")
        elif rec <= 3.8: mom_score = 3; mom_notes.append("Consenso: Underperform")
        else: mom_score = 1; mom_notes.append("Consenso: Sell")
    
    if target_mean and price and price > 0:
        upside = ((target_mean / price) - 1) * 100
        mom_notes.append(f"Upside analistas: {upside:+.1f}%")
        if upside > 20: mom_score = min(mom_score + 2, 10)
        elif upside < -10: mom_score = max(mom_score - 2, 0)
    
    scores['Consenso'] = min(mom_score, 10)
    details['Consenso'] = mom_notes
    
    # 6. RIESGO ESTRUCTURAL - 10 pts máx (10 = bajo riesgo)
    risk_score = 7
    risk_notes = []
    beta = fin_data['risk'].get('beta')
    short_pct = fin_data['risk'].get('shortPercentOfFloat')
    insiders = fin_data['risk'].get('heldPercentInsiders')
    
    if beta is not None:
        if beta < 0.8: risk_score = 9; risk_notes.append("Baja volatilidad (defensivo)")
        elif beta < 1.2: risk_score = 7; risk_notes.append("Volatilidad de mercado")
        elif beta < 1.8: risk_score = 4; risk_notes.append("Alta volatilidad")
        else: risk_score = 2; risk_notes.append("Volatilidad extrema")
    
    if short_pct is not None:
        if short_pct > 0.15: risk_score = max(risk_score - 3, 0); risk_notes.append("Alto short interest (>15%)")
        elif short_pct > 0.08: risk_notes.append("Short interest moderado")
    
    if insiders is not None and insiders > 0.10: risk_notes.append(f"Insiders poseen {insiders*100:.1f}%")
    
    scores['Riesgo'] = min(risk_score, 10)
    details['Riesgo'] = risk_notes
    
    # TOTAL
    total = sum(scores.values())
    
    # Clasificación
    if total >= 80: grade = "A+"; grade_text = "INVERSIÓN PREMIUM"; grade_color = "#00e676"
    elif total >= 70: grade = "A"; grade_text = "MUY BUENA INVERSIÓN"; grade_color = "#28a745"
    elif total >= 60: grade = "B+"; grade_text = "BUENA INVERSIÓN"; grade_color = "#4ade80"
    elif total >= 50: grade = "B"; grade_text = "INVERSIÓN ACEPTABLE"; grade_color = "#ffc107"
    elif total >= 40: grade = "C"; grade_text = "INVERSIÓN ESPECULATIVA"; grade_color = "#fd7e14"
    elif total >= 30: grade = "D"; grade_text = "ALTO RIESGO"; grade_color = "#dc3545"
    else: grade = "F"; grade_text = "NO RECOMENDADA"; grade_color = "#ff1744"
    
    return {
        'scores': scores,
        'details': details,
        'total': total,
        'grade': grade,
        'grade_text': grade_text,
        'grade_color': grade_color
    }


def get_deep_technical_analysis(ticker):
    """Análisis técnico profundo multi-timeframe."""
    try:
        t = yf.Ticker(ticker)
        
        # Datos diarios (6 meses)
        daily = t.history(period='6mo', interval='1d', auto_adjust=True)
        if daily.empty:
            return None
        
        if isinstance(daily.columns, pd.MultiIndex):
            daily.columns = daily.columns.get_level_values(0)
        
        price = daily['Close'].iloc[-1]
        
        # Medias Móviles
        ema_9 = daily['Close'].ewm(span=9).mean().iloc[-1]
        ema_20 = daily['Close'].ewm(span=20).mean().iloc[-1]
        ema_50 = daily['Close'].ewm(span=50).mean().iloc[-1]
        sma_200 = daily['Close'].rolling(200).mean().iloc[-1] if len(daily) >= 200 else None
        
        # RSI
        rsi = calculate_rsi(daily['Close']).iloc[-1]
        
        # MACD
        exp1 = daily['Close'].ewm(span=12, adjust=False).mean()
        exp2 = daily['Close'].ewm(span=26, adjust=False).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=9, adjust=False).mean()
        macd_val = macd.iloc[-1]
        signal_val = signal.iloc[-1]
        macd_hist = macd_val - signal_val
        
        # Bollinger Bands
        bb_mid = daily['Close'].rolling(20).mean().iloc[-1]
        bb_std = daily['Close'].rolling(20).std().iloc[-1]
        bb_upper = bb_mid + (bb_std * 2)
        bb_lower = bb_mid - (bb_std * 2)
        bb_position = ((price - bb_lower) / (bb_upper - bb_lower)) * 100 if (bb_upper - bb_lower) > 0 else 50
        
        # ATR y ADX
        atr = calculate_atr(daily).iloc[-1]
        adx = calculate_adx(daily).iloc[-1]
        
        # Volumen
        avg_vol_20 = daily['Volume'].tail(20).mean()
        avg_vol_50 = daily['Volume'].tail(50).mean()
        vol_today = daily['Volume'].iloc[-1]
        vol_ratio = vol_today / avg_vol_20 if avg_vol_20 > 0 else 0
        
        # Cambios porcentuales
        chg_1d = ((price / daily['Close'].iloc[-2]) - 1) * 100 if len(daily) > 1 else 0
        chg_5d = ((price / daily['Close'].iloc[-6]) - 1) * 100 if len(daily) > 5 else 0
        chg_1m = ((price / daily['Close'].iloc[-22]) - 1) * 100 if len(daily) > 22 else 0
        chg_3m = ((price / daily['Close'].iloc[-66]) - 1) * 100 if len(daily) > 66 else 0
        chg_6m = ((price / daily['Close'].iloc[0]) - 1) * 100
        
        # Fibonacci (6 meses)
        fib_high = daily['High'].max()
        fib_low = daily['Low'].min()
        fib_diff = fib_high - fib_low
        fib_levels = {
            '0% (Max)': fib_high,
            '23.6%': fib_high - 0.236 * fib_diff,
            '38.2%': fib_high - 0.382 * fib_diff,
            '50%': fib_high - 0.5 * fib_diff,
            '61.8%': fib_high - 0.618 * fib_diff,
            '78.6%': fib_high - 0.786 * fib_diff,
            '100% (Min)': fib_low
        }
        
        # Tendencia
        trend_score = 0
        if price > ema_20: trend_score += 1
        if price > ema_50: trend_score += 1
        if sma_200 and price > sma_200: trend_score += 1
        if ema_20 > ema_50: trend_score += 1
        if macd_hist > 0: trend_score += 1
        
        if trend_score >= 4: trend_status = "ALCISTA FUERTE"
        elif trend_score >= 3: trend_status = "ALCISTA"
        elif trend_score >= 2: trend_status = "NEUTRAL"
        elif trend_score >= 1: trend_status = "BAJISTA"
        else: trend_status = "BAJISTA FUERTE"
        
        # Volatilidad diaria
        daily_vol = daily['Close'].pct_change().tail(20).std()
        
        return {
            'price': price,
            'ema_9': ema_9, 'ema_20': ema_20, 'ema_50': ema_50, 'sma_200': sma_200,
            'rsi': rsi,
            'macd': macd_val, 'signal': signal_val, 'macd_hist': macd_hist,
            'bb_upper': bb_upper, 'bb_lower': bb_lower, 'bb_position': bb_position,
            'atr': atr, 'adx': adx,
            'vol_ratio': vol_ratio, 'avg_vol_20': avg_vol_20,
            'chg_1d': chg_1d, 'chg_5d': chg_5d, 'chg_1m': chg_1m, 'chg_3m': chg_3m, 'chg_6m': chg_6m,
            'fib_levels': fib_levels,
            'trend_score': trend_score, 'trend_status': trend_status,
            'daily_data': daily,
            'daily_vol': daily_vol,
        }
    except Exception as e:
        print(f"Error Deep Technical: {e}")
        return None


def ai_deep_dive_analysis(api_key, ticker, fin_data, tech_data, options_data=None):
    """Genera un análisis IA exhaustivo de inversión combinando TODOS los datos."""
    if not api_key:
        return "Sin API Key de IA. Configura GROQ_API_KEY en .env"
    
    try:
        g = fin_data['general']
        v = fin_data['valuation']
        p = fin_data['profitability']
        gr = fin_data['growth']
        s = fin_data['solvency']
        a = fin_data['analyst']
        r = fin_data['risk']
        
        # Formateo
        def fmt_num(val, suffix='', pct=False, billions=False):
            if val is None: return 'N/D'
            if billions:
                if abs(val) >= 1e12: return f"{val/1e12:.1f}T{suffix}"
                if abs(val) >= 1e9: return f"{val/1e9:.1f}B{suffix}"
                if abs(val) >= 1e6: return f"{val/1e6:.0f}M{suffix}"
                return f"{val:,.0f}{suffix}"
            if pct: return f"{val*100:.1f}%"
            return f"{val:.2f}{suffix}"
        
        # Fibonacci string
        fib_str = ""
        if tech_data and tech_data.get('fib_levels'):
            for name, price in tech_data['fib_levels'].items():
                fib_str += f"\n  {name}: {price:.2f}"
        
        # Income history string
        inc_str = ""
        for yr in fin_data.get('income_history', [])[:4]:
            inc_str += f"\n  {yr['period']}: Ingresos={fmt_num(yr['totalRevenue'], billions=True)} | Beneficio Neto={fmt_num(yr['netIncome'], billions=True)} | EBITDA={fmt_num(yr['ebitda'], billions=True)}"
        
        prompt = f"""Eres el DIRECTOR DE INVESTIGACIÓN de un Fondo de Cobertura con 20 años de experiencia. 
Realiza un ANÁLISIS DE INVERSIÓN INSTITUCIONAL sobre esta empresa. Tu informe será leído por inversores sofisticados.

═══════════════════════════════════════
 PERFIL: {g['shortName']} ({ticker})
═══════════════════════════════════════
Sector: {g.get('sector', 'N/D')} | Industria: {g.get('industry', 'N/D')} | País: {g.get('country', 'N/D')}
Empleados: {g.get('employees', 0) or 0:,}
Capitalización: {fmt_num(g.get('marketCap', 0), billions=True)}
Enterprise Value: {fmt_num(g.get('enterpriseValue', 0), billions=True)}
Precio Actual: {fmt_num(g.get('price', 0))} | 52W High: {fmt_num(g.get('high_52', 0))} ({g.get('dist_52h', 0) or 0:+.1f}%) | 52W Low: {fmt_num(g.get('low_52', 0))} ({g.get('dist_52l', 0) or 0:+.1f}%)

═══ VALUACIÓN ═══
P/E Trailing: {fmt_num(v['trailingPE'], 'x')} | P/E Forward: {fmt_num(v['forwardPE'], 'x')}
PEG Ratio: {fmt_num(v['pegRatio'])} | P/B: {fmt_num(v['priceToBook'], 'x')}
P/S: {fmt_num(v['priceToSales'], 'x')} | EV/Revenue: {fmt_num(v['evToRevenue'], 'x')} | EV/EBITDA: {fmt_num(v['evToEbitda'], 'x')}

═══ RENTABILIDAD ═══
Margen Bruto: {fmt_num(p['grossMargin'], pct=True)} | Margen Operativo: {fmt_num(p['operatingMargin'], pct=True)} | Margen Neto: {fmt_num(p['profitMargin'], pct=True)}
ROE: {fmt_num(p['returnOnEquity'], pct=True)} | ROA: {fmt_num(p['returnOnAssets'], pct=True)}

═══ CRECIMIENTO ═══
Crecimiento Ingresos: {fmt_num(gr['revenueGrowth'], pct=True)} | Crecimiento Beneficios: {fmt_num(gr['earningsGrowth'], pct=True)}
Crecimiento Trimestral (Rev): {fmt_num(gr['revenueQuarterlyGrowth'], pct=True)} | (Earn): {fmt_num(gr['earningsQuarterlyGrowth'], pct=True)}

═══ HISTORIAL FINANCIERO ═══{inc_str if inc_str else '  No disponible'}

═══ SOLVENCIA Y DEUDA ═══
Deuda Total: {fmt_num(s['totalDebt'], billions=True)} | Cash: {fmt_num(s['totalCash'], billions=True)}
Deuda/Equity: {fmt_num(s['debtToEquity'])} | Current Ratio: {fmt_num(s['currentRatio'])} | Quick Ratio: {fmt_num(s['quickRatio'])}
Free Cash Flow: {fmt_num(s['freeCashflow'], billions=True)} | Operating CF: {fmt_num(s['operatingCashflow'], billions=True)}

═══ RIESGO ═══
Beta: {fmt_num(r['beta'])} | Short Ratio: {fmt_num(r['shortRatio'])}
Short % Float: {fmt_num(r['shortPercentOfFloat'], pct=True)}
Insiders: {fmt_num(r['heldPercentInsiders'], pct=True)} | Instituciones: {fmt_num(r['heldPercentInstitutions'], pct=True)}

═══ CONSENSO ANALISTAS ({a['numberOfAnalystOpinions']} analistas) ═══
Recomendación: {a.get('recommendationKey', 'N/D')} (Score: {fmt_num(a['recommendationMean'])}/5)
EPS Trailing: {fmt_num(a['trailingEps'])} | EPS Forward: {fmt_num(a['forwardEps'])}
Precio Objetivo: Min {fmt_num(a['targetLowPrice'])} | Media {fmt_num(a['targetMeanPrice'])} | Max {fmt_num(a['targetHighPrice'])}"""

        if tech_data:
            sma200_str = f"{tech_data['sma_200']:.2f}" if tech_data.get('sma_200') else 'N/D'
            prompt += f"""

═══ ANÁLISIS TÉCNICO ═══
Tendencia: {tech_data['trend_status']} (Score: {tech_data['trend_score']}/5)
Precio: {tech_data.get('price', 0):.2f} | EMA9: {tech_data.get('ema_9', 0):.2f} | EMA20: {tech_data.get('ema_20', 0):.2f} | EMA50: {tech_data.get('ema_50', 0):.2f}
SMA200: {sma200_str}
RSI(14): {tech_data.get('rsi', 0):.1f} | ADX: {tech_data.get('adx', 0):.1f}
MACD: {tech_data.get('macd', 0):.4f} | Señal: {tech_data.get('signal', 0):.4f} | Histograma: {tech_data.get('macd_hist', 0):.4f}
Bollinger: Posición {tech_data.get('bb_position', 50):.0f}% (Lower: {tech_data.get('bb_lower', 0):.2f} | Upper: {tech_data.get('bb_upper', 0):.2f})
ATR(14): {tech_data.get('atr', 0):.2f} | Vol Ratio: {tech_data.get('vol_ratio', 0):.2f}x
Performance: 1D={tech_data.get('chg_1d', 0):+.2f}% | 5D={tech_data.get('chg_5d', 0):+.2f}% | 1M={tech_data.get('chg_1m', 0):+.2f}% | 3M={tech_data.get('chg_3m', 0):+.2f}% | 6M={tech_data.get('chg_6m', 0):+.2f}%
Fibonacci (6M):{fib_str}"""
        
        if options_data:
            prompt += f"""

═══ FLUJO DE OPCIONES (Smart Money) ═══
Put/Call Ratio: {options_data.get('ratio', 0):.2f} | Sentimiento: {options_data.get('sent', 'N/D')}
Call Wall (Resistencia): {options_data.get('call_wall', 0):,.0f} | Put Wall (Soporte): {options_data.get('put_wall', 0):,.0f}
Vol. Calls: {int(options_data.get('vol_calls', 0)):,} | Vol. Puts: {int(options_data.get('vol_puts', 0)):,}"""

        prompt += """

═══════════════════════════════════════
TU INFORME DE INVERSIÓN (en español, estilo institucional):
═══════════════════════════════════════

Estructura tu respuesta EXACTAMENTE así:

## 1. RESUMEN EJECUTIVO
Una breve sinopsis de la empresa, su posición competitiva y tu conclusión principal (3-4 líneas).

## 2. TESIS DE INVERSIÓN
¿Por qué un inversor debería (o no) considerar esta acción? Argumenta con los datos financieros.

## 3. FORTALEZAS (BULL CASE)
Los 3 principales argumentos a favor de la inversión.

## 4. RIESGOS (BEAR CASE)
Los 3 principales riesgos o argumentos en contra.

## 5. ANÁLISIS TÉCNICO
¿Qué dice la estructura de precio? ¿En qué zona estamos? Menciona Fibonacci, RSI, tendencia y soportes/resistencias clave.

## 6. POTENCIAL DE REVALORIZACIÓN
Basándote en los targets de analistas y la estructura técnica, estima el potencial de subida/bajada (porcentaje) a 3, 6 y 12 meses.

## 7. VEREDICTO FINAL
- **Clasificación:** COMPRA FUERTE / COMPRA / MANTENER / VENDER / VENDER FUERTE
- **Horizonte ideal:** Corto / Medio / Largo plazo
- **Perfil de inversor requerido:** Conservador / Moderado / Agresivo
- **Nivel de convicción:** Bajo / Medio / Alto
- **Precio de Entrada Sugerido (si aplica)**
- **Stop Loss Sugerido**
- **Take Profit Sugerido**

REGLAS:
- Escribe con datos y argumentos, no con opiniones vagas.
- Máximo 600 palabras.
- No uses el símbolo '$' pegado a los números (usa USD o solo el número).
- No uses formato LaTeX.
- Al final, incluye la línea: [SL: valor, TP: valor]"""

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": "llama-3.3-70b-versatile",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.3,
            "max_tokens": 2000
        }
        resp = requests.post("https://api.groq.com/openai/v1/chat/completions",
                           headers=headers, json=payload, timeout=30)
        if resp.status_code == 200:
            return resp.json()['choices'][0]['message']['content']
        return f"Error API ({resp.status_code}): {resp.text[:200]}"
    except Exception as e:
        return f"Error en análisis IA: {str(e)}"


# --- MOMENTUM SCANNER ---
SCANNER_UNIVERSE = [
    'GLD', 'XBI', 'XLB', 'XLC', 'XLE', 'XLF', 'XLI', 'XLK', 'XLP', 'XLRE', 'XLU', 'XLV', 'XLY', 'XOP',
    'AAPL', 'MSFT', 'NVDA', 'AMZN', 'GOOGL', 'META', 'TSLA', 'BRK-B', 'AVGO', 'JPM',
    'LLY', 'V', 'UNH', 'MA', 'XOM', 'COST', 'HD', 'PG', 'JNJ', 'ABBV',
    'WMT', 'NFLX', 'CRM', 'BAC', 'ORCL', 'CVX', 'KO', 'MRK', 'AMD', 'PEP',
    'TMO', 'CSCO', 'ACN', 'LIN', 'ADBE', 'MCD', 'ABT', 'WFC', 'DHR', 'PM',
    'NOW', 'TXN', 'QCOM', 'INTU', 'ISRG', 'CAT', 'IBM', 'GE', 'AMAT', 'AMGN',
    'VZ', 'BKNG', 'AXP', 'MS', 'GS', 'SPGI', 'BLK', 'PFE', 'T', 'LOW',
    'NEE', 'UNP', 'RTX', 'HON', 'SYK', 'DE', 'BA', 'LMT', 'SBUX', 'MMM',
    'GILD', 'MDLZ', 'ADI', 'LRCX', 'KLAC', 'PANW', 'SNPS', 'CDNS', 'MRVL', 'CRWD',
    'PLTR', 'COIN', 'SQ', 'SHOP', 'SNOW', 'DKNG', 'MELI', 'MU', 'INTC', 'PYPL',
    'ABNB', 'ARM', 'SMCI', 'DASH', 'UBER', 'NET', 'ZS', 'RBLX', 'ENPH', 'RIVN',
    'SOFI', 'SNAP', 'HOOD', 'CLSK', 'MARA', 'RIOT', 'F', 'AAL', 'VALE', 'NIO',
    'PARA', 'PINS', 'UPST', 'AFRM', 'AI', 'PLUG', 'RUN', 'BABA', 'JD', 'CPNG',
    'GRAB', 'SE', 'NU', 'DLO', 'U', 'ROKU', 'GME', 'XPEV', 'LI', 'FUTU'
]


@st.cache_data(ttl=900, show_spinner=False)
def scan_momentum_stocks(price_min, price_max, min_volume, smooth_momentum=False):

    """Escanea el universo de acciones buscando momentum alcista."""
    results = []
    progress_text = st.empty()
    progress_bar = st.progress(0)
    total = len(SCANNER_UNIVERSE)
    
    for i, ticker in enumerate(SCANNER_UNIVERSE):
        progress_bar.progress((i + 1) / total)
        progress_text.caption(f"Escaneando {ticker}... ({i+1}/{total})")
        try:
            t = yf.Ticker(ticker)
            hist = t.history(period='3mo', interval='1d', auto_adjust=True)
            if hist.empty or len(hist) < 20:
                continue
            
            price = hist['Close'].iloc[-1]
            # Filtro de precio
            if price < price_min or price > price_max:
                continue
            
            # Filtro de volumen
            avg_vol = hist['Volume'].tail(20).mean()
            if avg_vol < min_volume:
                continue
            
            # Calcular indicadores técnicos
            ema_20 = hist['Close'].ewm(span=20).mean().iloc[-1]
            ema_50 = hist['Close'].ewm(span=50).mean().iloc[-1]
            
            # RSI
            delta = hist['Close'].diff()
            gain = delta.where(delta > 0, 0).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss.replace(0, np.nan)
            rsi = (100 - (100 / (1 + rs))).iloc[-1]
            
            # Cambios porcentuales
            chg_1d = ((price / hist['Close'].iloc[-2]) - 1) * 100 if len(hist) > 1 else 0
            chg_5d = ((price / hist['Close'].iloc[-6]) - 1) * 100 if len(hist) > 5 else 0
            chg_20d = ((price / hist['Close'].iloc[-21]) - 1) * 100 if len(hist) > 20 else 0
            
            # Volumen relativo (hoy vs promedio)
            vol_today = hist['Volume'].iloc[-1]
            vol_ratio = vol_today / avg_vol if avg_vol > 0 else 0
            
            # Volatilidad (Desviación estándar de retornos diarios - 20 días)
            daily_vol = hist['Close'].pct_change().tail(20).std()

            
            # Momentum Score (0-100)
            score = 0
            # Precio > EMA 20 (+20)
            if price > ema_20: score += 20
            # EMA 20 > EMA 50 (+20)
            if ema_20 > ema_50: score += 20
            # RSI entre 50-70 (+20) o >70 (+10)
            if 50 <= rsi <= 70: score += 20
            elif rsi > 70: score += 10
            # Cambio 5d positivo (+20)
            if chg_5d > 0: score += 20
            # Volumen superior al promedio (+20)
            if vol_ratio > 1.0: score += 20
            
            # --- FILTRO DE MOMENTUM SUAVE (Baja Volatilidad) ---
            if smooth_momentum:
                # Si la volatilidad diaria es baja (< 2%), premiamos la estabilidad
                if daily_vol < 0.02: 
                    score += 20
                # Si es muy alta (> 4%), penalizamos fuertemente
                elif daily_vol > 0.04:
                    score -= 40

            
            # Solo incluir si tiene mínimo 40 de score
            if score >= 40:
                results.append({
                    'Ticker': ticker,
                    'Precio': round(price, 2),
                    '1D%': round(chg_1d, 2),
                    '5D%': round(chg_5d, 2),
                    '20D%': round(chg_20d, 2),
                    'RSI': round(rsi, 1),
                    'Vol Ratio': round(vol_ratio, 2),
                    'Vol Avg': f"{avg_vol/1e6:.1f}M",
                    'Score': score
                })
        except Exception:
            continue
    
    progress_bar.empty()
    progress_text.empty()
    
    if results:
        df = pd.DataFrame(results).sort_values('Score', ascending=False).reset_index(drop=True)
        return df
    return pd.DataFrame()

def get_stock_fundamentals(ticker):
    """Obtiene datos fundamentales de una acción vía yfinance."""
    try:
        t = yf.Ticker(ticker)
        info = t.info
        
        # 52-week high distance
        price = info.get('currentPrice', info.get('regularMarketPrice', 0))
        high_52 = info.get('fiftyTwoWeekHigh', 0)
        dist_52 = ((price / high_52) - 1) * 100 if high_52 > 0 else 0
        
        return {
            'Sector': info.get('sector', 'N/D'),
            'Industry': info.get('industry', 'N/D'),
            'Market Cap': info.get('marketCap', 0),
            'P/E Ratio': info.get('trailingPE', info.get('forwardPE', 'N/D')),
            'Revenue Growth': info.get('revenueGrowth', 'N/D'),
            'Profit Margin': info.get('profitMargins', 'N/D'),
            'Beta': info.get('beta', 'N/D'),
            'EPS': info.get('trailingEps', 'N/D'),
            '52W High Dist': round(dist_52, 1),
            'Div Yield': info.get('dividendYield', 0),
            'Short Name': info.get('shortName', ticker)
        }
    except Exception as e:
        return {'Error': str(e)}

def ai_stock_analysis(api_key, ticker, tech_data, fundamentals, fib_levels=None):
    """Genera un análisis IA combinando técnicos y fundamentales."""
    try:
        # Formatear fundamentales
        pe = fundamentals.get('P/E Ratio', 'N/D')
        pe_str = f"{pe:.1f}x" if isinstance(pe, (int, float)) else str(pe)
        
        rev_g = fundamentals.get('Revenue Growth', 'N/D')
        rev_str = f"{rev_g*100:.1f}%" if isinstance(rev_g, (int, float)) else str(rev_g)
        
        margin = fundamentals.get('Profit Margin', 'N/D')
        margin_str = f"{margin*100:.1f}%" if isinstance(margin, (int, float)) else str(margin)
        
        beta = fundamentals.get('Beta', 'N/D')
        beta_str = f"{beta:.2f}" if isinstance(beta, (int, float)) else str(beta)
        
        mcap = fundamentals.get('Market Cap', 0)
        if mcap > 1e12: mcap_str = f"${mcap/1e12:.1f}T"
        elif mcap > 1e9: mcap_str = f"${mcap/1e9:.1f}B"
        else: mcap_str = f"${mcap/1e6:.0f}M"

        prompt = f"""Eres un ANALISTA SENIOR de un Fondo de Cobertura. Analiza esta acción para una operación SWING (3-10 días).

=== DATOS TÉCNICOS ===
Ticker: {ticker}
Precio: ${tech_data.get('Precio', 'N/D')}
Cambio 1D: {tech_data.get('1D%', 0)}% | 5D: {tech_data.get('5D%', 0)}% | 20D: {tech_data.get('20D%', 0)}%
RSI(14): {tech_data.get('RSI', 'N/D')}
Vol Ratio (Hoy vs Avg): {tech_data.get('Vol Ratio', 'N/D')}x
Momentum Score: {tech_data.get('Score', 0)}/100

=== DATOS FUNDAMENTALES ===
Sector: {fundamentals.get('Sector', 'N/D')} | {fundamentals.get('Industry', 'N/D')}
Market Cap: {mcap_str}
P/E Ratio: {pe_str}
Revenue Growth: {rev_str}
Profit Margin: {margin_str}
Beta: {beta_str}
Distancia al 52W High: {fundamentals.get('52W High Dist', 'N/D')}%"""
        
        # Agregar niveles Fibonacci si están disponibles
        prompt += "\n\n=== FIBONACCI (3 meses) ==="
        if fib_levels:
            for name, price in fib_levels.items():
                prompt += f"\n{name}: ${price:.2f}"
        else:
            prompt += "\nNo disponible"
        
        prompt += """

TU ANÁLISIS (en español, Max 250 palabras):
1. VEREDICTO: COMPRAR / ESPERAR / EVITAR
2. JUSTIFICACIÓN TÉCNICA (2 líneas, menciona Fibonacci si aplica)
3. JUSTIFICACIÓN FUNDAMENTAL (2 líneas)
4. NIVELES CLAVE: Soporte y Resistencia basados en Fibonacci
5. RIESGO: Bajo / Medio / Alto (y por qué)
6. HORIZONTE: Cuánto tiempo mantener la posición
7. STOP LOSS RECOMENDADO: Nivel técnico sugerido basado en Fibonacci (inferior al soporte actual).
8. TAKE PROFIT RECOMENDADO: Nivel técnico sugerido basado en Fibonacci (Resistencia superior).

IMPORTANTE: 
- Al final del análisis, incluye SIEMPRE una línea exacta con este formato: [SL: valor, TP: valor] usando números decimales.
- No uses el símbolo '$' pegado a los números, usa 'USD' o simplemente el número si es claro por el contexto.
- Asegúrate de separar bien las palabras con espacios. NO uses formato LaTeX.
"""




        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": "llama-3.3-70b-versatile",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.3,
            "max_tokens": 600
        }
        resp = requests.post("https://api.groq.com/openai/v1/chat/completions", 
                           headers=headers, json=payload, timeout=20)
        if resp.status_code == 200:
            return resp.json()['choices'][0]['message']['content']
        return "Error al consultar la IA. Intenta de nuevo."
    except Exception as e:
        return f"Error IA: {str(e)}"

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
        # Usar ruta absoluta para evitar ambigüedades
        self.db_file = os.path.abspath(db_file)
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
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS journal (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticker TEXT,
                    entry_date TEXT,
                    entry_price REAL,
                    sl_price REAL,
                    tp_price REAL,
                    score INTEGER,
                    verdict TEXT,
                    reasoning TEXT,
                    status TEXT DEFAULT 'OPEN'
                )
            ''')
            # Migración: Añadir columnas si no existen (evita errores en DBs viejas)
            try:
                cursor.execute("ALTER TABLE journal ADD COLUMN sl_price REAL")
                cursor.execute("ALTER TABLE journal ADD COLUMN tp_price REAL")
            except:
                pass 

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

    def save_journal_entry(self, ticker, price, score, verdict, reasoning, sl=0.0, tp=0.0):
        conn = None
        try:
            conn = sqlite3.connect(self.db_file)
            now = dt.now().strftime('%Y-%m-%d %H:%M:%S')
            
            # --- CONVERSIÓN ESTRICTA DE TIPOS ---
            t_val = str(ticker)
            d_val = str(now)
            p_val = float(price)
            s_val = int(score)
            v_val = str(verdict)
            r_val = str(reasoning)
            sl_val = float(sl)
            tp_val = float(tp)
            
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO journal (ticker, entry_date, entry_price, sl_price, tp_price, score, verdict, reasoning, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'OPEN')
            ''', (t_val, d_val, p_val, sl_val, tp_val, s_val, v_val, r_val))

            
            conn.commit()
            return True
        except Exception as e:
            print(f"CRITICAL ERROR SAVING JOURNAL: {e}")
            import traceback
            traceback.print_exc()
            return False
        finally:
            if conn:
                conn.close()

    def get_journal_entries(self):
        conn = sqlite3.connect(self.db_file)
        query = "SELECT * FROM journal ORDER BY id DESC"
        df = pd.read_sql_query(query, conn)
        conn.close()
        return df

    def delete_journal_entry(self, entry_id):
        try:
            conn = sqlite3.connect(self.db_file)
            conn.execute("DELETE FROM journal WHERE id = ?", (entry_id,))
            conn.commit()
            conn.close()
            return True
        except:
            return False

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
    tab_market, tab_stats, tab_brain, tab_scanner, tab_deepdive, tab_calendar, tab_history = st.tabs([
        "📟 Market Desk", "📊 Estadísticas", "🧠 Inteligencia IA", "🔬 Scanner", "🔍 Deep Dive", "📅 Agenda Económica", "📜 Historial"
    ])

    # --- NOTIFICACIONES GLOBALES (Instantáneas tras Guardar) ---
    if 'save_success' in st.session_state:
        msg = st.session_state.pop('save_success')
        st.balloons()
        st.toast(msg, icon="✅")
    if 'save_error' in st.session_state:
        st.error(st.session_state.pop('save_error'))

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

    # --- MOMENTUM SCANNER TAB ---
    with tab_scanner:
        st.subheader("🔬 Momentum Scanner & Rotación de Sectores")
        st.info("💡 Análisis de flujo de capital. Identifica qué sectores lideran el mercado antes de elegir una acción.")
        
        # Filtros en columnas
        filt_col1, filt_col2, filt_col3 = st.columns(3)
        with filt_col1:
            price_range = st.slider("💲 Rango de Precio ($)", min_value=5, max_value=2000, value=(10, 800), step=5)
        with filt_col2:
            vol_options = {'500K+': 500_000, '1M+': 1_000_000, '5M+': 5_000_000, '10M+': 10_000_000}
            vol_label = st.selectbox("📊 Volumen Mínimo", list(vol_options.keys()), index=1)
            min_vol = vol_options[vol_label]
        with filt_col3:
            force_filter = st.selectbox("⚡ Fuerza Mínima", ['Moderado (40+)', 'Fuerte (60+)', 'Explosivo (80+)'], index=0)
            min_score = int(force_filter.split('(')[1].replace('+)', ''))
        
        # Filtro de Volatilidad (Smooth Momentum)
        smooth_check = st.checkbox("🧘 Filtrar por 'Momentum Suave' (Busca subidas constantes, evita saltos violentos)", value=False)

        
        # --- CALCULADORA DE POSICIÓN (SIDEBAR) ---
        with st.sidebar.expander("🧮 Calculadora de Gestión de Riesgo", expanded=True):
            st.markdown("#### Planifica tu Trade")
            account_size = st.number_input("💰 Capital Total ($)", value=10000, step=1000)
            risk_pct = st.slider("⚠️ Riesgo por Operación (%)", 0.5, 5.0, 1.0, 0.5)
            
            risk_amount = account_size * (risk_pct / 100)
            st.info(f"Riesgo Máximo: **${risk_amount:.2f}**")
            
            # Inicializar valores en session_state si no existen
            if 'calc_entry' not in st.session_state: st.session_state['calc_entry'] = 0.0
            if 'calc_stop' not in st.session_state: st.session_state['calc_stop'] = 0.0
            if 'calc_tp' not in st.session_state: st.session_state['calc_tp'] = 0.0

            # Usar variables locales para los widgets, alimentados por el estado
            calc_entry = st.number_input("Precio Entrada ($)", value=float(st.session_state['calc_entry']), step=0.1)
            calc_stop = st.number_input("Stop Loss ($)", value=float(st.session_state['calc_stop']), step=0.1)
            calc_tp = st.number_input("Take Profit ($)", value=float(st.session_state['calc_tp']), step=0.1)
            
            # Actualizar el estado con el valor actual del widget para persistencia
            st.session_state['calc_entry'] = calc_entry
            st.session_state['calc_stop'] = calc_stop
            st.session_state['calc_tp'] = calc_tp

            
            if calc_entry > 0 and calc_stop > 0 and calc_entry > calc_stop:
                risk_per_share = calc_entry - calc_stop
                shares = int(risk_amount // risk_per_share)
                position_value = shares * calc_entry
                
                st.markdown("---")
                st.success(f"🎯 **Comprar: {shares} acciones**")
                st.caption(f"Valor Posición: ${position_value:,.2f}")

                # Cálculo de R/R si hay Take Profit
                if calc_tp > calc_entry:
                    profit_per_share = calc_tp - calc_entry
                    total_profit = shares * profit_per_share
                    rr_ratio = profit_per_share / risk_per_share
                    st.info(f"💰 Ganancia Estimada: **${total_profit:.2f}**\n\n⚖️ R/R Ratio: **1:{rr_ratio:.2f}**")


                
                if position_value > account_size:

                    st.warning("⚠️ ¡Cuidado! Esta posición usa margin (aplacamiento).")
            elif calc_entry > 0 and calc_stop >= calc_entry:
                st.error("El Stop Loss debe ser menor a la Entrada.")
        
        if st.button("🚀 Iniciar Escaneo de Mercado", use_container_width=True, type="primary"):
            scan_df = scan_momentum_stocks(price_range[0], price_range[1], min_vol, smooth_check)

            if not scan_df.empty:
                # Enriquecer con Sectores para el Heatmap (solo para los resultados)
                with st.spinner('Mapeando sectores...'):
                    sectors = []
                    for t in scan_df['Ticker']:
                        try:
                            # Cachear el sector para no saturar API
                            s_info = yf.Ticker(t).info.get('sector', 'Otros')
                            sectors.append(s_info)
                        except:
                            sectors.append('N/D')
                    scan_df['Sector'] = sectors
                
                scan_df = scan_df[scan_df['Score'] >= min_score].reset_index(drop=True)
                st.session_state['scan_results'] = scan_df
            else:
                st.session_state['scan_results'] = pd.DataFrame()
                st.warning("No se encontraron acciones. Intenta ampliar los filtros.")
        
        # --- NUEVA SECCIÓN: DASHBOARD DE SECTORES ---
        if 'scan_results' in st.session_state and not st.session_state['scan_results'].empty:
            df_res = st.session_state['scan_results']
            scan_df = df_res  # Asignar para uso posterior
            
            st.markdown("---")
            
            # Verificar si existe la columna 'Sector' antes de mostrar el dashboard
            if 'Sector' in df_res.columns:
                dash_col1, dash_col2 = st.columns([2, 1])
                
                with dash_col1:
                    st.markdown("#### 🔥 Mapa de Calor por Sectores (Score Promedio)")
                    sector_stats = df_res.groupby('Sector')['Score'].mean().sort_values(ascending=False)
                    
                    fig_heat = go.Figure(go.Bar(
                        x=sector_stats.values,
                        y=sector_stats.index,
                        orientation='h',
                        marker=dict(
                            color=sector_stats.values,
                            colorscale='Viridis',
                            showscale=False
                        ),
                        text=[f"{v:.1f}" for v in sector_stats.values],
                        textposition='auto',
                    ))
                    fig_heat.update_layout(
                        height=300, margin=dict(l=0, r=0, t=10, b=10),
                        template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                        xaxis=dict(title="Momentum Score Promedio", showgrid=False),
                        yaxis=dict(showgrid=False)
                    )
                    st.plotly_chart(fig_heat, use_container_width=True)

                with dash_col2:
                    st.markdown("#### 🚀 Salud del Mercado")
                    bullish_count = len(df_res[df_res['Score'] >= 70])
                    total_count = len(df_res)
                    health_pct = (bullish_count / total_count * 100) if total_count > 0 else 0
                    
                    fig_gauge = go.Figure(go.Indicator(
                        mode = "gauge+number",
                        value = health_pct,
                        domain = {'x': [0, 1], 'y': [0, 1]},
                        title = {'text': "Fuerza Bullish %", 'font': {'size': 14}},
                        gauge = {
                            'axis': {'range': [0, 100], 'tickwidth': 1},
                            'bar': {'color': "#28a745"},
                            'steps': [
                                {'range': [0, 40], 'color': "#444"},
                                {'range': [40, 70], 'color': "#666"},
                                {'range': [70, 100], 'color': "#28a745"}
                            ],
                            'threshold': {
                                'line': {'color': "white", 'width': 4},
                                'thickness': 0.75,
                                'value': 70
                            }
                        }
                    ))
                    fig_gauge.update_layout(height=250, margin=dict(l=10, r=10, t=40, b=10), template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)')
                    st.plotly_chart(fig_gauge, use_container_width=True)
                
                st.markdown("---")
            else:
                # Si no hay columna Sector (resultados antiguos), mostrar advertencia
                st.warning("⚠️ Ejecuta un nuevo escaneo para ver el análisis de sectores.")

            # Botón de Descarga de Resultados
            csv = df_res.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Descargar Resultados (CSV)",
                data=csv,
                file_name=f"momentum_scan_{dt.now().strftime('%Y-%m-%d')}.csv",
                mime='text/csv',
                use_container_width=True
            )

            st.markdown("---")
            st.success(f"✅ {len(df_res)} acciones encontradas con momentum alcista")
            
            # Colorear filas por Score
            def color_score(val):
                if val >= 80: return 'background-color: rgba(40, 167, 69, 0.4); font-weight: bold'
                elif val >= 60: return 'background-color: rgba(255, 193, 7, 0.3)'
                return ''
            
            def color_change(val):
                try:
                    v = float(val)
                    if v > 0: return 'color: #28a745'
                    elif v < 0: return 'color: #dc3545'
                except: pass
                return ''
            
            styled = scan_df.style.map(color_score, subset=['Score'])
            styled = styled.map(color_change, subset=['1D%', '5D%', '20D%'])
            st.dataframe(styled, use_container_width=True, hide_index=True, height=400)
            
            # Selección de acción para análisis profundo
            st.markdown("---")
            st.subheader("📈 Análisis Profundo")
            
            selected_ticker = st.selectbox(
                "Selecciona una acción para análisis detallado:",
                scan_df['Ticker'].tolist(),
                key='scanner_select'
            )
            
            if selected_ticker:
                chart_col, info_col = st.columns([3, 2])
                
                with chart_col:
                    # Gráfico Candlestick
                    st.caption(f"📊 Gráfico de {selected_ticker} (3 meses)")
                    try:
                        tk = yf.Ticker(selected_ticker)
                        chart_data = tk.history(period='3mo', interval='1d', auto_adjust=True)
                        
                        if not chart_data.empty:
                            ema20 = chart_data['Close'].ewm(span=20).mean()
                            ema50 = chart_data['Close'].ewm(span=50).mean()
                            
                            # Bollinger Bands
                            std = chart_data['Close'].rolling(window=20).std()
                            bb_upper = ema20 + (std * 2)
                            bb_lower = ema20 - (std * 2)
                            
                            fig = go.Figure()
                            # Velas
                            fig.add_trace(go.Candlestick(
                                x=chart_data.index,
                                open=chart_data['Open'], high=chart_data['High'],
                                low=chart_data['Low'], close=chart_data['Close'],
                                name=selected_ticker
                            ))
                            # Indicadores
                            fig.add_trace(go.Scatter(x=chart_data.index, y=ema20, name='EMA 20', line=dict(color='#00d4ff', width=1)))
                            fig.add_trace(go.Scatter(x=chart_data.index, y=ema50, name='EMA 50', line=dict(color='#ff6b35', width=1)))
                            fig.add_trace(go.Scatter(x=chart_data.index, y=bb_upper, name='BB Upper', line=dict(color='rgba(173,216,230,0.3)', width=1, dash='dot'), showlegend=False))
                            fig.add_trace(go.Scatter(x=chart_data.index, y=bb_lower, name='BB Lower', line=dict(color='rgba(173,216,230,0.3)', width=1, dash='dot'), fill='tonexty', fillcolor='rgba(173,216,230,0.05)', showlegend=False))
                            
                            # Volumen como subplot
                            vol_colors = ['#28a745' if c >= o else '#dc3545' 
                                         for c, o in zip(chart_data['Close'], chart_data['Open'])]
                            fig.add_trace(go.Bar(x=chart_data.index, y=chart_data['Volume'],
                                                name='Volumen', marker_color=vol_colors, opacity=0.2,
                                                yaxis='y2'))
                            
                            # --- FIBONACCI RETRACEMENT (sutil) ---
                            fib_high = chart_data['High'].max()
                            fib_low = chart_data['Low'].min()
                            fib_diff = fib_high - fib_low
                            
                            fib_levels = [
                                ('23.6%', fib_high - 0.236 * fib_diff, 'rgba(255,107,107,0.5)'),
                                ('38.2%', fib_high - 0.382 * fib_diff, 'rgba(255,169,77,0.5)'),
                                ('50.0%', fib_high - 0.5 * fib_diff,   'rgba(255,212,59,0.6)'),
                                ('61.8%', fib_high - 0.618 * fib_diff, 'rgba(105,219,124,0.6)'),
                                ('78.6%', fib_high - 0.786 * fib_diff, 'rgba(56,217,169,0.5)'),
                            ]
                            
                            for label, price, color in fib_levels:
                                fig.add_hline(
                                    y=price,
                                    line_dash="dash",
                                    line_color=color,
                                    line_width=0.8,
                                    annotation_text=f"{label}  ${price:.0f}",
                                    annotation_position="right",
                                    annotation_font=dict(size=8, color=color),
                                    annotation_bgcolor="rgba(0,0,0,0.5)",
                                )
                            
                            fig.update_layout(
                                height=500,
                                template='plotly_dark',
                                paper_bgcolor='rgba(0,0,0,0)',
                                plot_bgcolor='rgba(0,0,0,0)',
                                xaxis_rangeslider_visible=False,
                                legend=dict(orientation='h', yanchor='bottom', y=1.02),
                                margin=dict(l=10, r=10, t=30, b=10),
                                yaxis2=dict(overlaying='y', side='right', showgrid=False, showticklabels=False, range=[0, chart_data['Volume'].max()*4])
                            )
                            st.plotly_chart(fig, use_container_width=True)
                    except Exception as e:
                        st.error(f"Error al cargar gráfico: {e}")
                
                with info_col:
                    # Fundamentales
                    st.caption("📋 Datos Fundamentales")
                    with st.spinner('Cargando fundamentales...'):
                        fundies = get_stock_fundamentals(selected_ticker)
                    
                    if 'Error' not in fundies:
                        st.markdown(f"**{fundies.get('Short Name', selected_ticker)}**")
                        st.markdown(f"🏢 {fundies.get('Sector', 'N/D')} | {fundies.get('Industry', 'N/D')}")
                        
                        # Market Cap
                        mcap = fundies.get('Market Cap', 0)
                        if mcap > 1e12: mcap_s = f"${mcap/1e12:.1f}T"
                        elif mcap > 1e9: mcap_s = f"${mcap/1e9:.1f}B"
                        else: mcap_s = f"${mcap/1e6:.0f}M"
                        
                        fc1, fc2 = st.columns(2)
                        with fc1:
                            pe = fundies.get('P/E Ratio', 'N/D')
                            pe_s = f"{pe:.1f}x" if isinstance(pe, (int, float)) else pe
                            st.metric("P/E Ratio", pe_s)
                            st.metric("Market Cap", mcap_s)
                            beta = fundies.get('Beta', 'N/D')
                            beta_s = f"{beta:.2f}" if isinstance(beta, (int, float)) else beta
                            st.metric("Beta", beta_s)
                        with fc2:
                            rev = fundies.get('Revenue Growth', 'N/D')
                            rev_s = f"{rev*100:.1f}%" if isinstance(rev, (int, float)) else rev
                            st.metric("Rev. Growth", rev_s)
                            mg = fundies.get('Profit Margin', 'N/D')
                            mg_s = f"{mg*100:.1f}%" if isinstance(mg, (int, float)) else mg
                            st.metric("Profit Margin", mg_s)
                            st.metric("vs 52W High", f"{fundies.get('52W High Dist', 0)}%")
                    else:
                        st.warning("No se pudieron cargar los fundamentales.")
                
                # Análisis IA
                st.markdown("---")
                if groq_api_key:
                    if st.button(f"🧠 Análisis IA de {selected_ticker}", use_container_width=True, type='primary'):
                        with st.spinner('La IA está analizando...'):
                            tech_row = scan_df[scan_df['Ticker'] == selected_ticker].iloc[0].to_dict()
                            # Calcular Fibonacci para pasarle a la IA
                            try:
                                fib_hist = yf.Ticker(selected_ticker).history(period='3mo', interval='1d')
                                fib_h = fib_hist['High'].max()
                                fib_l = fib_hist['Low'].min()
                                fib_d = fib_h - fib_l
                                fib_data = {
                                    '0% (Max)': fib_h,
                                    '23.6%': fib_h - 0.236 * fib_d,
                                    '38.2%': fib_h - 0.382 * fib_d,
                                    '50%': fib_h - 0.5 * fib_d,
                                    '61.8%': fib_h - 0.618 * fib_d,
                                    '78.6%': fib_h - 0.786 * fib_d,
                                    '100% (Min)': fib_l
                                }
                            except:
                                fib_data = None
                            
                            if 'Error' not in fundies:
                                analysis = ai_stock_analysis(groq_api_key, selected_ticker, tech_row, fundies, fib_data)
                            else:
                                analysis = ai_stock_analysis(groq_api_key, selected_ticker, tech_row, {}, fib_data)
                        
                        st.markdown(f"### 🧠 Veredicto IA: {selected_ticker}")
                        
                        # Extraer veredicto para estilo visual
                        verdict = "ESP"
                        v_color = "#ffc107" # Amarillo por defecto
                        if "COMPRAR" in analysis.upper() or "🟢" in analysis:
                            verdict = "COMPRA RECOMENDADA"
                            v_color = "#28a745"
                        elif "EVITAR" in analysis.upper() or "🔴" in analysis:
                            verdict = "EVITAR / ALTO RIESGO"
                            v_color = "#dc3545"
                        elif "ESPERAR" in analysis.upper() or "🟡" in analysis:
                            verdict = "MANTENER / OBSERVAR"
                            v_color = "#ffc107"

                        st.markdown(f"""
                        <div style="background: rgba(0,0,0,0.3); border-left: 5px solid {v_color}; padding: 15px; border-radius: 10px; margin-bottom: 20px;">
                            <h2 style="color:{v_color}; margin:0; font-size: 1.2em;">{verdict}</h2>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        st.markdown(analysis.replace("$", "\$"))
                        
                        # --- EXTRACCIÓN DE SL/TP PARA AUTOMATIZACIÓN ---
                        ai_sl = 0.0
                        ai_tp = 0.0
                        try:
                            import re
                            # Buscar el patrón [SL: valor, TP: valor]
                            pattern = r'\[SL:\s*([\d\.]+),\s*TP:\s*([\d\.]+)\]'
                            match = re.search(pattern, analysis)
                            if match:
                                ai_sl = float(match.group(1))
                                ai_tp = float(match.group(2))
                        except:
                            pass

                        # --- LÓGICA DE GUARDADO CON CALLBACK ---
                        def save_trade_callback(t, p, s, v, r, sl_v, tp_v):
                            success = market_db.save_journal_entry(t, p, s, v, r, sl=sl_v, tp=tp_v)
                            if success:
                                st.session_state['save_success'] = f"Trade de {t} guardado con éxito!"
                            else:
                                st.session_state['save_error'] = "❌ Error al guardar en base de datos."

                        st.write("---")
                        
                        st.button(
                            "💾 Guardar en mi Diario de Operaciones", 
                            key=f"save_btn_{selected_ticker}", 
                            use_container_width=True,
                            on_click=save_trade_callback,
                            args=(selected_ticker, tech_row.get('Precio', 0), tech_row.get('Score', 0), verdict, analysis[:500], ai_sl, ai_tp)
                        )

                else:
                    st.warning("Configura GROQ_API_KEY en .env para habilitar el análisis IA.")

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
            
            # IA Context
            summary_cal = []
            for _, row in cal_df.head(15).iterrows():
                summary_cal.append(f"- {row.get('Fecha', '')} {row.get('Hora', '')} | {row.get('Evento', 'N/D')} | Act: {row.get('Actual', '-')} Prev: {row.get('Previsto', '-')}")
            st.session_state['calendar_text'] = "\n".join(summary_cal)

    # --- DEEP DIVE TAB ---
    with tab_deepdive:
        st.markdown("""
        <div style="padding:20px; border-radius:15px; background: linear-gradient(135deg, rgba(30,30,60,0.8), rgba(60,20,80,0.6)); border-left: 8px solid #a855f7; margin-bottom: 25px;">
            <h2 style="margin:0; color:#e0e0e0;">🔍 Stock Deep Dive</h2>
            <p style="margin:5px 0 0 0; color:#bbb; font-size:0.95em;">Análisis de inversión institucional. Introduce cualquier ticker para obtener un diagnóstico completo: financiero, técnico y estratégico.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Input de Ticker
        dd_col_input1, dd_col_input2, dd_col_input3 = st.columns([2, 1, 1])
        with dd_col_input1:
            dd_ticker = st.text_input(
                "🎯 Ticker del activo",
                value="",
                placeholder="Ej: AAPL, TSLA, MELI, NVDA...",
                help="Escribe el símbolo de cualquier acción listada en Yahoo Finance.",
                key="dd_ticker_input"
            ).upper().strip()
        with dd_col_input2:
            dd_include_options = st.checkbox("📊 Incluir Opciones", value=True, help="Analizar flujo de opciones (P/C Ratio, Muros)")
        with dd_col_input3:
            dd_run = st.button("🚀 Ejecutar Deep Dive", use_container_width=True, type='primary', key='dd_run_btn')
        
        if dd_run and dd_ticker:
            # --- FASE 1: RECOLECCIÓN DE DATOS ---
            with st.spinner(f'📡 Recopilando inteligencia de {dd_ticker}...'):
                dd_fin = get_deep_financials(dd_ticker)
            
            if not dd_fin.get('success'):
                st.error(f"❌ No se pudo obtener datos de '{dd_ticker}'. Verifica que el ticker sea correcto (Ej: AAPL, TSLA, MELI).")
            else:
                st.session_state['dd_data'] = dd_fin
                st.session_state['dd_ticker_active'] = dd_ticker
                
                with st.spinner('📊 Procesando análisis técnico...'):
                    dd_tech = get_deep_technical_analysis(dd_ticker)
                st.session_state['dd_tech'] = dd_tech
                
                dd_opts = None
                if dd_include_options:
                    with st.spinner('🎰 Leyendo flujo de opciones...'):
                        dd_opts = get_options_sentiment(dd_ticker)
                st.session_state['dd_opts'] = dd_opts
                
                # Health Score
                dd_health = calculate_financial_health_score(dd_fin)
                st.session_state['dd_health'] = dd_health
        
        # --- MOSTRAR RESULTADOS PERSISTENTES ---
        if 'dd_data' in st.session_state and st.session_state.get('dd_ticker_active'):
            dd_fin = st.session_state['dd_data']
            dd_tech = st.session_state.get('dd_tech')
            dd_opts = st.session_state.get('dd_opts')
            dd_health = st.session_state.get('dd_health')
            dd_ticker = st.session_state['dd_ticker_active']
            g = dd_fin['general']
            
            # --- HEADER: PERFIL DE LA EMPRESA ---
            st.markdown(f"""
            <div style="padding:20px; border-radius:12px; background: rgba(0,0,0,0.3); margin-bottom:15px;">
                <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap;">
                    <div>
                        <h2 style="margin:0; color:white;">{g['shortName']} ({dd_ticker})</h2>
                        <p style="margin:3px 0; color:#aaa;">{g['sector']} | {g['industry']} | {g['country']}</p>
                    </div>
                    <div style="text-align:right;">
                        <h2 style="margin:0; color:white;">{g['price']:.2f} USD</h2>
                        <p style="margin:3px 0; color:#aaa;">52W: {g['low_52']:.2f} — {g['high_52']:.2f} ({g['dist_52h']:+.1f}%)</p>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # --- FILA 1: HEALTH SCORE + RADAR ---
            st.markdown("### 🏥 Diagnóstico de Salud Financiera")
            
            h_col1, h_col2 = st.columns([1, 2])
            
            with h_col1:
                # Gauge de Health Score
                fig_health = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=dd_health['total'],
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': f"<b>{dd_health['grade']}</b><br><span style='font-size:0.7em;color:{dd_health['grade_color']}'>{dd_health['grade_text']}</span>", 'font': {'size': 16}},
                    gauge={
                        'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': '#444'},
                        'bar': {'color': dd_health['grade_color'], 'thickness': 0.3},
                        'bgcolor': 'rgba(0,0,0,0)',
                        'borderwidth': 0,
                        'steps': [
                            {'range': [0, 30], 'color': 'rgba(220,53,69,0.15)'},
                            {'range': [30, 50], 'color': 'rgba(253,126,20,0.15)'},
                            {'range': [50, 70], 'color': 'rgba(255,193,7,0.15)'},
                            {'range': [70, 85], 'color': 'rgba(40,167,69,0.15)'},
                            {'range': [85, 100], 'color': 'rgba(0,230,118,0.15)'},
                        ],
                        'threshold': {'line': {'color': 'white', 'width': 3}, 'thickness': 0.8, 'value': dd_health['total']}
                    }
                ))
                fig_health.update_layout(height=280, margin=dict(l=20, r=20, t=60, b=10), paper_bgcolor='rgba(0,0,0,0)', font={'color': 'white'})
                st.plotly_chart(fig_health, use_container_width=True)
            
            with h_col2:
                # Radar Chart de Dimensiones
                categories = list(dd_health['scores'].keys())
                max_vals = [20, 20, 20, 20, 10, 10]
                normalized = [(dd_health['scores'][c] / m) * 100 for c, m in zip(categories, max_vals)]
                
                fig_radar = go.Figure()
                fig_radar.add_trace(go.Scatterpolar(
                    r=normalized + [normalized[0]],
                    theta=categories + [categories[0]],
                    fill='toself',
                    fillcolor='rgba(168,85,247,0.2)',
                    line=dict(color='#a855f7', width=2),
                    name='Score'
                ))
                fig_radar.update_layout(
                    polar=dict(
                        radialaxis=dict(visible=True, range=[0, 100], tickfont=dict(size=9, color='#666'), gridcolor='rgba(255,255,255,0.1)'),
                        angularaxis=dict(tickfont=dict(size=11, color='#ccc'), gridcolor='rgba(255,255,255,0.1)'),
                        bgcolor='rgba(0,0,0,0)',
                    ),
                    showlegend=False,
                    height=300,
                    margin=dict(l=60, r=60, t=30, b=30),
                    paper_bgcolor='rgba(0,0,0,0)',
                )
                st.plotly_chart(fig_radar, use_container_width=True)
            
            # Detalle del Score
            with st.expander("📋 Detalle del Score por Dimensión", expanded=False):
                for dim, pts in dd_health['scores'].items():
                    max_p = 20 if dim in ['Valuación', 'Rentabilidad', 'Crecimiento', 'Solvencia'] else 10
                    pct = (pts / max_p) * 100
                    bar_color = '#28a745' if pct >= 70 else '#ffc107' if pct >= 40 else '#dc3545'
                    notes = ' | '.join(dd_health['details'].get(dim, [])) or 'Sin datos suficientes'
                    st.markdown(f"""
                    <div style="margin-bottom:8px;">
                        <div style="display:flex; justify-content:space-between; margin-bottom:2px;">
                            <span style="font-weight:bold; color:#eee;">{dim}</span>
                            <span style="color:{bar_color}; font-weight:bold;">{pts}/{max_p}</span>
                        </div>
                        <div style="background:rgba(255,255,255,0.1); border-radius:5px; height:8px; overflow:hidden;">
                            <div style="background:{bar_color}; height:100%; width:{pct}%; border-radius:5px; transition: width 0.5s;"></div>
                        </div>
                        <p style="margin:2px 0 0 0; font-size:0.75em; color:#999;">{notes}</p>
                    </div>
                    """, unsafe_allow_html=True)
            
            st.markdown("---")
            
            # --- FILA 2: MÉTRICAS CLAVE ---
            st.markdown("### 📊 Métricas Clave")
            
            mk_col1, mk_col2, mk_col3, mk_col4, mk_col5, mk_col6 = st.columns(6)
            
            # Helpers
            def fmt_v(val, pct=False, suffix=''):
                if val is None: return 'N/D'
                if pct: return f"{val*100:.1f}%"
                return f"{val:.2f}{suffix}"
            def fmt_b(val):
                if val is None or val == 0: return 'N/D'
                if abs(val) >= 1e12: return f"{val/1e12:.1f}T"
                if abs(val) >= 1e9: return f"{val/1e9:.1f}B"
                if abs(val) >= 1e6: return f"{val/1e6:.0f}M"
                return f"{val:,.0f}"
            
            pe_val = dd_fin['valuation']['forwardPE'] or dd_fin['valuation']['trailingPE']
            mk_col1.metric("P/E Ratio", fmt_v(pe_val, suffix='x'))
            mk_col2.metric("PEG Ratio", fmt_v(dd_fin['valuation']['pegRatio']))
            mk_col3.metric("Market Cap", fmt_b(g['marketCap']))
            mk_col4.metric("ROE", fmt_v(dd_fin['profitability']['returnOnEquity'], pct=True))
            mk_col5.metric("Margen Neto", fmt_v(dd_fin['profitability']['profitMargin'], pct=True))
            mk_col6.metric("Crec. Ingresos", fmt_v(dd_fin['growth']['revenueGrowth'], pct=True))
            
            mk2_col1, mk2_col2, mk2_col3, mk2_col4, mk2_col5, mk2_col6 = st.columns(6)
            mk2_col1.metric("Deuda/Equity", fmt_v(dd_fin['solvency']['debtToEquity']))
            mk2_col2.metric("Current Ratio", fmt_v(dd_fin['solvency']['currentRatio']))
            mk2_col3.metric("Free Cash Flow", fmt_b(dd_fin['solvency']['freeCashflow']))
            mk2_col4.metric("Beta", fmt_v(dd_fin['risk']['beta']))
            mk2_col5.metric("Div. Yield", fmt_v(dd_fin['dividends']['dividendYield'], pct=True))
            mk2_col6.metric("EPS Forward", fmt_v(dd_fin['analyst']['forwardEps']))
            
            st.markdown("---")
            
            # --- FILA 3: GRÁFICOS (Income + Analyst Targets + Chart Técnico) ---
            chart_col1, chart_col2 = st.columns([1, 1])
            
            with chart_col1:
                st.markdown("#### 📈 Evolución de Resultados (Anual)")
                inc_hist = dd_fin.get('income_history', [])
                if inc_hist:
                    inc_df = pd.DataFrame(inc_hist).sort_values('period')
                    fig_inc = go.Figure()
                    fig_inc.add_trace(go.Bar(
                        x=inc_df['period'], y=inc_df['totalRevenue'],
                        name='Ingresos', marker_color='rgba(99, 102, 241, 0.7)',
                        text=[fmt_b(v) for v in inc_df['totalRevenue']], textposition='outside'
                    ))
                    fig_inc.add_trace(go.Bar(
                        x=inc_df['period'], y=inc_df['netIncome'],
                        name='Beneficio Neto', marker_color='rgba(16, 185, 129, 0.7)',
                        text=[fmt_b(v) for v in inc_df['netIncome']], textposition='outside'
                    ))
                    fig_inc.update_layout(
                        height=350, barmode='group',
                        template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                        margin=dict(l=10, r=10, t=10, b=10),
                        legend=dict(orientation='h', yanchor='bottom', y=1.02),
                        yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)'),
                    )
                    st.plotly_chart(fig_inc, use_container_width=True)
                else:
                    st.info("Sin historial de ingresos disponible para este activo.")
            
            with chart_col2:
                st.markdown("#### 🎯 Precio Objetivo Analistas")
                a = dd_fin['analyst']
                if a.get('targetMeanPrice') and a.get('targetLowPrice') and a.get('targetHighPrice'):
                    current_price = g['price']
                    upside_mean = ((a['targetMeanPrice'] / current_price) - 1) * 100
                    
                    fig_target = go.Figure()
                    fig_target.add_trace(go.Indicator(
                        mode="number+delta",
                        value=a['targetMeanPrice'],
                        delta={'reference': current_price, 'relative': True, 'valueformat': '.1%'},
                        title={'text': f"Target Medio ({a['numberOfAnalystOpinions']} analistas)", 'font': {'size': 14}},
                        domain={'x': [0, 1], 'y': [0.6, 1]},
                        number={'font': {'size': 36}}
                    ))
                    
                    # Bar de rango de targets
                    fig_target.add_trace(go.Bar(
                        x=[a['targetLowPrice']],
                        y=['Target'],
                        orientation='h',
                        marker_color='rgba(220,53,69,0.5)',
                        name=f"Min: {a['targetLowPrice']:.2f}",
                        width=0.4,
                    ))
                    fig_target.add_trace(go.Bar(
                        x=[a['targetMeanPrice'] - a['targetLowPrice']],
                        y=['Target'],
                        orientation='h',
                        marker_color='rgba(255,193,7,0.6)',
                        name=f"Media: {a['targetMeanPrice']:.2f}",
                        width=0.4,
                    ))
                    fig_target.add_trace(go.Bar(
                        x=[a['targetHighPrice'] - a['targetMeanPrice']],
                        y=['Target'],
                        orientation='h',
                        marker_color='rgba(40,167,69,0.6)',
                        name=f"Max: {a['targetHighPrice']:.2f}",
                        width=0.4,
                    ))
                    
                    # Línea de precio actual
                    fig_target.add_vline(
                        x=current_price, line_dash='dash', line_color='white', line_width=2,
                        annotation_text=f"Actual: {current_price:.2f}", annotation_position='top',
                        annotation_font=dict(color='white', size=10)
                    )
                    
                    fig_target.update_layout(
                        height=350, barmode='stack',
                        template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                        margin=dict(l=10, r=10, t=10, b=10),
                        legend=dict(orientation='h', yanchor='bottom', y=-0.15, font=dict(size=10)),
                        showlegend=True,
                        yaxis=dict(showgrid=False),
                        xaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)', title='Precio (USD)'),
                    )
                    st.plotly_chart(fig_target, use_container_width=True)
                else:
                    rec_key = a.get('recommendationKey', 'N/D')
                    st.info(f"Consenso de analistas: **{rec_key}**. Sin datos detallados de target para este activo.")
            
            st.markdown("---")
            
            # --- FILA 4: GRÁFICO TÉCNICO AVANZADO ---
            if dd_tech:
                st.markdown("### 📉 Análisis Técnico Avanzado")
                
                # Performance Badges
                perf_cols = st.columns(5)
                perf_data = [
                    ('1 Día', dd_tech['chg_1d']),
                    ('5 Días', dd_tech['chg_5d']),
                    ('1 Mes', dd_tech['chg_1m']),
                    ('3 Meses', dd_tech['chg_3m']),
                    ('6 Meses', dd_tech['chg_6m']),
                ]
                for col, (label, val) in zip(perf_cols, perf_data):
                    color = '#28a745' if val >= 0 else '#dc3545'
                    col.markdown(f"""
                    <div style="text-align:center; padding:8px; border-radius:8px; background:rgba(0,0,0,0.2); border: 1px solid {color};">
                        <p style="margin:0; font-size:0.75em; color:#aaa;">{label}</p>
                        <p style="margin:0; font-size:1.2em; font-weight:bold; color:{color};">{val:+.2f}%</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Indicadores técnicos
                ti_col1, ti_col2, ti_col3, ti_col4, ti_col5 = st.columns(5)
                
                rsi_color = '#dc3545' if dd_tech['rsi'] > 70 else '#28a745' if dd_tech['rsi'] < 30 else '#ffc107'
                ti_col1.markdown(f"<div style='text-align:center;'><span style='color:#aaa; font-size:0.8em;'>RSI(14)</span><br><span style='color:{rsi_color}; font-weight:bold; font-size:1.3em;'>{dd_tech['rsi']:.1f}</span></div>", unsafe_allow_html=True)
                
                trend_color = '#28a745' if 'ALCISTA' in dd_tech['trend_status'] else '#dc3545' if 'BAJISTA' in dd_tech['trend_status'] else '#ffc107'
                ti_col2.markdown(f"<div style='text-align:center;'><span style='color:#aaa; font-size:0.8em;'>Tendencia</span><br><span style='color:{trend_color}; font-weight:bold; font-size:0.9em;'>{dd_tech['trend_status']}</span></div>", unsafe_allow_html=True)
                
                adx_text = 'Fuerte' if dd_tech['adx'] > 25 else 'Débil'
                ti_col3.markdown(f"<div style='text-align:center;'><span style='color:#aaa; font-size:0.8em;'>ADX ({adx_text})</span><br><span style='color:white; font-weight:bold; font-size:1.3em;'>{dd_tech['adx']:.1f}</span></div>", unsafe_allow_html=True)
                
                macd_color = '#28a745' if dd_tech['macd_hist'] > 0 else '#dc3545'
                ti_col4.markdown(f"<div style='text-align:center;'><span style='color:#aaa; font-size:0.8em;'>MACD Hist</span><br><span style='color:{macd_color}; font-weight:bold; font-size:1.3em;'>{dd_tech['macd_hist']:.4f}</span></div>", unsafe_allow_html=True)
                
                vr_color = '#28a745' if dd_tech['vol_ratio'] > 1.2 else '#ffc107'
                ti_col5.markdown(f"<div style='text-align:center;'><span style='color:#aaa; font-size:0.8em;'>Vol Ratio</span><br><span style='color:{vr_color}; font-weight:bold; font-size:1.3em;'>{dd_tech['vol_ratio']:.2f}x</span></div>", unsafe_allow_html=True)
                
                # Gráfico Candlestick con Fibonacci + EMAs
                chart_data = dd_tech['daily_data']
                fig_dd = go.Figure()
                
                fig_dd.add_trace(go.Candlestick(
                    x=chart_data.index,
                    open=chart_data['Open'], high=chart_data['High'],
                    low=chart_data['Low'], close=chart_data['Close'],
                    name=dd_ticker, increasing_line_color='#28a745', decreasing_line_color='#dc3545'
                ))
                
                # EMAs
                ema20_series = chart_data['Close'].ewm(span=20).mean()
                ema50_series = chart_data['Close'].ewm(span=50).mean()
                fig_dd.add_trace(go.Scatter(x=chart_data.index, y=ema20_series, name='EMA 20', line=dict(color='#00b4d8', width=1)))
                fig_dd.add_trace(go.Scatter(x=chart_data.index, y=ema50_series, name='EMA 50', line=dict(color='#ff6b35', width=1)))
                
                # Bollinger Bands
                bb_mid_s = chart_data['Close'].rolling(20).mean()
                bb_std_s = chart_data['Close'].rolling(20).std()
                fig_dd.add_trace(go.Scatter(x=chart_data.index, y=bb_mid_s + bb_std_s*2, name='BB Upper', line=dict(color='rgba(173,216,230,0.3)', width=0.8, dash='dot'), showlegend=False))
                fig_dd.add_trace(go.Scatter(x=chart_data.index, y=bb_mid_s - bb_std_s*2, name='BB Lower', line=dict(color='rgba(173,216,230,0.3)', width=0.8, dash='dot'), fill='tonexty', fillcolor='rgba(173,216,230,0.04)', showlegend=False))
                
                # Fibonacci Levels
                fib_colors = [
                    ('23.6%', 'rgba(255,107,107,0.6)'),
                    ('38.2%', 'rgba(255,169,77,0.6)'),
                    ('50%', 'rgba(255,212,59,0.7)'),
                    ('61.8%', 'rgba(105,219,124,0.7)'),
                    ('78.6%', 'rgba(56,217,169,0.6)'),
                ]
                for fib_label, fib_color in fib_colors:
                    fib_price = dd_tech['fib_levels'].get(fib_label)
                    if fib_price:
                        fig_dd.add_hline(
                            y=fib_price, line_dash='dash', line_color=fib_color, line_width=0.8,
                            annotation_text=f"{fib_label} {fib_price:.2f}",
                            annotation_position='right',
                            annotation_font=dict(size=8, color=fib_color),
                            annotation_bgcolor='rgba(0,0,0,0.5)'
                        )
                
                # Volumen
                vol_colors = ['#28a745' if c >= o else '#dc3545' for c, o in zip(chart_data['Close'], chart_data['Open'])]
                fig_dd.add_trace(go.Bar(x=chart_data.index, y=chart_data['Volume'], name='Volumen', marker_color=vol_colors, opacity=0.15, yaxis='y2'))
                
                fig_dd.update_layout(
                    height=550,
                    template='plotly_dark',
                    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                    xaxis_rangeslider_visible=False,
                    legend=dict(orientation='h', yanchor='bottom', y=1.02, font=dict(size=10)),
                    margin=dict(l=10, r=10, t=30, b=10),
                    yaxis2=dict(overlaying='y', side='right', showgrid=False, showticklabels=False, range=[0, chart_data['Volume'].max() * 4])
                )
                st.plotly_chart(fig_dd, use_container_width=True)
            
            st.markdown("---")
            
            # --- FILA 5: OPCIONES + BALANCE ---
            extra_col1, extra_col2 = st.columns(2)
            
            with extra_col1:
                if dd_opts:
                    st.markdown("#### 🎰 Flujo de Opciones (Smart Money)")
                    st.markdown(f"""
                    <div style="padding:15px; border-radius:10px; border: 1px solid {dd_opts['color']}; background: rgba(0,0,0,0.2);">
                        <div style="display:flex; justify-content:space-between; align-items:center;">
                            <h4 style="margin:0; color:{dd_opts['color']};">{dd_opts['sent']}</h4>
                            <span style="font-size:0.8em; color:#fff; background:{dd_opts['color']}; padding:2px 8px; border-radius:10px;">P/C: {dd_opts['ratio']:.2f}</span>
                        </div>
                        <div style="margin-top:10px; padding:10px; background:rgba(255,255,255,0.05); border-radius:5px;">
                            <div style="display:flex; justify-content:space-between; margin-bottom:5px;">
                                <span style="color:#ff6b6b; font-weight:bold;">🧱 Call Wall:</span>
                                <span style="color:white;">{dd_opts['call_wall']:,.0f}</span>
                            </div>
                            <div style="display:flex; justify-content:space-between;">
                                <span style="color:#51cf66; font-weight:bold;">🛡️ Put Wall:</span>
                                <span style="color:white;">{dd_opts['put_wall']:,.0f}</span>
                            </div>
                        </div>
                        <div style="display:flex; justify-content:space-between; font-size:0.75em; margin-top:8px; color:#999;">
                            <span>Vol. Calls: {int(dd_opts['vol_calls']):,}</span>
                            <span>Vol. Puts: {int(dd_opts['vol_puts']):,}</span>
                            <span>Exp: {dd_opts['exp']}</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.info("Sin datos de opciones para este activo.")
            
            with extra_col2:
                st.markdown("#### 🏦 Estructura de Balance")
                bal = dd_fin.get('balance', {})
                if bal:
                    assets = bal.get('totalAssets', 0)
                    liab = bal.get('totalLiabilities', 0)
                    equity = bal.get('totalEquity', 0)
                    cash = bal.get('cash', 0)
                    debt = bal.get('totalDebt', 0)
                    
                    if assets > 0:
                        fig_bal = go.Figure()
                        fig_bal.add_trace(go.Bar(
                            x=['Activos', 'Pasivos', 'Patrimonio', 'Caja', 'Deuda'],
                            y=[assets, liab, equity, cash, debt],
                            marker_color=['#6366f1', '#f87171', '#22c55e', '#06b6d4', '#f59e0b'],
                            text=[fmt_b(v) for v in [assets, liab, equity, cash, debt]],
                            textposition='outside',
                        ))
                        fig_bal.update_layout(
                            height=280,
                            template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                            margin=dict(l=10, r=10, t=10, b=10),
                            yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)'),
                        )
                        st.plotly_chart(fig_bal, use_container_width=True)
                    else:
                        st.info("Sin datos de balance disponibles.")
                else:
                    st.info("Sin datos de balance disponibles.")
            
            st.markdown("---")
            
            # --- FILA 6: DESCRIPCIÓN + DATOS ADICIONALES ---
            with st.expander("📖 Descripción de la Empresa", expanded=False):
                st.write(g.get('summary', 'Sin descripción disponible.'))
                if g.get('website'):
                    st.markdown(f"🌐 [{g['website']}]({g['website']})")
                if g.get('employees') and g['employees'] > 0:
                    st.markdown(f"👥 Empleados: **{g['employees']:,}**")
            
            st.markdown("---")
            
            # --- FILA 7: ANÁLISIS IA EXHAUSTIVO ---
            st.markdown("### 🧠 Informe de Inversión IA")
            
            if groq_api_key:
                if st.button(f"🧠 Generar Informe Institucional de {dd_ticker}", use_container_width=True, type='primary', key='dd_ai_btn'):
                    with st.spinner(f'🔬 La IA está analizando {dd_ticker} en profundidad...'):
                        dd_analysis = ai_deep_dive_analysis(groq_api_key, dd_ticker, dd_fin, dd_tech, dd_opts)
                    
                    st.session_state['dd_analysis'] = dd_analysis
                
                if 'dd_analysis' in st.session_state:
                    dd_analysis = st.session_state['dd_analysis']
                    
                    # Extraer veredicto para visual
                    dd_verdict = "ANÁLISIS"
                    dd_v_color = "#a855f7"
                    analysis_upper = dd_analysis.upper()
                    if "COMPRA FUERTE" in analysis_upper: dd_verdict = "🟢 COMPRA FUERTE"; dd_v_color = "#00e676"
                    elif "COMPRA" in analysis_upper and "NO COMPRA" not in analysis_upper: dd_verdict = "🟢 COMPRA"; dd_v_color = "#28a745"
                    elif "VENDER FUERTE" in analysis_upper: dd_verdict = "🔴 VENDER FUERTE"; dd_v_color = "#ff1744"
                    elif "VENDER" in analysis_upper: dd_verdict = "🔴 VENDER"; dd_v_color = "#dc3545"
                    elif "MANTENER" in analysis_upper: dd_verdict = "🟡 MANTENER"; dd_v_color = "#ffc107"
                    
                    st.markdown(f"""
                    <div style="background: linear-gradient(135deg, rgba(0,0,0,0.4), rgba(30,30,60,0.4)); border-left: 6px solid {dd_v_color}; padding: 18px; border-radius: 12px; margin-bottom: 20px;">
                        <h2 style="color:{dd_v_color}; margin:0; font-size: 1.3em;">{dd_verdict}</h2>
                        <p style="margin:5px 0 0 0; color:#999; font-size:0.85em;">Informe generado por IA basado en {len(dd_fin.get('income_history', []))} años de datos financieros, análisis técnico de 6 meses y flujo de opciones.</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown(dd_analysis.replace("$", "\\$"))
                    
                    # Extracción de SL/TP
                    dd_sl = 0.0
                    dd_tp = 0.0
                    try:
                        pattern = r'\[SL:\s*([\d\.]+),\s*TP:\s*([\d\.]+)\]'
                        match = re.search(pattern, dd_analysis)
                        if match:
                            dd_sl = float(match.group(1))
                            dd_tp = float(match.group(2))
                    except:
                        pass
                    
                    # Botones de acción
                    st.markdown("---")
                    dd_act_col1, dd_act_col2 = st.columns(2)
                    
                    with dd_act_col1:
                        def dd_save_callback(t, p, s, v, r, sl_v, tp_v):
                            success = market_db.save_journal_entry(t, p, s, v, r, sl=sl_v, tp=tp_v)
                            if success:
                                st.session_state['save_success'] = f"Trade de {t} guardado con éxito!"
                            else:
                                st.session_state['save_error'] = "Error al guardar."
                        
                        st.button(
                            "💾 Guardar en Diario de Operaciones",
                            key=f"dd_save_{dd_ticker}",
                            use_container_width=True,
                            on_click=dd_save_callback,
                            args=(dd_ticker, g['price'], dd_health['total'], dd_verdict, dd_analysis[:500], dd_sl, dd_tp)
                        )
                    
                    with dd_act_col2:
                        if dd_sl > 0 and dd_tp > 0:
                            if st.button("🧮 Enviar a Calculadora de Riesgo", key='dd_calc_btn', use_container_width=True):
                                st.session_state['calc_entry'] = float(g['price'])
                                st.session_state['calc_stop'] = float(dd_sl)
                                st.session_state['calc_tp'] = float(dd_tp)
                                st.rerun()
            else:
                st.warning("⚠️ Configura GROQ_API_KEY en .env para habilitar el análisis IA profundo.")

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
                            st.markdown(analysis.replace("$", "\$"))
                            
                    if btn_briefing:
                        with st.spinner("Leyendo noticias y cruzando datos..."):
                            news = get_market_news(ticker)
                            cal_txt = st.session_state.get('calendar_text', "Sin eventos macro reportados.")
                            briefing = get_pre_market_briefing(groq_api_key, ctx, news, cal_txt)
                            st.success("### 🌅 Briefing Pre-Mercado (Macro + Técnico)")
                            st.markdown(briefing.replace("$", "\$"))
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
        st.subheader("📚 Centro de Registro y Bitácora")
        
        # Dos sub-pestañas internas o secciones
        hist_sel = st.radio("Ver registros de:", ["📅 Diario de Operaciones (Scanner)", "🤖 Bitácora de Predicciones AI"], horizontal=True)
        
        if hist_sel == "📅 Diario de Operaciones (Scanner)":
            st.markdown("#### 🗒️ Operaciones Guardadas (SQLite Local)")
            
            # Botón para forzar actualización de precios (evita lentitud al cargar)
            refresh_prices = st.button("🔄 Actualizar Precios Actuales")
            
            try:
                journal_data = market_db.get_journal_entries()
            except Exception as e:
                st.error(f"Error al leer la base de datos: {e}")
                journal_data = pd.DataFrame()
            
            if not journal_data.empty:
                for idx, row in journal_data.iterrows():
                    with st.expander(f"📌 {row['ticker']} | {row['entry_date']} | {row['verdict']}"):
                        jcol1, jcol2, jcol3 = st.columns([1, 1, 1])
                        
                        cur_p = "N/D (Refresh)"
                        perf = 0
                        p_color = "#bbb"
                        
                        # Solo actualizar precios si el usuario lo pide
                        if refresh_prices:
                            try:
                                cur_p_val = yf.Ticker(row['ticker']).history(period='1d')['Close'].iloc[-1]
                                cur_p = cur_p_val
                                perf = ((cur_p / row['entry_price']) - 1) * 100
                                p_color = "#28a745" if perf >= 0 else "#dc3545"
                            except:
                                cur_p = "Error Conexión"
                        
                        jcol1.metric("Precio Entrada", f"${row['entry_price']:.2f}")
                        
                        if isinstance(cur_p, (int, float)):
                            jcol2.metric("Precio Actual", f"${cur_p:.2f}", delta=f"{perf:.2f}%")
                        else:
                            jcol2.metric("Precio Actual", str(cur_p))
                            
                        # Mostrar niveles de estrategia si existen (manejo de None para trades viejos)
                        sl_v = row.get('sl_price') if row.get('sl_price') is not None else 0.0
                        tp_v = row.get('tp_price') if row.get('tp_price') is not None else 0.0
                        
                        if float(sl_v) > 0 and float(tp_v) > 0:
                            entry_p = float(row['entry_price'])
                            risk = entry_p - float(sl_v)
                            reward = float(tp_v) - entry_p
                            rr = reward / risk if risk != 0 else 0
                            jcol3.metric("Ratio R/R (IA)", f"1:{rr:.1f}")
                        else:
                            jcol3.metric("Score Original", f"{row['score']}/100")

                        
                        if sl_v > 0 or tp_v > 0:
                            st.markdown(f"**🛡️ Estrategia Sugerida:** SL: `${sl_v:.2f}` | TP: `${tp_v:.2f}`")

                        
                        st.markdown("**🧠 Razonamiento IA:**")
                        st.caption(row['reasoning'])
                        
                        # --- BOTONES DE ACCIÓN ---
                        hcol1, hcol2, hcol3, hcol4 = st.columns(4)
                        
                        with hcol1:
                            if st.button(f"🎯 Entrada", key=f"load_{row['id']}", use_container_width=True):
                                st.session_state['calc_entry'] = float(row['entry_price'])
                                if sl_v > 0: st.session_state['calc_stop'] = float(sl_v)
                                if tp_v > 0: st.session_state['calc_tp'] = float(tp_v)
                                st.rerun()
                        
                        with hcol2:
                            if st.button(f"🧠 Estrategia", key=f"load_strat_{row['id']}", use_container_width=True):
                                st.session_state['calc_entry'] = float(row['entry_price'])
                                st.session_state['calc_stop'] = float(sl_v) if sl_v > 0 else float(row['entry_price']) * 0.95
                                st.session_state['calc_tp'] = float(tp_v) if tp_v > 0 else float(row['entry_price']) * 1.10
                                st.rerun()



                        
                        with hcol3:
                            if st.button(f"📈 Actual", key=f"load_curr_{row['id']}", use_container_width=True):
                                with st.spinner("Buscando..."):
                                    try:
                                        actual_p = yf.Ticker(row['ticker']).history(period='1d')['Close'].iloc[-1]
                                        st.session_state['calc_entry'] = float(actual_p)
                                        st.session_state['calc_stop'] = float(sl_v) if sl_v > 0 else float(actual_p) * 0.95
                                        st.session_state['calc_tp'] = float(tp_v) if tp_v > 0 else float(actual_p) * 1.10
                                        st.rerun()
                                    except: st.error("Error")



                        
                        with hcol4:
                            if st.button(f"🗑️ Eliminar", key=f"del_{row['id']}", use_container_width=True):
                                if market_db.delete_journal_entry(row['id']):
                                    st.success("Removido")
                                    time.sleep(1)
                                    st.rerun()



            else:
                st.info("Tu bitácora está vacía. Guarda operaciones desde el Scanner.")
        
        else:
            st.markdown(f"#### 🤖 Registro de Predicciones: {ticker}")
            history_df = market_db.get_predictions(ticker, limit=30)
            if not history_df.empty:
                history_df['Dirección'] = history_df['prediction_value'].map({1: 'Subida 🟢', 0: 'Bajada 🔴', -1: 'Neutral 🟡'})
                st.dataframe(history_df[['execution_date', 'prediction_date', 'Dirección', 'prob_up', 'prob_down']], use_container_width=True, hide_index=True)
            else:
                st.info("No hay registros de predicciones para este activo aún.")



    st.markdown("---")
    st.caption("⚠️ Advertencia: Plataforma de análisis algorítmico. No constituye asesoría financiera directa.")

if __name__ == '__main__':
    main()
