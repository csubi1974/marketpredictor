import pandas as pd
import numpy as np
import re
import io
from dotenv import load_dotenv
import yfinance as yf
import datetime
from datetime import datetime as dt, date as d_type, timedelta
import streamlit as st
import os
import plotly.graph_objects as go
import time
import requests
import json
import sqlite3
from sqlite3 import Error
from scipy.stats import norm
from concurrent.futures import ThreadPoolExecutor
from datetime import timezone as tz, timedelta as td, time as t_time
import asyncio


try:
    from dotenv import load_dotenv
    load_dotenv() # Carga variables del archivo .env
except ImportError:
    pass

# --- FIX: Event Loop for Windows/Streamlit/yfinance ---
if os.name == 'nt':
    try:
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    except:
        pass

# Lista de las 20 acciones más importantes (puedes modificar esta lista según tus preferencias)
TOP_20_STOCKS = {
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

def get_weinstein_stage(df):
    """Determina la etapa de Weinstein (1-4) basada en la SMA 150 y su pendiente."""
    if len(df) < 150:
        return "N/D"
    
    # Usamos 150 periodos como aproximación de 30 semanas (Weinstein estándar)
    sma_150 = df['Close'].rolling(window=150).mean()
    if sma_150.isna().all():
        return "N/D"
        
    curr_price = df['Close'].iloc[-1]
    curr_sma = sma_150.iloc[-1]
    
    # Pendiente basada en los últimos 20 días (1 mes de trading)
    prev_sma = sma_150.iloc[-21] if len(sma_150) > 21 else sma_150.iloc[0]
    slope = (curr_sma - prev_sma) / prev_sma
    dist = (curr_price - curr_sma) / curr_sma
    
    # Definición de etapas
    # Etapa 2: Avanzando (Precio > SMA + SMA subiendo)
    if dist > 0.02 and slope > 0.005:
        try:
            # Validar volumen: volumen de hoy (o media reciente) vs media de 20 días
            recent_vol = df['Volume'].iloc[-1]
            avg_vol_20 = df['Volume'].tail(20).mean()
            if avg_vol_20 > 0 and (recent_vol / avg_vol_20) > 2.0:
                return "Etapa 2 (Valid. Vol)"
        except:
            pass
        return "Etapa 2 (Alcista)"
    # Etapa 4: Declive (Precio < SMA + SMA bajando)
    elif dist < -0.02 and slope < -0.005:
        return "Etapa 4 (Bajista / Declive)"
    # Etapa 1 o 3 (Consolidación / Techo)
    else:
        # Si el precio está cerca de la media o la media está plana
        if abs(slope) < 0.005:
            if curr_price > curr_sma: return "Etapa 3 (Distribución / Techo)"
            else: return "Etapa 1 (Acumulación / Suelo)"
        
        # Casos de transición
        if slope > 0: return "Etapa 1-2 (Transición Alcista)"
        else: return "Etapa 3-4 (Transición Bajista)"



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
        print(f"Error Opciones o Intraday: {e}")
        return None

def get_llm_analysis(api_key, context_data):
    """Consulta a Groq (Llama 3) para un análisis táctico."""
    if not api_key: return "⚠️ Por favor ingresa tu API Key de Groq en el sidebar."
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    prompt = f"""
    Actúa como un Estratega Jefe de Inversiones de StratEdge Portfolio. Tu objetivo es analizar datos para horizontes de Corto, Mediano y Largo Plazo.
    
    DATOS DEL MERCADO:
    - Tendencia de Activo: {context_data.get('sniper_status', 'N/A')}
    - Sentimiento Opciones: {context_data.get('options_sent', 'N/A')}
    - Niveles Clave (Call/Put Wall): {context_data.get('call_wall', 'N/A')} / {context_data.get('put_wall', 'N/A')}
    
    TU MISIÓN:
    1. PANORAMA MULTI-HORIZONTE: Define la situación técnica en el Corto (días), Mediano (semanas) y Largo Plazo (meses).
    2. ESTRATEGIA RECOMENDADA: ¿Es momento de Acumular, Mantener, Proteger o Liquidar?
    3. FACTOR DE RIESGO: ¿Cuál es el principal obstáculo para esta tesis de inversión?
    
    REGLAS CRÍTICAS:
    - ZERO HALLUCINATION: No inventes datos. Cíñete a los proporcionados.
    - Si un dato aparece como 'N/A', no hagas suposiciones sobre él.
    
    Responde en formato Markdown, con estilo profesional y estratégico. Sé directo.
    """
    
    payload = {
        "model": "llama-3.1-8b-instant",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.1
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
                            headlines.append(f"- {title} ({n.get('publisher', 'Intelligence Hub')})")
                            seen_titles.add(title)
                        if len(headlines) >= 8: break
            except:
                continue
                
        return "\n".join(headlines) if headlines else "No se detectaron noticias urgentes en los canales de StratEdge Intelligence ahora mismo."
    except Exception as e:
        return f"Nota: Servicio de noticias temporalmente limitado. Enfoque en análisis técnico. ({str(e)})"

@st.cache_data(ttl=60)
def get_ticker_snapshot(ticker):
    """Obtiene un resumen rápido del precio actual y cambio del día."""
    try:
        tk = yf.Ticker(ticker)
        data = tk.history(period='2d')
        if len(data) < 1: return f"No se hallaron datos para {ticker}."
        
        last_close = data['Close'].iloc[-1]
        prev_close = data['Close'].iloc[-2] if len(data) > 1 else data['Open'].iloc[-1]
        change = last_close - prev_close
        pct = (change / prev_close) * 100
        
        info = tk.info
        name = info.get('shortName', ticker)
        
        return {
            "Ticker": ticker,
            "Nombre": name,
            "Precio": f"${last_close:.2f}",
            "Cambio": f"{change:+.2f} ({pct:+.2f}%)",
            "Rango Hoy": f"${data['Low'].iloc[-1]:.2f} - ${data['High'].iloc[-1]:.2f}",
            "Volumen": f"{data['Volume'].iloc[-1]:,}"
        }
    except Exception as e:
        return f"Error obteniendo snapshot de {ticker}: {e}"

@st.cache_data(ttl=3600)
def get_platform_info():
    """Lee el manual/README del proyecto para explicar funciones al usuario."""
    try:
        path = "README.md"
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return f.read()
        return "Manual no disponible. StratEdge Portfolio ofrece Strategy Hub, Asset Scanner, Wheel Strategy y Strategic Analysis."
    except:
        return "Error al leer el manual."

@st.cache_data(ttl=1800)
def get_economic_calendar(date_str=None, days=7):
    """Obtiene el calendario para una fecha o un rango de días (semana completa por defecto)."""
    try:
        ny_now = get_ny_time()
        # Eliminar info de zona horaria para cálculos de timedelta si date_str es None
        if not date_str:
            start_date = ny_now.replace(tzinfo=None)
        else:
            start_date = dt.strptime(date_str, "%Y-%m-%d")
            
        all_results = []
        for i in range(days):
            current_day = (start_date + timedelta(days=i)).strftime("%Y-%m-%d")
            try:
                # Intentamos obtener datos de Yahoo Finance
                url = f"https://finance.yahoo.com/calendar/economic?day={current_day}&region=US"
                headers = {'User-Agent': 'Mozilla/5.0'}
                response = requests.get(url, headers=headers, timeout=5)
                if response.status_code == 200:
                    tables = pd.read_html(io.StringIO(response.text))
                    if tables:
                        scrape_df = tables[0]
                        # Limpiar nombres de columnas (quitar espacios especiales)
                        scrape_df.columns = [str(c).replace('\xa0', ' ').strip() for c in scrape_df.columns]
                        
                        # Mapeo flexible de columnas de Yahoo Finance
                        col_map = {
                            'Event Time': 'Hora',
                            'Time (ET)': 'Hora',
                            'Event': 'Evento',
                            'Market Expectation': 'Previsto',
                            'Briefing': 'Previsto',
                            'Actual': 'Actual',
                            'Prior to This': 'Anterior',
                            'Prior': 'Anterior',
                            'Country': 'Pais'
                        }
                        
                        for _, row in scrape_df.iterrows():
                            # Filtrar solo eventos de EE.UU. para evitar ruido
                            pais = row.get('Country', row.get('Pais', ''))
                            if str(pais).upper() != 'US':
                                continue
                                
                            all_results.append({
                                "Fecha": current_day,
                                "Hora": row.get('Event Time', row.get('Time (ET)', '-')),
                                "Evento": str(row.get('Event', '')).split('*')[0].strip(),
                                "Actual": row.get('Actual', '-'),
                                "Previsto": row.get('Market Expectation', row.get('Briefing', '-')),
                                "Anterior": row.get('Prior to This', row.get('Prior', '-')),
                                "Pais": 'US'
                            })
            except: continue
        
        df_scrape = pd.DataFrame(all_results)
        
        # Combinar con blueprint estático para mayor robustez
        blueprint_path = "macro_blueprint.json"
        if os.path.exists(blueprint_path):
            with open(blueprint_path, 'r') as f:
                bp_data = json.load(f)
            df_bp = pd.DataFrame(bp_data['events'])
            
            # Normalizar fechas para comparación y deduplicación
            # El blueprint usa "Feb 17, 2026", el scraper usa "2026-02-17"
            def normalize_date(d):
                try:
                    if ',' in str(d): # Formato blueprint
                        return dt.strptime(str(d), "%b %d, %Y").strftime("%Y-%m-%d")
                    return str(d) # Formato scraper yf
                except: return str(d)
                
            df_bp['Fecha_Sort'] = df_bp['Fecha'].apply(normalize_date)
            if not df_scrape.empty:
                df_scrape['Fecha_Sort'] = df_scrape['Fecha']
                df_final = pd.concat([df_bp, df_scrape]).drop_duplicates(subset=['Fecha_Sort', 'Evento'], keep='last')
            else:
                df_final = df_bp
            
            # Restaurar formato visible de fecha
            df_final['Fecha'] = df_final['Fecha_Sort'].apply(lambda x: dt.strptime(x, "%Y-%m-%d").strftime("%b %d, %Y"))
        else:
            df_final = df_scrape
            if not df_final.empty:
                df_final['Fecha_Sort'] = df_final['Fecha']
                df_final['Fecha'] = df_final['Fecha'].apply(lambda x: dt.strptime(x, "%Y-%m-%d").strftime("%b %d, %Y"))

        if df_final.empty: 
            return pd.DataFrame(columns=['Fecha', 'Hora', 'Evento', 'Actual', 'Previsto', 'Anterior'])
        
        df_final = df_final.replace('nan', '-').fillna('-')
        # Asegurar orden cronológico
        df_final = df_final.sort_values(['Fecha_Sort', 'Hora']).reset_index(drop=True)
        return df_final[['Fecha', 'Hora', 'Evento', 'Actual', 'Previsto', 'Anterior']]

    except Exception as e:
        return pd.DataFrame({"Error": [f"Error detectado: {str(e)}"]})


# --- DEEP DIVE ANALYSIS MODULE ---

def translate_text(api_key, text, target_lang="Spanish"):
    """Traduce un texto usando la IA de Groq."""
    if not api_key or not text or text == 'Sin descripción disponible.':
        return text
    try:
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        prompt = f"Traduce el siguiente texto de descripción de empresa al {target_lang}. Mantén un tono profesional y técnico. No añadas comentarios extra, solo la traducción:\n\n{text}"
        payload = {
            "model": "llama-3.1-8b-instant",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.2,
            "max_tokens": 1000
        }
        resp = requests.post("https://api.groq.com/openai/v1/chat/completions",
                           headers=headers, json=payload, timeout=20)
        if resp.status_code == 200:
            return resp.json()['choices'][0]['message']['content']
        return text
    except:
        return text

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
        
        # Obtener resumen y traducir si es posible
        summary_en = info.get('longBusinessSummary', 'Sin descripción disponible.')
        
        # Intentar traducción si hay API Key (usamos la global o la de st.secrets si existiera)
        # Para simplificar, lo dejamos para el retorno final
        
        general = {
            'shortName': info.get('shortName', ticker),
            'sector': info.get('sector', 'N/D'),
            'industry': info.get('industry', 'N/D'),
            'country': info.get('country', 'N/D'),
            'employees': info.get('fullTimeEmployees', 0),
            'website': info.get('website', ''),
            'summary': summary_en,
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
                # Mapeo de nombres posibles para yfinance
                map_inc = {
                    'revenue': ['Total Revenue', 'Total Operating IncomeAs Reported', 'Operating Revenue'],
                    'gross': ['Gross Profit'],
                    'op_inc': ['Operating Income', 'Operating Income (Loss)'],
                    'net': ['Net Income', 'Net Income Common Stockholders', 'Net Income From Continuing Operation Net Minority Interest'],
                    'ebitda': ['EBITDA']
                }
                
                for col in inc.columns:
                    year_data = {}
                    year_data['period'] = col.strftime('%Y') if hasattr(col, 'strftime') else str(col)
                    
                    # Función auxiliar para buscar el primer KEY que exista
                    def get_f(df, keys, c):
                        for k in keys:
                            if k in df.index: return float(df.loc[k, c])
                        return 0.0

                    year_data['totalRevenue'] = get_f(inc, map_inc['revenue'], col)
                    year_data['grossProfit'] = get_f(inc, map_inc['gross'], col)
                    year_data['operatingIncome'] = get_f(inc, map_inc['op_inc'], col)
                    year_data['netIncome'] = get_f(inc, map_inc['net'], col)
                    year_data['ebitda'] = get_f(inc, map_inc['ebitda'], col)
                    income_history.append(year_data)
        except:
            pass
 
        # --- BALANCE (Balance Sheet) ---
        balance_data = {}
        try:
            bs = t.balance_sheet
            if bs is not None and not bs.empty:
                latest = bs.iloc[:, 0]
                def get_bs(s, keys):
                    for k in keys:
                        if k in s.index: return float(s[k])
                    return 0.0
                
                balance_data['totalAssets'] = get_bs(latest, ['Total Assets', 'Total Assets As Reported'])
                balance_data['totalLiabilities'] = get_bs(latest, ['Total Liabilities Net Minority Interest', 'Total Liabilities As Reported', 'Total Debt'])
                balance_data['totalEquity'] = get_bs(latest, ['Stockholders Equity', 'Total Equity Gross Minority Interest', 'Common Stock Equity'])
                balance_data['cash'] = get_bs(latest, ['Cash And Cash Equivalents', 'Cash Cash Equivalents And Short Term Investments'])
                balance_data['totalDebt'] = get_bs(latest, ['Total Debt', 'Long Term Debt'])
        except:
            pass
 
        # --- CASH FLOW ---
        cashflow_data = {}
        try:
            cf = t.cashflow
            if cf is not None and not cf.empty:
                latest = cf.iloc[:, 0]
                def get_cf(s, keys):
                    for k in keys:
                        if k in s.index: return float(s[k])
                    return 0.0

                cashflow_data['operatingCashflow'] = get_cf(latest, ['Operating Cash Flow', 'Total Cash From Operating Activities', 'Cash Flow From Continuing Operating Activities'])
                # Capex suele ser negativo en yfinance
                capex = get_cf(latest, ['Capital Expenditure', 'Purchase Of PPE', 'Net PPE Purchase And Sale'])
                cashflow_data['capitalExpenditure'] = cax = float(capex)
                cashflow_data['freeCashflow'] = cashflow_data['operatingCashflow'] + cax # Suma porque capex suele ser negativo
                cashflow_data['dividendsPaid'] = get_cf(latest, ['Common Stock Dividend Paid', 'Cash Dividends Paid'])
                cashflow_data['shareRepurchase'] = get_cf(latest, ['Repurchase Of Capital Stock'])
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
        
        # Datos diarios (1 año para SMA 150/200 y Weinstein)
        daily = t.history(period='1y', interval='1d', auto_adjust=True)
        if daily.empty:
            return None
        
        if isinstance(daily.columns, pd.MultiIndex):
            daily.columns = daily.columns.get_level_values(0)
        
        price = daily['Close'].iloc[-1]
        
        # Weinstein Stage
        w_stage = get_weinstein_stage(daily)

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
        chg_6m = ((price / daily['Close'].iloc[-132]) - 1) * 100 if len(daily) > 132 else 0
        chg_1y = ((price / daily['Close'].iloc[0]) - 1) * 100 if len(daily) > 250 else 0
        
        # Fibonacci (1 año)
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
            'weinstein': w_stage,
            'ema_9': ema_9, 'ema_20': ema_20, 'ema_50': ema_50, 'sma_200': sma_200,
            'rsi': rsi,
            'macd': macd_val, 'signal': signal_val, 'macd_hist': macd_hist,
            'bb_upper': bb_upper, 'bb_lower': bb_lower, 'bb_position': bb_position,
            'atr': atr, 'adx': adx,
            'vol_ratio': vol_ratio, 'avg_vol_20': avg_vol_20,
            'chg_1d': chg_1d, 'chg_5d': chg_5d, 'chg_1m': chg_1m, 'chg_3m': chg_3m, 'chg_6m': chg_6m, 'chg_1y': chg_1y,
            'fib_levels': fib_levels,
            'trend_score': trend_score, 'trend_status': trend_status,
            'daily_data': daily,
            'daily_vol': daily_vol,
        }
    except Exception as e:
        print(f"Error Deep Technical: {e}")
        import traceback
        traceback.print_exc()
        return None
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
Etapa de Weinstein: {tech_data.get('weinstein', 'N/D')}
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
- ZERO HALLUCINATION: No inventes datos. Cíñete estrictamente a las métricas proporcionadas.
- Escribe con argumentos cuantitativos, no con opiniones vagas.
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
            "temperature": 0.1,
            "max_tokens": 2000
        }
        resp = requests.post("https://api.groq.com/openai/v1/chat/completions",
                           headers=headers, json=payload, timeout=30)
        if resp.status_code == 200:
            return resp.json()['choices'][0]['message']['content']
        return f"Error API ({resp.status_code}): {resp.text[:200]}"
    except Exception as e:
        return f"Error en análisis IA: {str(e)}"


def get_platform_context():
    """Recopila el estado actual de la plataforma para darle contexto a AlphaPilot."""
    ny_now = get_ny_time()
    ctx = f"FECHA Y HORA ACTUAL (NY): {ny_now.strftime('%A, %b %d, %Y %H:%M')}\n"
    ctx += "ESTADO ACTUAL DE LA PLATAFORMA:\n"
    
    # 1. Ticker Activo
    if 'last_ticker' in st.session_state:
        ctx += f"- Ticker en pantalla: {st.session_state['last_ticker']}\n"
    
    # 2. Portafolio de la Rueda (Widget Multiselect)
    if 'wheel_multi_selection_v2' in st.session_state and st.session_state['wheel_multi_selection_v2']:
        tickers = st.session_state['wheel_multi_selection_v2']
        ctx += f"- Portafolio Rueda Seleccionado: {', '.join(tickers)}\n"
        # Si hay un análisis previo, incluir el veredicto
        if 'wheel_ai_report' in st.session_state and st.session_state['wheel_ai_report']:
            ctx += f"- El análisis previo de la Rueda sugiere riesgos específicos en esos activos.\n"

    # 3. Datos del Scanner (si existen)
    if 'scanner_results' in st.session_state and not st.session_state['scanner_results'].empty:
        top_m = st.session_state['scanner_results'].head(3)['Ticker'].tolist()
        ctx += f"- Mejores candidatos de Momentum: {', '.join(top_m)}\n"

    # 4. Resultados de Deep Dive
    if 'dd_ticker_active' in st.session_state:
        ctx += f"- Deep Dive Activo: {st.session_state['dd_ticker_active']} (Salud: {st.session_state.get('dd_health', {}).get('total', 'N/D')}/100)\n"

    # 6. Bitácora de Operaciones
    journal_df = market_db.get_journal_entries()
    if not journal_df.empty:
        ctx += f"- Bitácora: {len(journal_df)} operaciones guardadas (puedes consultarlas con 'get_user_journal').\n"

    return ctx

def get_wheel_portfolio_details():
    """Retorna los detalles técnicos de la selección actual del usuario en la estrategia de la Rueda."""
    if 'wheel_multi_selection_v2' not in st.session_state or not st.session_state['wheel_multi_selection_v2']:
        return "El usuario no ha seleccionado ningún activo en 'The Wheel' aún."
    
    tickers = st.session_state['wheel_multi_selection_v2']
    df_cache = market_db.get_wheel_cache()
    if df_cache.empty:
        return "No hay datos en el historial de la Rueda. El usuario debe ejecutar un escaneo primero."
    
    portfolio_df = df_cache[df_cache['Ticker'].isin(tickers)]
    return portfolio_df.to_json(orient='records')

def get_user_journal():
    """Retorna la bitácora de operaciones guardadas por el usuario."""
    df = market_db.get_journal_entries()
    if df.empty:
        return "La bitácora de operaciones está vacía."
    return df.to_json(orient='records')

def get_alpha_pilot_response(api_key, user_input, chat_history):
    """Genera la respuesta de AlphaPilot usando el contexto de la plataforma y herramientas."""
    if not api_key: return "⚠️ Configura la API Key en el archivo .env"
    
    platform_ctx = get_platform_context()
    
    # 1. Definición de Herramientas (Tools)
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_ticker_snapshot",
                "description": "Obtiene el precio actual, cambio del día y rango de precio para un ticker específico.",
                "parameters": {
                    "type": "object",
                    "properties": { "ticker": {"type": "string", "description": "Ticker (ej: AMD, TSLA, AAPL)"} },
                    "required": ["ticker"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "get_deep_technical_analysis",
                "description": "Análisis técnico exhaustivo: RSI, MACD, Fibonacci y rendimientos históricos (1d, 5d, 1m, 6m, 1y).",
                "parameters": {
                    "type": "object",
                    "properties": { "ticker": {"type": "string", "description": "Ticker (ej: AAPL)"} },
                    "required": ["ticker"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "get_economic_calendar",
                "description": "Obtiene la agenda económica para un día o periodo específico.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "date_str": {"type": "string", "description": "Fecha de inicio (YYYY-MM-DD)."},
                        "days": {"type": "integer", "description": "Número de días a consultar (ej: 7 para una semana)."}
                    }
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "get_market_news",
                "description": "Obtiene noticias en tiempo real para un ticker o mercado general.",
                "parameters": {
                    "type": "object",
                    "properties": { "ticker": {"type": "string", "description": "Ticker (ej: AAPL, NVDA) o índices (^GSPC, ^IXIC, XLK para tecnología)"} },
                    "required": ["ticker"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "get_platform_info",
                "description": "Obtiene información detallada sobre las funciones de StratEdge Portfolio (Investment Desk, Scanner, etc.)"
            }
        },
        {
            "type": "function",
            "function": {
                "name": "get_wheel_portfolio_details",
                "description": "Obtiene los detalles técnicos (strike, prima, anualizado) de los activos seleccionados por el usuario en 'The Wheel'."
            }
        },
        {
            "type": "function",
            "function": {
                "name": "get_user_journal",
                "description": "Obtiene la bitácora de operaciones guardadas por el usuario en su diario local."
            }
        }
    ]
    
    system_prompt = f"""
    Eres 'AlphaPilot', el Agente Jefe de Estrategia de StratEdge Portfolio. 
    ESTADO TEMPORAL: {platform_ctx.split('ESTADO ACTUAL')[0].strip()}

    FILOSOFÍA:
    - Eres un experto en mercados financieros, con un tono profesional, analítico y directo.
    - Esta es una plataforma institucional para inversores estratégicos.

    REGLAS DE OPERACIÓN:
    1. HERRAMIENTAS: ÚSALAS siempre que el usuario pregunte por precios, noticias, rendimientos o calendario.
    2. SECTORES: Para Tecnología usa XLK o ^IXIC. Para el mercado general usa ^GSPC.
    3. ZERO HALLUCINATION: No inventes precios. Si una herramienta no devuelve datos, admítelo.
    4. NO MENCIONAR SINTAXIS: Reporta los resultados, no nombres de funciones.
    5. IDIOMA: Responde exclusivamente en ESPAÑOL.
    """
    
    # 2. Limpieza Robusta del Historial (Evita Error 400 y Error 429)
    # Filtramos SOLO mensajes con contenido de texto (eliminamos tool_calls técnicos del historial pasado)
    clean_history = []
    for msg in chat_history:
        if not isinstance(msg, dict): continue
        role = msg.get('role')
        content = msg.get('content')
        
        # Solo conservamos mensajes con texto real para evitar romper el esquema de la API
        if role in ['user', 'assistant'] and content and isinstance(content, str):
            # No enviar errores técnicos al modelo
            if "AlphaPilot offline" in content or "Tuve un problema procesando" in content:
                continue
            clean_history.append({"role": role, "content": content})
            
    # Reducimos la ventana de memoria a los últimos 5 mensajes para ahorrar tokens (Evita Error 429)
    messages = [{"role": "system", "content": system_prompt}] + clean_history[-5:] + [{"role": "user", "content": user_input}]
    
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    
    # Modelo sugerido para mayor velocidad y límites: llama-3.1-8b-instant
    # Modelo sugerido para alta inteligencia: llama-3.3-70b-versatile (pero tiene límites bajos)
    MODEL_ID = "llama-3.1-8b-instant" 

    try:
        # Llamada inicial
        payload = {
            "model": MODEL_ID,
            "messages": messages,
            "tools": tools,
            "tool_choice": "auto",
            "temperature": 0.1
        }
        
        resp = requests.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=payload, timeout=25)
        
        if resp.status_code == 429:
            return "⚠️ El motor de IA (Groq) ha alcanzado su límite de velocidad temporal. Por favor, reintenta en 10-20 segundos."
        
        if resp.status_code != 200:
            # Si el modelo instant falla, intentamos una vez más con un modelo alternativo robusto
            alternativo = "llama3-70b-8192"
            payload["model"] = alternativo
            resp = requests.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=payload, timeout=25)
            if resp.status_code != 200:
                return f"AlphaPilot está saturado (Error {resp.status_code}). Prueba con una pregunta más corta."
            
        resp_data = resp.json()
        message = resp_data['choices'][0]['message']
        
        # 3. Manejo de Tool Calls
        if 'tool_calls' in message and message['tool_calls']:
            assistant_msg = {
                "role": "assistant",
                "tool_calls": message['tool_calls'],
                "content": message.get('content') or ""
            }
            messages.append(assistant_msg)
            
            for tool_call in message['tool_calls']:
                func_name = tool_call['function']['name']
                try:
                    args = json.loads(tool_call['function']['arguments'])
                except:
                    args = {}
                
                result_str = ""
                if func_name == "get_platform_info":
                    result_str = get_platform_info()
                elif func_name == "get_ticker_snapshot":
                    t = args.get('ticker', '').upper()
                    snap = get_ticker_snapshot(t)
                    result_str = json.dumps(snap) if isinstance(snap, dict) else str(snap)
                elif func_name == "get_deep_technical_analysis":
                    t = args.get('ticker', '').upper()
                    tech = get_deep_technical_analysis(t)
                    if tech:
                        result_str = (f"Técnico {t}: ${tech['price']:.2f}, RSI: {tech['rsi']:.1f}, "
                                     f"1Y Chg: {tech['chg_1y']:.2f}%, Trend: {tech['trend_status']}")
                    else: result_str = f"Error obteniendo análisis para {t}"
                elif func_name == "get_economic_calendar":
                    d_str = args.get('date_str') or dt.now().strftime('%Y-%m-%d')
                    df_cal = get_economic_calendar(date_str=d_str, days=int(args.get('days', 7)))
                    result_str = df_cal.to_string() if not df_cal.empty else "No hay eventos."
                elif func_name == "get_market_news":
                    ticker = args.get('ticker', '^GSPC').upper()
                    # Mapeo inteligente para sectores si el usuario pregunta genérico
                    if "TECNOLOG" in ticker or "TECH" in ticker: ticker = "XLK"
                    elif "NASD" in ticker: ticker = "^IXIC"
                    result_str = get_market_news(ticker)
                elif func_name == "get_wheel_portfolio_details":
                    result_str = get_wheel_portfolio_details()
                elif func_name == "get_user_journal":
                    result_str = get_user_journal()
                
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call['id'],
                    "name": func_name,
                    "content": result_str
                })
            
            # Segunda llamada final
            final_payload = {
                "model": MODEL_ID,
                "messages": messages,
                "temperature": 0.1
            }
            final_resp = requests.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=final_payload, timeout=25)
            if final_resp.status_code == 200:
                return final_resp.json()['choices'][0]['message']['content']
            else:
                return "AlphaPilot procesó los datos pero el servidor de respuesta está ocupado. Intenta de nuevo."
        
        return message.get('content', "No pude procesar esa solicitud.")

    except Exception as e:
        return f"Error de conexión con AlphaPilot: {str(e)}"

    except Exception as e:
        return f"Error de conexión con AlphaPilot: {e}"


def calculate_portfolio_correlation(tickers):
    """Calcula correlaciones, betas y simulación de caída para un portafolio vs SPY."""
    try:
        # Descarga focalizada: solo los tickers del portafolio + SPY como benchmark
        all_tickers = list(set(tickers + ['SPY']))
        data = yf.download(all_tickers, period='6mo', interval='1d', auto_adjust=True, progress=False)
        
        # Extraer precios de cierre
        if isinstance(data.columns, pd.MultiIndex):
            closes = data['Close'].dropna()
        else:
            closes = data[['Close']].dropna()
            closes.columns = all_tickers
        
        if closes.empty or len(closes) < 30:
            return None
        
        # Retornos diarios porcentuales
        returns = closes.pct_change().dropna()
        
        # --- 1. Matriz de correlación completa ---
        corr_matrix = returns.corr()
        
        # --- 2. Beta y correlación individual vs SPY ---
        spy_var = returns['SPY'].var()
        individual_stats = []
        
        for t in tickers:
            if t not in returns.columns or t == 'SPY':
                continue
            cov_with_spy = returns[t].cov(returns['SPY'])
            beta = cov_with_spy / spy_var if spy_var > 0 else 1.0
            corr_spy = corr_matrix.loc[t, 'SPY'] if t in corr_matrix.index else 0
            # Volatilidad anualizada
            vol = returns[t].std() * np.sqrt(252) * 100
            individual_stats.append({
                'ticker': t,
                'beta': round(beta, 2),
                'corr_spy': round(corr_spy, 2),
                'volatility': round(vol, 1)
            })
        
        # --- 3. Correlación promedio entre activos del portafolio (diversificación) ---
        port_tickers = [t for t in tickers if t in corr_matrix.index and t != 'SPY']
        avg_corr = 0
        pair_count = 0
        high_corr_pairs = []
        
        for i, t1 in enumerate(port_tickers):
            for t2 in port_tickers[i+1:]:
                c = corr_matrix.loc[t1, t2]
                avg_corr += c
                pair_count += 1
                if c > 0.75:
                    high_corr_pairs.append(f"{t1}/{t2}: {c:.2f}")
        
        avg_corr = round(avg_corr / pair_count, 2) if pair_count > 0 else 0
        
        # Clasificación de diversificación
        if avg_corr < 0.35:
            div_label = "EXCELENTE (Baja correlación)"
        elif avg_corr < 0.55:
            div_label = "BUENA (Correlación moderada)"
        elif avg_corr < 0.75:
            div_label = "LIMITADA (Alta correlación)"
        else:
            div_label = "POBRE (Muy alta correlación — activos se mueven juntos)"
        
        # --- 4. Simulación de caída -10% SPY ---
        simulated_losses = []
        for stat in individual_stats:
            loss = round(stat['beta'] * -10.0, 1)
            simulated_losses.append({'ticker': stat['ticker'], 'loss': loss, 'beta': stat['beta']})
        
        # Beta ponderado (asumimos pesos iguales si no tenemos capital por ticker)
        if individual_stats:
            avg_beta = round(sum(s['beta'] for s in individual_stats) / len(individual_stats), 2)
            portfolio_loss = round(avg_beta * -10.0, 1)
        else:
            avg_beta = 1.0
            portfolio_loss = -10.0
        
        # Activo más/menos vulnerable
        most_vulnerable = max(simulated_losses, key=lambda x: abs(x['loss'])) if simulated_losses else None
        most_defensive = min(simulated_losses, key=lambda x: abs(x['loss'])) if simulated_losses else None
        
        return {
            'individual_stats': individual_stats,
            'avg_correlation': avg_corr,
            'diversification_label': div_label,
            'high_corr_pairs': high_corr_pairs,
            'avg_beta': avg_beta,
            'portfolio_loss_sim': portfolio_loss,
            'simulated_losses': simulated_losses,
            'most_vulnerable': most_vulnerable,
            'most_defensive': most_defensive
        }
    except Exception as e:
        print(f"Error calculating portfolio correlation: {e}")
        return None


def ai_wheel_portfolio_analysis(api_key, portfolio_df, total_budget, correlation_data=None):
    """Realiza un análisis estratégico de un portafolio de la Rueda usando IA, con datos de correlación reales."""
    if not api_key: return "Configura la API Key para el análisis."

    
    portfolio_summary = []
    total_collateral = 0
    sectors = {}
    
    for _, row in portfolio_df.iterrows():
        ticker = row['Ticker']
        sectors[row['Sector']] = sectors.get(row['Sector'], 0) + 1
        collateral = row['Capital Requerido']
        total_collateral += collateral
        portfolio_summary.append(
            f"- {ticker}: Strike ${row['Strike']}, Prima ${row['Prima']}, Retorno Anual {row['Anualizado']}, Sector {row['Sector']}, Salud {row['Salud']}"
        )
    
    sector_str = ", ".join([f"{k} ({v})" for k, v in sectors.items()])
    
    # --- Construir sección de correlación para el prompt ---
    correlation_section = ""
    if correlation_data:
        correlation_section = f"""
    
    ═══ ANÁLISIS DE CORRELACIÓN (Datos reales últimos 6 meses) ═══
    Correlación promedio entre activos: {correlation_data['avg_correlation']} → Diversificación: {correlation_data['diversification_label']}
    Beta ponderado del portafolio vs S&P 500: {correlation_data['avg_beta']}
    """
        
        # Betas individuales
        if correlation_data['individual_stats']:
            correlation_section += "\n    Betas individuales vs SPY:\n"
            for stat in correlation_data['individual_stats']:
                correlation_section += f"    - {stat['ticker']}: β={stat['beta']} | Correlación SPY: {stat['corr_spy']} | Volatilidad anualizada: {stat['volatility']}%\n"
        
        # Pares altamente correlacionados (riesgo de concentración oculta)
        if correlation_data['high_corr_pairs']:
            correlation_section += f"\n    ⚠️ PARES CON ALTA CORRELACIÓN (>0.75): {', '.join(correlation_data['high_corr_pairs'])}\n"
        
        # Simulación de caída
        correlation_section += f"""
    ═══ SIMULACIÓN: CAÍDA -10% DEL S&P 500 ═══
    Pérdida estimada del portafolio: {correlation_data['portfolio_loss_sim']}%
    """
        for sim in correlation_data.get('simulated_losses', []):
            correlation_section += f"    - {sim['ticker']}: {sim['loss']}% (β={sim['beta']})\n"
        
        if correlation_data.get('most_vulnerable'):
            correlation_section += f"    → Activo MÁS vulnerable: {correlation_data['most_vulnerable']['ticker']} ({correlation_data['most_vulnerable']['loss']}%)\n"
        if correlation_data.get('most_defensive'):
            correlation_section += f"    → Activo MÁS defensivo: {correlation_data['most_defensive']['ticker']} ({correlation_data['most_defensive']['loss']}%)\n"
    
    prompt = f"""
    Eres un experto en gestión de riesgos y estrategias de opciones (The Wheel). 
    Analiza este portafolio de Cash Secured Puts:
    
    CAPITAL TOTAL: ${total_budget:,.2f}
    CAPITAL COMPROMETIDO: ${total_collateral:,.2f} ({(total_collateral/total_budget)*100:.1f}%)
    DIVERSIFICACIÓN SECTORIAL: {sector_str}
    
    ACTIVOS SELECCIONADOS:
    {chr(10).join(portfolio_summary)}
    {correlation_section}
    
    OBJETIVO DEL ANÁLISIS:
    1. CONCENTRACIÓN: ¿Hay demasiada exposición a un solo sector o activo? Usa los datos de correlación si están disponibles para identificar concentración oculta (activos de distintos sectores que se mueven juntos).
    2. RIESGO SISTÉMICO: Usando los Betas y la simulación de caída, explica con números concretos cómo impactaría una corrección del S&P 500. Identifica el activo más riesgoso y el más defensivo.
    3. DIVERSIFICACIÓN REAL: Analiza la correlación promedio entre activos. Si es alta (>0.65), advierte que la diversificación sectorial es superficial. Si es baja (<0.40), destaca la fortaleza del portafolio.
    4. SALUD FINANCIERA: ¿Son empresas robustas para mantener en caso de asignación?
    5. VERDICTO ESTRATÉGICO: Asigna una calificación del 1 al 10 al riesgo y da un veredicto formal.
    
    REGLAS CRÍTICAS (ZERO HALLUCINATION):
    - No inventes datos. Basa tus conclusiones ÚNICAMENTE en los números proporcionados arriba.
    - Si los datos de correlación están disponibles, ÚSALOS para dar un análisis cuantitativo preciso.
    - Sé directo y accionable. No uses frases genéricas como "no hay información suficiente".
    
    Responde en Español, con un tono profesional e institucional. Usa Markdown con headers ##.
    """
    
    try:
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        payload = {
            "model": "llama-3.3-70b-versatile",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.1,
            "max_tokens": 2000
        }
        resp = requests.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=payload, timeout=30)
        if resp.status_code == 200:
            return resp.json()['choices'][0]['message']['content']
        return f"Error en la comunicación con la IA (Status {resp.status_code})."
    except Exception as e:
        return f"Error: {e}"


# --- UTILIDADES DE OPCIONES ---
def calculate_delta(S, K, T, r, sigma, option_type='call'):
    """Calcula el Delta de una opción usando Black-Scholes."""
    if sigma <= 0 or T <= 0: return 0
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    if option_type == 'call':
        return norm.cdf(d1)
    else:
        return norm.cdf(d1) - 1

# --- MOMENTUM SCANNER ---
@st.cache_data(ttl=86400, show_spinner=False)
def get_complete_universe():
    base_universe = [
        # Índices y ETFs (Líquidos para Opciones)
        'SPY', 'QQQ', 'IWM', 'DIA', 'XLK', 'XLF', 'XLV', 'XLY', 'XLE', 'XLI', 'XLP', 'XLU', 'XLB', 'XLRE', 'XLC',
        'SMH', 'SOXX', 'IBB', 'XBI', 'GLD', 'SLV', 'TLT', 'EEM', 'GDX', 'KRE', 'XOP', 'XRT', 'TAN', 'ARKK',
        # Tecnología / Megacaps (Líquidos)
        'AAPL', 'MSFT', 'NVDA', 'AMZN', 'GOOGL', 'META', 'TSLA', 'NFLX', 'AVGO', 'CSCO', 'ADBE', 'ORCL', 'CRM', 
        'AMD', 'INTC', 'TXN', 'QCOM', 'MU', 'AMAT', 'LRCX', 'ADI', 'PANW', 'FTNT', 'CRWD', 'SNPS', 'CDNS',
        'SHOP', 'SQ', 'PYPL', 'PLTR', 'SNOW', 'MDB', 'TEAM', 'WDAY', 'NOW', 'DDOG', 'ZS', 'OKTA', 'NET', 'DOCU',
        # Consumo / Retail
        'WMT', 'COST', 'HD', 'LOW', 'TGT', 'NKE', 'SBUX', 'MCD', 'COKE', 'PEP', 'PG', 'CL', 'EL', 'PM', 'MO',
        'LULU', 'TJX', 'MAR', 'HLT', 'BKNG', 'ABNB', 'DASH', 'UBER', 'LYFT', 'RIVN', 'LCID', 'F', 'GM',
        # Financiero
        'JPM', 'BAC', 'GS', 'MS', 'AXP', 'V', 'MA', 'COF', 'C', 'WFC', 'BLK', 'SPGI', 'MCO', 'PYPL', 'HOOD', 'COIN',
        # Salud / Pharma
        'LLY', 'UNH', 'JNJ', 'PFE', 'ABBV', 'MRK', 'AMGN', 'GILD', 'BMY', 'VRTX', 'REGN', 'ISRG', 'TMO', 'DHR',
        # Energía / Materiales / Industrials
        'XOM', 'CVX', 'SLB', 'COP', 'OXY', 'CAT', 'DE', 'GE', 'HON', 'MMM', 'UPS', 'FDX', 'LMT', 'RTX', 'BA', 'NOC',
        'LIN', 'APD', 'FCX', 'NEM', 'NUE', 'AA', 'VALE', 'BHP', 'RIO',
        # Otros Growth / Mid-Cap / Relevantes
        'MELI', 'BABA', 'JD', 'PDD', 'SE', 'CPNG', 'NU', 'DKNG', 'PINS', 'SNAP', 'ROKU', 'U', 'RBLX', 'MSTR', 'MARA', 'RIOT',
        'AFRM', 'UPST', 'SOFI', 'AI', 'ARM', 'SMCI', 'PLUG', 'RUN', 'ENPH', 'FSLR', 'TSM', 'ASML', 'GME', 'AMC'
    ]
    
    try:
        import requests
        import io
        url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
        html = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}).text
        table = pd.read_html(io.StringIO(html))
        df = table[0]
        sp500 = [t.replace('.', '-') for t in df['Symbol'].tolist()]
        return sorted(list(set(base_universe + sp500)))
    except:
        fallback = [
            'T', 'VZ', 'TMUS', 'DIS', 'CMCSA', 'CHTR', 'NFLX', 'SONY', 'PARA', 'WBD',
            'ADP', 'PAYX', 'FIS', 'FISV', 'GPN', 'INTU', 'ADSK', 'ANSS', 'TEAM', 'ZM',
            'SYK', 'BSX', 'EW', 'ZTS', 'IDXX', 'ALGN', 'A', 'STZ', 'BRK-B', 'KDP', 'MDLZ',
            'K', 'GIS', 'CPB', 'SJM', 'HSY', 'ADM', 'TSN', 'KHC', 'SYY', 'TFC', 'USB', 'PNC', 
            'TROW', 'MET', 'PRU', 'AIG', 'TRV', 'CB', 'CME', 'ICE', 'ETN', 'ITW', 'EMR',
            'WM', 'RSG', 'NSC', 'UNP', 'CSX', 'ODFL', 'MAR', 'EXPE', 'CCL', 'RCL', 'NCLH',
            'DHR', 'TMO', 'WAT', 'VTRS', 'HCA', 'HUM', 'CI', 'CVS', 'CNC', 'MCK', 'COR',
            'EOG', 'MPC', 'PSX', 'VLO', 'DVN', 'FANG', 'O', 'AMT', 'CCI', 'PLD',
            'EQIX', 'PSA', 'DLR', 'VICI', 'WY', 'SPG', 'AVB', 'EQR'
        ]
        return sorted(list(set(base_universe + fallback)))


def analyze_single_ticker_wheel(ticker, hist, budget, max_price_filter, r=0.045):
    """Procesa un solo ticker para el ciclo de la Rueda usando datos pre-filtrados."""
    try:
        if hist is None or hist.empty: return None
        curr_price = float(hist['Close'].iloc[-1])
        
        # Weinstein (ya filtrado pero re-calculamos para mostrarlo)
        w_stage = get_weinstein_stage(hist)
            
        fin_data = get_deep_financials(ticker)
        health = calculate_financial_health_score(fin_data)
        if health['total'] < 50:
            return None
            
        t = yf.Ticker(ticker)
        expirations = t.options
        if not expirations: return None
        
        target_date = None
        today = dt.now()
        for exp in expirations:
            try:
                exp_dt = dt.strptime(exp, '%Y-%m-%d')
                days_to_exp = (exp_dt - today).days
                if 25 <= days_to_exp <= 55: # Rueda estándar 30-45 dte
                    target_date = exp
                    T = days_to_exp / 365.0
                    break
            except: continue
        
        if not target_date: return None
        
        opts = t.option_chain(target_date)
        puts = opts.puts
        
        best_put = None
        min_delta_diff = float('inf')
        for _, put in puts.iterrows():
            strike = put['strike']
            iv = put['impliedVolatility']
            if strike >= curr_price or iv < 0.01: continue
            calc_d = calculate_delta(curr_price, strike, T, r, iv, 'put')
            diff = abs(calc_d - (-0.30))
            if diff < min_delta_diff:
                min_delta_diff = diff
                best_put = put
                best_put['delta'] = calc_d
        
        if best_put is not None:
            premium = (best_put['bid'] + best_put['ask']) / 2
            if premium <= 0: premium = best_put['lastPrice']
            collateral = best_put['strike'] * 100
            yield_pct = (premium * 100 / collateral) * 100
            days_to_exp = (dt.strptime(target_date, '%Y-%m-%d') - today).days
            annualized = (yield_pct / days_to_exp) * 365
            
            dte_earnings = None
            try:
                cal = t.calendar
                if cal and 'Earnings Date' in cal and cal['Earnings Date']:
                    earn_date = cal['Earnings Date'][0]
                    dte_earnings = (earn_date - today.date()).days
            except:
                pass
            
            return {
                'Ticker': ticker,
                'Sector': fin_data['general'].get('sector', 'N/D'),
                'Precio': round(curr_price, 2),
                'W Stage': w_stage,
                'Salud': f"{health['total']}/100",
                'Strike': best_put['strike'],
                'Delta': round(best_put['delta'], 2),
                'Prima': round(premium, 2),
                'Capital Requerido': collateral,
                'yield_pct': yield_pct,
                'annualized': annualized,
                'Retorno': f"{yield_pct:.2f}%",
                'Anualizado': f"{annualized:.1f}%",
                'Vencimiento': target_date,
                'Días a Earnings': dte_earnings
            }
    except Exception as e:
        print(f"Error Wheel {ticker}: {e}")
    return None

def get_wheel_recommendations(budget, max_price_filter=None):
    """
    Busca mejores acciones para 'The Wheel' usando Bulk Download + Parallel Processing.
    Optimizado para evitar rate-limits de Yahoo Finance.
    """
    results = []
    WHEEL_UNIVERSE = get_complete_universe() 
    
    with st.spinner(f"🚀 Pre-escaneando {len(WHEEL_UNIVERSE)} activos..."):
        try:
            # Descarga masiva para filtros rápidos (Weinstein + Precio)
            data = yf.download(WHEEL_UNIVERSE, period='1y', group_by='ticker', progress=False, auto_adjust=True)
        except:
            return pd.DataFrame()

    if data.empty: return pd.DataFrame()

    survivors = []
    the_max_p = max_price_filter if max_price_filter else (budget / 100)
    
    for ticker in WHEEL_UNIVERSE:
        try:
            if ticker in data.columns.get_level_values(0):
                hist = data[ticker].dropna()
                if hist.empty or len(hist) < 150: continue
                
                curr_price = float(hist['Close'].iloc[-1])
                if curr_price > the_max_p: continue
                
                w_stage = get_weinstein_stage(hist)
                # Solo Etapa 1 o 2 (Acumulación o Tendencia)
                if "1" not in str(w_stage) and "2" not in str(w_stage): continue
                
                survivors.append((ticker, hist))
        except: continue

    if not survivors:
        return pd.DataFrame()

    progress_bar = st.progress(0)
    status_text = st.empty()
    total = len(survivors)
    
    # Solo procesamos los sobrevivientes en paralelo (Llamadas caras de Opciones/Financials)
    # Reducimos los workers a 4 para prevenir bloqueos por rate-limit
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(analyze_single_ticker_wheel, t, h, budget, max_price_filter): t for t, h in survivors}
        for i, future in enumerate(futures):
            ticker = futures[future]
            res = future.result()
            if res: results.append(res)
            progress_bar.progress((i + 1) / total)
            status_text.caption(f"Calculando opciones de {ticker}... ({i+1}/{total})")
            
    progress_bar.empty()
    status_text.empty()
    return pd.DataFrame(results)

def analyze_single_ticker_momentum(ticker, hist, price_min, price_max, min_volume, smooth_momentum=False):
    """Procesa un solo ticker para el escáner de momentum usando datos pre-cargados."""
    try:
        if hist is None or hist.empty or len(hist) < 20:
            return None
        
        # Aplanar si es MultiIndex (aunque yf.download maneja esto, por seguridad)
        if isinstance(hist.columns, pd.MultiIndex):
            hist.columns = hist.columns.get_level_values(0)

        price = float(hist['Close'].iloc[-1])
        # Filtro de precio
        if price < price_min or price > price_max:
            return None
        
        # Filtro de volumen
        avg_vol = hist['Volume'].tail(20).mean()
        if avg_vol < min_volume:
            return None
        
        # Calcular etapa de Weinstein (Aproximación local para velocidad)
        w_stage = get_weinstein_stage(hist)

        # Calcular indicadores técnicos
        ema_20 = hist['Close'].ewm(span=20).mean().iloc[-1]
        ema_50 = hist['Close'].ewm(span=50).mean().iloc[-1]
        
        # RSI
        delta = hist['Close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss.replace(0, np.nan)
        rsi_series = (100 - (100 / (1 + rs)))
        rsi = rsi_series.iloc[-1] if not rsi_series.empty else 50
        
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
        # RSI entre 50-70 (+20) o >70 (+10) - Score Momentum puro
        if 50 <= rsi <= 70: score += 20
        elif rsi > 70: score += 10
        # Cambio 5d positivo (+20)
        if chg_5d > 0: score += 20
        # Volumen superior al promedio (+20)
        if vol_ratio > 1.1: score += 20
        
        # --- FILTRO DE MOMENTUM SUAVE ---
        if smooth_momentum:
            # Si la volatilidad diaria es baja (< 2%), premiamos la estabilidad
            if daily_vol < 0.02: 
                score += 20
            # Si es muy alta (> 4%), penalizamos fuertemente
            elif daily_vol > 0.04:
                score -= 20
        
        # Calcular Distancia a SMA 200 (Tendencia largo plazo)
        sma_200 = hist['Close'].rolling(window=200).mean().iloc[-1] if len(hist) >= 200 else ema_50 # Fallback
        dist_sma200 = ((price / sma_200) - 1) * 100

        return {
            'Ticker': ticker,
            'Precio': round(price, 2),
            'Score': score,
            'RSI': round(rsi, 2),
            'Etapa W': w_stage,
            '1D%': round(chg_1d, 2),
            '5D%': round(chg_5d, 2),
            '20D%': round(chg_20d, 2),
            'Vol_Ratio': round(vol_ratio, 2),
            'Dist_SMA200': round(dist_sma200, 2)
        }
    except:
        return None

@st.cache_data(ttl=900, show_spinner=False)
def scan_momentum_stocks(price_min, price_max, min_volume, smooth_momentum=False):
    """Escanea el universo usando descargas masivas (Bulk) para máxima velocidad y seguridad."""
    results = []
    universe = get_complete_universe()
    
    with st.spinner(f"🚀 Descargando datos de {len(universe)} activos..."):
        try:
            # Descargar TODO el universo en una sola petición (Ahorra ~250 peticiones)
            # Usamos 1y para tener suficiente historial para SMA 150 (Weinstein)
            data = yf.download(universe, period='1y', interval='1d', group_by='ticker', progress=False, auto_adjust=True)
        except Exception as e:
            st.error(f"Error en descarga masiva: {e}")
            return pd.DataFrame()

    if data.empty:
        return pd.DataFrame()

    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Procesar cada ticker del dataframe masivo
    for i, ticker in enumerate(universe):
        try:
            # Extraer el historial de este ticker específico del gran DataFrame
            if ticker in data.columns.get_level_values(0):
                hist = data[ticker].dropna()
                res = analyze_single_ticker_momentum(ticker, hist, price_min, price_max, min_volume, smooth_momentum)
                if res:
                    results.append(res)
        except:
            continue
            
        progress_bar.progress((i + 1) / len(universe))
        status_text.caption(f"Analizando {ticker}...")

    progress_bar.empty()
    status_text.empty()
    
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
            'Market Cap': info.get('marketCap', t.fast_info.get('marketCap', 0)),
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
            "model": "llama-3.1-8b-instant",
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
    - Radar Opciones: {context_data.get('options_sent', 'N/A')} (P/C: {context_data.get('pc_ratio', 'N/A')})
    - Muros Clave: Call {context_data.get('call_wall', 'N/A')} / Put {context_data.get('put_wall', 'N/A')}
    
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
        "model": "llama-3.1-8b-instant",
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

            cursor.execute('''
                CREATE TABLE IF NOT EXISTS wheel_cache (
                    ticker TEXT PRIMARY KEY,
                    sector TEXT,
                    price REAL,
                    w_stage TEXT,
                    health TEXT,
                    strike REAL,
                    delta REAL,
                    premium REAL,
                    collateral REAL,
                    yield_pct REAL,
                    annualized REAL,
                    expiration TEXT,
                    last_updated TEXT
                )
            ''')
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS scanner_cache (
                    ticker TEXT PRIMARY KEY,
                    price REAL,
                    score INTEGER,
                    rsi REAL,
                    etapa_w TEXT,
                    chg_1d REAL,
                    chg_5d REAL,
                    chg_20d REAL,
                    vol_ratio REAL,
                    dist_sma200 REAL,
                    sector TEXT,
                    last_updated TEXT
                )
            ''')
            # Migración: Añadir columnas nuevas si no existen
            try:
                cursor.execute("ALTER TABLE scanner_cache ADD COLUMN rsi REAL")
                cursor.execute("ALTER TABLE scanner_cache ADD COLUMN dist_sma200 REAL")
            except:
                pass
                
            try:
                cursor.execute("ALTER TABLE wheel_cache ADD COLUMN dte_earnings INTEGER")
            except:
                pass
            
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"DB Error: {e}")

    def save_wheel_results(self, df):
        if df.empty: return
        try:
            conn = sqlite3.connect(self.db_file)
            now = get_ny_time().strftime('%Y-%m-%d %H:%M:%S')
            for _, row in df.iterrows():
                # Limpiar % de las cadenas si subieron como string
                y_val = float(str(row['Retorno']).replace('%', '')) if isinstance(row['Retorno'], str) else row['Retorno']
                a_val = float(str(row['Anualizado']).replace('%', '')) if isinstance(row['Anualizado'], str) else row['Anualizado']
                
                conn.execute('''
                    INSERT OR REPLACE INTO wheel_cache 
                    (ticker, sector, price, w_stage, health, strike, delta, premium, collateral, yield_pct, annualized, expiration, last_updated, dte_earnings)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    row['Ticker'], row['Sector'], row['Precio'], row['W Stage'], row['Salud'],
                    row['Strike'], row['Delta'], row['Prima'], row['Capital Requerido'],
                    y_val, a_val, row['Vencimiento'], now, row.get('Días a Earnings')
                ))
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"Error saving wheel cache: {e}")

    def get_wheel_cache(self):
        try:
            conn = sqlite3.connect(self.db_file)
            df = pd.read_sql_query("SELECT * FROM wheel_cache ORDER BY annualized DESC", conn)
            conn.close()
            # Restaurar formatos para la UI
            if not df.empty:
                df['Retorno'] = df['yield_pct'].apply(lambda x: f"{x:.2f}%")
                df['Anualizado'] = df['annualized'].apply(lambda x: f"{x:.1f}%")
                if 'dte_earnings' not in df.columns:
                    df['dte_earnings'] = None
                # Renombrar columnas para consistencia con la lógica existente
                df = df.rename(columns={
                    'ticker': 'Ticker', 'sector': 'Sector', 'price': 'Precio',
                    'w_stage': 'W Stage', 'health': 'Salud', 'strike': 'Strike',
                    'delta': 'Delta', 'premium': 'Prima', 'collateral': 'Capital Requerido',
                    'expiration': 'Vencimiento', 'last_updated': 'Última Actualización',
                    'dte_earnings': 'Días a Earnings'
                })
            return df
        except:
            return pd.DataFrame()

    def save_scanner_results(self, df):
        """Guarda los resultados del Momentum Scanner en la base de datos."""
        if df.empty: return
        try:
            conn = sqlite3.connect(self.db_file)
            now = get_ny_time().strftime('%Y-%m-%d %H:%M:%S')
            # Limpiar tabla antes de insertar (reemplazo completo)
            conn.execute('DELETE FROM scanner_cache')
            for _, row in df.iterrows():
                conn.execute('''
                    INSERT OR REPLACE INTO scanner_cache 
                    (ticker, price, score, rsi, etapa_w, chg_1d, chg_5d, chg_20d, vol_ratio, dist_sma200, sector, last_updated)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    row['Ticker'], row['Precio'], row['Score'], row.get('RSI', 50), row.get('Etapa W', 'N/D'),
                    row.get('1D%', 0), row.get('5D%', 0), row.get('20D%', 0),
                    row.get('Vol_Ratio', 0), row.get('Dist_SMA200', 0), row.get('Sector', 'N/D'), now
                ))
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"Error saving scanner cache: {e}")

    def get_scanner_cache(self):
        """Carga los resultados del último escaneo desde la base de datos."""
        try:
            conn = sqlite3.connect(self.db_file)
            df = pd.read_sql_query("SELECT * FROM scanner_cache ORDER BY score DESC", conn)
            conn.close()
            if not df.empty:
                df = df.rename(columns={
                    'ticker': 'Ticker', 'price': 'Precio', 'score': 'Score',
                    'etapa_w': 'Etapa W', 'chg_1d': '1D%', 'chg_5d': '5D%',
                    'chg_20d': '20D%', 'vol_ratio': 'Vol_Ratio', 'sector': 'Sector',
                    'last_updated': 'Última Actualización'
                })
            return df
        except:
            return pd.DataFrame()

    def get_scanner_last_updated(self):
        """Retorna la fecha del último escaneo guardado."""
        try:
            conn = sqlite3.connect(self.db_file)
            cursor = conn.execute("SELECT last_updated FROM scanner_cache LIMIT 1")
            row = cursor.fetchone()
            conn.close()
            return row[0] if row else None
        except:
            return None

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


    def save_journal_entry(self, ticker, price, score, verdict, reasoning, sl=0.0, tp=0.0):
        conn = None
        try:
            conn = sqlite3.connect(self.db_file)
            now = get_ny_time().strftime('%Y-%m-%d %H:%M:%S')
            
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
            
            # --- VERIFICACIÓN DE DUPLICADOS ---
            # Evitar guardar la misma operación múltiples veces (mismo ticker, día, precio y razonamiento)
            check_query = '''
                SELECT id FROM journal 
                WHERE ticker = ? 
                AND date(entry_date) = date(?) 
                AND (abs(entry_price - ?) < 0.001 AND reasoning = ?)
            '''
            cursor.execute(check_query, (t_val, d_val, p_val, r_val))
            
            if cursor.fetchone():
                # Si ya existe, asumimos éxito (idempotencia) para no mostrar error al usuario
                return True

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

# --- UTILIDADES DE DATOS OBSOLETAS ---
# (Removidas para optimizar el rendimiento y eliminar logs DEBUG)
        


# --- UTILIDADES DE MERCADO Y CONTEXTO ---
@st.cache_data(ttl=3600, show_spinner=False)
def get_market_context():
    """Analiza la tendencia general del mercado (SPY) y sectorial."""
    try:
        # Tickers: SPY + 11 Sectores
        tickers = ['SPY', 'XLK', 'XLF', 'XLE', 'XLV', 'XLY', 'XLP', 'XLU', 'XLI', 'XLB', 'XLRE', 'XLC']
        # Descarga optimizada
        data = yf.download(tickers, period='1y', interval='1d', group_by='ticker', progress=False, auto_adjust=True)
        
        context = {}
        
        # 1. Analizar Tendencia SPY (Mercado General)
        # Manejo robusto de MultiIndex de yfinance
        try:
            spy_hist = data['SPY'].dropna()
        except KeyError:
             return None
             
        if not spy_hist.empty:
            price = spy_hist['Close'].iloc[-1]
            sma_200 = spy_hist['Close'].rolling(window=200).mean().iloc[-1]
            
            trend = "ALCISTA" if price > sma_200 else "BAJISTA"
            
            context['SPY'] = {
                'Price': price,
                'SMA200': sma_200,
                'Trend': trend,
                'Dist_SMA200': ((price/sma_200)-1)*100
            }
        
        # 2. Ranking de Sectores
        sector_perf = []
        sector_names = {
            'XLK': 'Tecnología', 'XLF': 'Financiero', 'XLE': 'Energía', 'XLV': 'Salud',
            'XLY': 'Consumo Disc.', 'XLP': 'Consumo Básico', 'XLU': 'Utilities', 
            'XLI': 'Industrial', 'XLB': 'Materiales', 'XLRE': 'Real Estate', 'XLC': 'Comunicaciones'
        }
        
        for t in tickers:
            if t == 'SPY': continue
            try:
                hist = data[t].dropna()
                if not hist.empty:
                    # Performance relativo a 20 días y distancia a SMA50
                    p = hist['Close'].iloc[-1]
                    sma_50 = hist['Close'].rolling(window=50).mean().iloc[-1]
                    chg_20d = ((p / hist['Close'].iloc[-21]) - 1) * 100 if len(hist) > 20 else 0
                    dist_sma50 = ((p / sma_50) - 1) * 100
                    
                    # Score simple de fuerza relativa
                    score = chg_20d + dist_sma50
                    sector_perf.append({
                        'Ticker': t, 
                        'Name': sector_names.get(t, t),
                        'Score': score, 
                        'Chg_20d': chg_20d
                    })
            except:
                continue
        
        context['Sectors'] = pd.DataFrame(sector_perf).sort_values('Score', ascending=False)
        return context
    except Exception as e:
        print(f"Market Context Error: {e}")
        return None

# Función principal de la app Streamlit
def main():
    # --- FIX: Event Loop handling for Streamlit ---
    try:
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
        if loop.is_closed():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
    except:
        pass

    st.set_page_config(
        page_title="StratEdge Portfolio | Multi-Horizon Strategy Suite",
        page_icon="⚖️",
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    # --- CONFIGURACIÓN DE APIS ---
    groq_api_key = os.getenv('GROQ_API_KEY')

    # --- HEADER & ESTATUS (RESTORED) ---
    ny_now = get_ny_time()
    status_text, status_code = check_market_status()
    status_color = "#28a745" if status_code == "open" else "#ffc107" if status_code == "pre" else "#dc3545"
    
    col_h1, col_h2 = st.columns([3, 1])
    with col_h1:
        st.title("StratEdge Portfolio")
    with col_h2:
        st.markdown(f"""
            <div style='text-align: right; padding: 10px; border-radius: 10px; background: rgba(0,0,0,0.3); border-right: 5px solid {status_color};'>
                <span style='color: #888; font-size: 0.8em;'>🕒 NY: {ny_now.strftime('%H:%M:%S')}</span><br>
                <strong style='color: {status_color};'>{status_text}</strong>
            </div>
        """, unsafe_allow_html=True)

    # --- VALORES DERIVADOS ---
    selected_stock = 'Apple (AAPL)'
    end_date = get_ny_time().date()
    default_start = end_date - timedelta(days=1825)
    start_date = default_start
    ticker = TOP_20_STOCKS.get(selected_stock, 'AAPL')

    # Validar rango de fecha de inicio (Limitar a 30 años para estabilidad)
    max_days_back = 10950 # ~30 años
    if (end_date - start_date).days > max_days_back:
        start_date = end_date - timedelta(days=max_days_back)


    # Resetear métricas si cambia el ticker
    if 'last_ticker' not in st.session_state or st.session_state['last_ticker'] != ticker:
        st.session_state['last_ticker'] = ticker
        if 'metrics' in st.session_state:
            del st.session_state['metrics']
            
    # --- ESTILOS PARA CHATBOT FLOTANTE PREMIUM ---
    st.markdown("""
        <style>
        /* Modern Scrollbar */
        #alpha_chat_box ::-webkit-scrollbar {
            width: 6px;
        }
        #alpha_chat_box ::-webkit-scrollbar-track {
            background: rgba(0,0,0,0.1);
        }
        #alpha_chat_box ::-webkit-scrollbar-thumb {
            background: rgba(40,167,69,0.3);
            border-radius: 10px;
        }
        #alpha_chat_box ::-webkit-scrollbar-thumb:hover {
            background: rgba(40,167,69,0.5);
        }

        /* Botón Flotante con Pulso */
        div.stButton > button[key="chat_bubble"] {
            position: fixed;
            bottom: 30px;
            right: 30px;
            width: 80px;
            height: 80px;
            border-radius: 50% !important;
            background: linear-gradient(135deg, #28a745 0%, #1e7e34 100%) !important;
            border: 1px solid rgba(255,255,255,0.15) !important;
            box-shadow: 0 8px 32px rgba(0,0,0,0.4), 0 0 0 0px rgba(40,167,69,0.4);
            z-index: 1000000;
            display: flex !important;
            align-items: center;
            justify-content: center;
            font-size: 40px;
            transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
            padding: 0 !important;
            animation: pulse-green 3s infinite;
        }

        @keyframes pulse-green {
            0% { box-shadow: 0 8px 32px rgba(0,0,0,0.4), 0 0 0 0px rgba(40,167,69,0.7); }
            70% { box-shadow: 0 8px 32px rgba(0,0,0,0.4), 0 0 0 15px rgba(40,167,69,0); }
            100% { box-shadow: 0 8px 32px rgba(0,0,0,0.4), 0 0 0 0px rgba(40,167,69,0); }
        }

        div.stButton > button[key="chat_bubble"]:hover {
            transform: scale(1.1) rotate(5deg) translateY(-5px);
            background: linear-gradient(135deg, #1e7e34 0%, #28a745 100%) !important;
        }
        
        /* Contenedor del Chat (Glassmorphism Avanzado) */
        div[data-testid="stVerticalBlock"] > div:has(#alpha_chat_anchor) {
            position: fixed;
            bottom: 120px;
            right: 30px;
            width: 420px;
            max-height: 700px;
            background: rgba(13, 17, 23, 0.9);
            backdrop-filter: blur(20px) saturate(180%);
            -webkit-backdrop-filter: blur(20px) saturate(180%);
            border: 1px solid rgba(40,167,69,0.3);
            border-radius: 28px;
            z-index: 1000001;
            padding: 0px;
            box-shadow: 0 25px 50px -12px rgba(0,0,0,0.8);
            animation: slideInChat 0.3s ease-out;
            overflow: hidden;
        }

        #alpha_chat_content {
            padding: 20px;
        }

        @keyframes slideInChat {
            from { opacity: 0; transform: translateY(30px) scale(0.95); }
            to { opacity: 1; transform: translateY(0) scale(1); }
        }

        /* Título y Header */
        .chat-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 15px;
            padding-bottom: 10px;
            border-bottom: 1px solid rgba(255,255,255,0.05);
        }

        /* Estilo de Mensajes AlphaPilot */
        #alpha_chat_box [data-testid="stChatMessage"] {
            background: rgba(255,255,255,0.02) !important;
            border-radius: 16px !important;
            border: 1px solid rgba(255,255,255,0.04) !important;
            margin-bottom: 10px !important;
            padding: 10px !important;
        }

        #alpha_chat_box [data-testid="stChatMessageContent"] p {
            font-size: 0.95rem !important;
            line-height: 1.5 !important;
        }

        /* Input del Chat */
        #alpha_chat_box .stChatInput {
            border-radius: 12px !important;
            border: 1px solid rgba(40,167,69,0.2) !important;
        }
        </style>
    """, unsafe_allow_html=True)

    # --- LÓGICA DEL CHATBOT FLOTANTE ---
    if "chat_open" not in st.session_state:
        st.session_state["chat_open"] = False
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []
    if "active_tab" not in st.session_state:
        st.session_state["active_tab"] = "🔬 Asset Scanner"

    # Botón Toggle
    bubble_icon = "✖️" if st.session_state["chat_open"] else "🤖"
    if st.button(bubble_icon, key="chat_bubble"):
        st.session_state["chat_open"] = not st.session_state["chat_open"]
        st.rerun()

    # Ventana de Chat
    if st.session_state["chat_open"]:
        with st.container():
            st.markdown('<div id="alpha_chat_anchor"></div>', unsafe_allow_html=True)
            st.markdown('<div id="alpha_chat_content">', unsafe_allow_html=True)
            
            # Header Personalizado
            h_col1, h_col2, h_col3 = st.columns([5, 1, 1])
            with h_col1:
                st.markdown("""
                    <div style='display: flex; align-items: center; gap: 10px;'>
                        <span style='font-size: 24px;'>🦾</span>
                        <div style='line-height: 1;'>
                            <strong style='font-size: 1.1em; color: #28a745;'>AlphaPilot</strong><br>
                            <small style='color: #888;'>Estratega de Mercado AI</small>
                        </div>
                    </div>
                """, unsafe_allow_html=True)
            with h_col2:
                if st.button("🗑️", help="Limpiar Historial", key="clear_chat_btn"):
                    st.session_state["chat_history"] = []
                    st.rerun()
            with h_col3:
                if st.button("✖️", help="Cerrar Chat", key="close_chat_btn"):
                    st.session_state["chat_open"] = False
                    st.rerun()

            st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
            
            # Área de Mensajes
            inner_chat = st.container(height=380)
            
            with inner_chat:
                if not st.session_state["chat_history"]:
                    st.info("👋 ¡Hola! Soy AlphaPilot. Analizo el mercado en tiempo real para ayudarte a encontrar las mejores oportunidades. ¿En qué puedo apoyarte hoy?")
                
                for msg in st.session_state["chat_history"]:
                    with st.chat_message(msg["role"]):
                        st.markdown(msg["content"])

            # Input y Lógica
            api_key = os.getenv('GROQ_API_KEY')
            if prompt := st.chat_input("¿Qué analizamos ahora?"):
                st.session_state["chat_history"].append({"role": "user", "content": prompt})
                with inner_chat:
                    with st.chat_message("user"):
                        st.markdown(prompt)

                    with st.chat_message("assistant"):
                        with st.spinner("AlphaPilot está procesando..."):
                            response = get_alpha_pilot_response(api_key, prompt, st.session_state["chat_history"])
                            st.markdown(response)
                            st.session_state["chat_history"].append({"role": "assistant", "content": response})
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)

    # --- CONFIGURACIÓN ESTRUCTURAL ---
    # Nota: Ya no cargamos datos globales aquí para evitar lentitud y logs innecesarios.
    # Cada módulo ahora gestiona sus propios datos bajo demanda.
    
    # --- ESTRUCTURA DE PANTALLA PERSISTENTE: STRATEDGE HUB ---
    tab_list = ["🔬 Asset Scanner", "🎡 The Wheel", "🔍 Strategic Analysis", "📅 Economic Outlook", "📜 History"]
    
    active_selection = st.segmented_control(
        "Navegación Principal",
        tab_list,
        default=st.session_state.get("active_tab", tab_list[0]),
        label_visibility="collapsed",
        key="main_nav_control"
    )
    
    if active_selection:
        st.session_state["active_tab"] = active_selection
    
    active_tab = st.session_state.get("active_tab", tab_list[0])

    # --- NOTIFICACIONES GLOBALES (Instantáneas tras Guardar) ---
    if 'save_success' in st.session_state:
        msg = st.session_state.pop('save_success')
        st.balloons()
        st.toast(msg, icon="✅")
    if 'save_error' in st.session_state:
        st.error(st.session_state.pop('save_error'))



    # --- CALCULADORA DE POSICIÓN GLOBAL (SIDEBAR) ---
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

    st.markdown("---")



    # --- MOMENTUM SCANNER TAB ---
    if active_tab == "🔬 Asset Scanner":
        st.subheader("🔬 Buscador de Oportunidades: Momentum & Reversión")
        
        # --- SEMÁFORO DE MERCADO ---
        mkt_context = get_market_context()
        if mkt_context and 'SPY' in mkt_context:
            spy_data = mkt_context['SPY']
            trend_color = "#28a745" if spy_data['Trend'] == 'ALCISTA' else "#dc3545"
            trend_icon = "📈" if spy_data['Trend'] == 'ALCISTA' else "📉"
            
            # Obtener top sectores
            top_sectors = ""
            if not mkt_context['Sectors'].empty:
                top_3 = mkt_context['Sectors'].head(3)
                top_sectors = " | ".join([f"**{r['Name']}**" for _, r in top_3.iterrows()])
            
            st.markdown(f"""
                <div style='padding: 15px; border-radius: 10px; background: rgba(0,0,0,0.2); border: 1px solid {trend_color}; margin-bottom: 20px;'>
                    <div style='display: flex; justify-content: space-between; align-items: center;'>
                        <div>
                            <span style='color: #bbb; font-size: 0.9em;'>TENDENCIA DE MERCADO (SPY)</span><br>
                            <strong style='color: {trend_color}; font-size: 1.2em;'>{trend_icon} {spy_data['Trend']}</strong>
                            <span style='color: #888; font-size: 0.8em; margin-left: 10px;'>(SMA 200 Dist: {spy_data['Dist_SMA200']:.1f}%)</span>
                        </div>
                        <div style='text-align: right;'>
                            <span style='color: #bbb; font-size: 0.9em;'>SECTORES FUERTES</span><br>
                            <span style='color: #eee;'>{top_sectors}</span>
                        </div>
                    </div>
                </div>
            """, unsafe_allow_html=True)

        st.info("💡 **Momentum:** Sigue la fuerza (comprar alto para vender más alto). **Reversión:** Busca rebotes en caídas (comprar barato en soporte).")
        
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
        
        
        # Filtros Avanzados
        fc_1, fc_2 = st.columns(2)
        with fc_1:
             # Filtro de Volatilidad (Smooth Momentum)
            smooth_check = st.checkbox("🧘 Filtrar por 'Momentum Suave' (Evita saltos violentos)", value=False)
        with fc_2:
            # Filtro de Sobreventa
            oversold_mode = st.checkbox("📉 Buscar 'Reversión / Sobreventa' (RSI < 35)", value=False)

        
        
        # --- CARGAR CACHÉ AL ABRIR (si no hay resultados en session_state) ---
        if 'scan_results' not in st.session_state or st.session_state['scan_results'].empty:
            cached_scanner = market_db.get_scanner_cache()
            if not cached_scanner.empty:
                st.session_state['scan_results'] = cached_scanner

        # Mostrar info del último escaneo
        last_scan_date = market_db.get_scanner_last_updated()
        if last_scan_date:
            st.caption(f"📅 Último escaneo guardado: **{last_scan_date}** (NY)")

        if st.button("🚀 Iniciar Escaneo de Mercado", type="primary"):
            scan_df = scan_momentum_stocks(price_range[0], price_range[1], min_vol, smooth_check)

            if not scan_df.empty:
                # Enriquecer con Sectores para el Heatmap (solo para los resultados)
                with st.spinner('Mapeando sectores en paralelo...'):
                    def get_ticker_sector(t):
                        try:
                            # Cachear el sector para no saturar API
                            return yf.Ticker(t).info.get('sector', 'Otros')
                        except:
                            return 'N/D'
                    
                    with ThreadPoolExecutor(max_workers=10) as executor:
                        sectors = list(executor.map(get_ticker_sector, scan_df['Ticker']))
                    scan_df['Sector'] = sectors
                
                scan_df = scan_df[scan_df['Score'] >= min_score].reset_index(drop=True)
                
                # --- LOGICA DE FILTRADO PARA SOBREVENTA ---
                if oversold_mode:
                    if 'RSI' in scan_df.columns:
                        # Filtrar RSI bajo (sobreventa)
                        # También verificamos que no sea una "caída libre" total (ej. precio > SMA200 preferiblemente, o RSI extremo < 25)
                        # Aquí somos permisivos con el filtro: RSI < 35
                        scan_df = scan_df[scan_df['RSI'] < 35].copy()
                        
                        # Ordenar por RSI ascendente (los más sobrevendidos primero) -> O tal vez por Score de "calidad"
                        # Vamos a ordenar por RSI ascendente para ver los más extremos
                        scan_df = scan_df.sort_values('RSI', ascending=True)
                        
                        st.success(f"🔍 Modo Reversión: {len(scan_df)} activos sobrevendidos encontrados.")
                    else:
                        st.warning("Datos de RSI no disponibles en el escaneo actual.")
                
                st.session_state['scan_results'] = scan_df

                
                # Guardar en base de datos para persistencia
                market_db.save_scanner_results(scan_df)
                st.success("✅ Resultados guardados en base de datos. Se cargarán automáticamente la próxima vez.")
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
            
            def color_weinstein(val):
                if "2" in str(val): return 'color: #28a745; font-weight: bold'
                elif "4" in str(val): return 'color: #dc3545; font-weight: bold'
                elif "1" in str(val) or "3" in str(val): return 'color: #ffc107'
                return ''

            def color_rsi(val):
                try:
                    v = float(val)
                    if v < 30: return 'color: #ffc107; font-weight: bold; background-color: rgba(255, 0, 0, 0.2)' # Sobrevendido crítico
                    elif v > 70: return 'color: #28a745; font-weight: bold' # Sobrecompra (Fuerza en momentum)
                except: pass
                return ''
            
            # Columnas a mostrar dinámicamente
            cols_to_show = ['Ticker', 'Precio', 'Score', 'RSI', 'Etapa W', '1D%', '5D%', 'Vol_Ratio', 'Sector']
            if not oversold_mode:
                # En modo normal, ocultamos RSI si se quiere simplificar, pero mejor mostrarlo siempre
                pass
            
            # Asegurar que existan las columnas
            available_cols = [c for c in cols_to_show if c in scan_df.columns]
            
            styled = scan_df[available_cols].style.map(color_score, subset=['Score'])
            styled = styled.map(color_change, subset=['1D%', '5D%'])
            styled = styled.map(color_weinstein, subset=['Etapa W'])
            if 'RSI' in available_cols:
                styled = styled.map(color_rsi, subset=['RSI'])
                
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
                        
                        # Extraer etapa de Weinstein de los resultados del scanner
                        w_stage_scan = scan_df[scan_df['Ticker'] == selected_ticker]['Etapa W'].iloc[0] if 'Etapa W' in scan_df.columns else "N/D"
                        st.markdown(f"🏢 {fundies.get('Sector', 'N/D')} | {fundies.get('Industry', 'N/D')} | **Weinstein: {w_stage_scan}**")
                        
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
                        
                        st.markdown(analysis.replace("$", "\\$"))
                        
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

    if active_tab == "📅 Economic Outlook":
        st.subheader("📅 Calendario Económico Semanal (EE.UU.)")
        st.info("💡 Resaltado en verde los eventos de HOY. Los datos pasados ayudan a entender el contexto de la semana.")
        
        cal_df = get_economic_calendar()
        if not cal_df.empty:
            cal_df = cal_df.reset_index(drop=True)
        
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

    # --- WHEEL STRATEGY TAB ---
    if active_tab == "🎡 The Wheel":
        st.markdown("""
        <div style="padding:20px; border-radius:15px; background: linear-gradient(135deg, rgba(20,40,40,0.8), rgba(10,60,30,0.6)); border-left: 8px solid #28a745; margin-bottom: 25px;">
            <h2 style="margin:0; color:#e0e0e0;">🎡 The Wheel: Generador de Rentas</h2>
            <p style="margin:5px 0 0 0; color:#bbb; font-size:0.95em;">Estrategia Cash Secured Puts. Identifica acciones de alta calidad para cobrar primas mensuales con seguridad.</p>
        </div>
        """, unsafe_allow_html=True)
        
        w_col1, w_col2, w_col3 = st.columns([2, 2, 1])
        with w_col1:
            wheel_budget = st.number_input("💰 Mi Capital Total ($)", value=10000, step=1000, help="Filtra activos que puedas cubrir con este capital.")
        with w_col2:
            wheel_max_p = st.number_input("💵 Precio Máximo Acción ($)", value=500, step=10, help="Límite máximo de precio por acción.")
        with w_col3:
            st.markdown("<br>", unsafe_allow_html=True)
            universe = get_complete_universe()
            run_wheel = st.button(f"🚀 Escanear {len(universe)} Activos", use_container_width=True, type='primary', help="Inicia escaneo multihilo")
            
        if run_wheel:
            with st.spinner(f"🔬 Escaneando el universo de {len(universe)} acciones en paralelo..."):
                # Escaneamos todo el universo (budget alto y sin filtro precio) para llenar el cache
                wheel_df = get_wheel_recommendations(999999, 500) 
                if not wheel_df.empty:
                    market_db.save_wheel_results(wheel_df)
                    st.success("✅ Escaneo completado y guardado en base de datos.")
                else:
                    st.warning("No se encontraron nuevas oportunidades en este momento.")

        # Cargar resultados (del escaneo actual o del caché)
        df_w_raw = market_db.get_wheel_cache()
        
        if not df_w_raw.empty:
            # Filtrar el caché por los parámetros actuales del usuario (capital y precio)
            df_w = df_w_raw[
                (df_w_raw['Capital Requerido'] <= wheel_budget) & 
                (df_w_raw['Precio'] <= wheel_max_p)
            ].copy().reset_index(drop=True)

            # Obtener fecha del raw ya que df_w podría estar vacío por los filtros
            last_upd = df_w_raw['Última Actualización'].iloc[0] if not df_w_raw.empty and 'Última Actualización' in df_w_raw.columns else "Desconocida"
            st.caption(f"📅 Último escaneo global: {last_upd} | {len(df_w)} activos cumplen tus filtros actuales.")
            
            if not df_w.empty:
                # Formateo de tabla
                def color_dte(val):
                    try:
                        v = float(val)
                        if v < 15: return 'color: white; background-color: rgba(220, 53, 69, 0.7); font-weight: bold'
                        elif v < 35: return 'color: #333; background-color: rgba(255, 193, 7, 0.7); font-weight: bold'
                        else: return 'color: white; background-color: rgba(40, 167, 69, 0.7); font-weight: bold'
                    except: pass
                    return ''

                def style_wheel(df):
                    s = df.style.format({
                        'Precio': '${:.2f}',
                        'Strike': '${:.2f}',
                        'Capital Requerido': '${:,.2f}',
                        'Prima': '${:.2f}',
                        'Días a Earnings': '{:.0f}'
                    }).map(lambda x: 'color: #28a745; font-weight: bold', subset=['Anualizado'])
                    
                    if 'Días a Earnings' in df.columns:
                        s = s.map(color_dte, subset=['Días a Earnings'])
                    return s

                st.dataframe(style_wheel(df_w), use_container_width=True, hide_index=True)
                
                # --- OBTENER FUERZA DE SECTORES ---
                market_ctx = get_market_context()
                sector_strength = {}
                if market_ctx and 'Sectors' in market_ctx:
                    df_sectors = market_ctx['Sectors']
                    # Crear diccionario: Nombre del Sector -> Puntuación de Fuerza
                    # Mapeo invertido porque en The Wheel tenemos nombres completos, no tickers del sector
                    for _, s_row in df_sectors.iterrows():
                        sector_strength[s_row['Name']] = s_row['Score']

                # --- LÓGICA DE SELECCIÓN AUTOMÁTICA (PROPUESTA) ---
                st.markdown("---")
                st.subheader("🏗️ Propuesta de Portafolio Diversificado")
                
                # Checkbox para Optimizar por Fuerza Sectorial
                optimize_sectors = st.checkbox(
                    "🎯 Optimizar por Fuerza Sectorial (Smart Alpha)", 
                    value=False, 
                    help="Si se activa, el algoritmo inclinará el portafolio combinando los sectores con mayor Momentum alcista + las mejores primas. Excluirá los sectores más débiles."
                )
                
                div_candidates = df_w.copy()
                
                if optimize_sectors and sector_strength:
                    # Aplicar peso combinado: Retorno Anualizado (60%) + Fuerza Sector (40%)
                    max_ann = div_candidates['annualized'].max() if div_candidates['annualized'].max() > 0 else 1
                    max_score = max(sector_strength.values()) if sector_strength.values() and max(sector_strength.values()) > 0 else 1
                    
                    def combined_score(row):
                        ann_norm = (row['annualized'] / max_ann) * 100
                        s_name = row['Sector']
                        sec_score = sector_strength.get(s_name, 0)
                        sec_norm = (sec_score / max_score) * 100 if max_score > 0 else 50
                        return (ann_norm * 0.6) + (sec_norm * 0.4)
                    
                    div_candidates['smart_score'] = div_candidates.apply(combined_score, axis=1)
                    
                    # Filtrar activos de los sectores con peor desempeño (score negativo o en el bottom 20% relativo)
                    min_acceptable_sec_score = min(max_score * 0.2, 0) if max_score > 0 else -10 # Umbral de corte
                    valid_sectors = [s_name for s_name, _score in sector_strength.items() if _score > min_acceptable_sec_score]
                    div_candidates = div_candidates[div_candidates['Sector'].isin(valid_sectors)]
                    
                    # Ordenar por el score combinado inteligente, permitiendo hasta 2 activos del mismo sector top
                    div_candidates = div_candidates.sort_values('smart_score', ascending=False)
                    
                    # Lógica de selección con inclinación
                    auto_selected = []
                    temp_cap = 0
                    proposed_rows = []
                    sector_counts = {}
                    
                    for _, row in div_candidates.iterrows():
                        sec = row['Sector']
                        if temp_cap + row['Capital Requerido'] <= wheel_budget and sector_counts.get(sec, 0) < 2:
                            auto_selected.append(row['Ticker'])
                            proposed_rows.append(row)
                            temp_cap += row['Capital Requerido']
                            sector_counts[sec] = sector_counts.get(sec, 0) + 1
                else:
                    # Lógica clásica (solo rentabilidad, 1 por sector)
                    div_candidates = div_candidates.sort_values('annualized', ascending=False).drop_duplicates('Sector')
                    auto_selected = []
                    temp_cap = 0
                    proposed_rows = []
                    for _, row in div_candidates.iterrows():
                        if temp_cap + row['Capital Requerido'] <= wheel_budget:
                            auto_selected.append(row['Ticker'])
                            proposed_rows.append(row)
                            temp_cap += row['Capital Requerido']
                
                def generate_alt_portfolio(df, budget):
                    import random
                    cands = df.sort_values('annualized', ascending=False).head(40)
                    if cands.empty: return []
                    cands = cands.sample(frac=1).reset_index(drop=True)
                    alt_sel = []
                    cap_u = 0
                    sec_c = {}
                    for _, r in cands.iterrows():
                        req = r['Capital Requerido']
                        sec = r['Sector']
                        if sec_c.get(sec, 0) >= 2: continue
                        if cap_u + req <= budget:
                            alt_sel.append(r['Ticker'])
                            cap_u += req
                            sec_c[sec] = sec_c.get(sec, 0) + 1
                    return alt_sel

                def calc_score(pdf, budget):
                    if pdf.empty: return 0
                    u = min(100, (pdf['Capital Requerido'].sum() / budget) * 100)
                    s = pdf['Sector'].value_counts().max() if not pdf.empty else 1
                    p = max(0, (s - 1) * 15)
                    r = min(100, pdf['annualized'].mean() * 200)
                    return int(max(0, min(100, (u*0.4) + (r*0.6) - p)))

                # --- SECCIÓN: PROPUESTA INICIAL (VISUAL) ---
                auto_df = df_w[df_w['Ticker'].isin(auto_selected)]
                auto_score = calc_score(auto_df, wheel_budget)
                
                if optimize_sectors:
                    st.caption(f"Selección sesgada inteligentemente hacia sectores fuertes de mercado. **⭐ Score: {auto_score}/100**")
                else:
                    st.caption(f"Selección recomendada priorizando prima y diversificación estándar. **⭐ Score: {auto_score}/100**")
                
                if proposed_rows:
                    p_cols = st.columns(len(proposed_rows))
                    for col, row in zip(p_cols, proposed_rows):
                        with col:
                            # Indicador visual si el sector es fuerte
                            sec_name = row['Sector']
                            is_strong = optimize_sectors and sector_strength.get(sec_name, 0) > 0
                            border_color = "#28a745" if not is_strong else "#00d2ff"
                            badge = '<span style="color:#00d2ff; font-size:10px;">⚡ Fuerte</span>' if is_strong else ''
                            
                            st.markdown(f"""
                            <div style="background:rgba(255,255,255,0.05); padding:10px; border-radius:10px; border-left: 4px solid {border_color}; min-height: 120px;">
                                <h4 style="margin:0; font-size:0.95em;">{row['Ticker']} {badge}</h4>
                                <p style="font-size:0.65em; color:#aaa; margin-bottom:5px;">{sec_name}</p>
                                <p style="margin:2px 0; font-size:0.8em;">Strike: <b>${row['Strike']}</b></p>
                                <p style="margin:5px 0 0 0; color:#28a745; font-weight:bold; font-size:0.85em;">{row['Anualizado']}</p>
                            </div>
                            """, unsafe_allow_html=True)
                    
                    st.markdown("<br>", unsafe_allow_html=True)
                    b_col1, b_col2 = st.columns(2)
                    with b_col1:
                        if st.button("🔄 Restablecer a Selección", use_container_width=True):
                            st.session_state['wheel_multi_selection_v2'] = auto_selected
                            st.session_state['wheel_ai_report'] = None
                            st.rerun()
                    with b_col2:
                        if st.button("🎲 Proponer Otra Combinación", use_container_width=True, type='primary'):
                            st.session_state['wheel_multi_selection_v2'] = generate_alt_portfolio(df_w, wheel_budget)
                            st.session_state['wheel_ai_report'] = None
                            st.rerun()
                
                # --- INTERACTOR DE PORTAFOLIO ---
                st.markdown("### 🛠️ Personalizar Selección")
                
                # Inicializar el widget si no existe
                if 'wheel_multi_selection_v2' not in st.session_state:
                    st.session_state['wheel_multi_selection_v2'] = auto_selected

                # Validación de tickers existentes en el widget state
                current_sel = st.session_state['wheel_multi_selection_v2']
                valid_options = df_w['Ticker'].tolist()
                st.session_state['wheel_multi_selection_v2'] = [t for t in current_sel if t in valid_options]

                selected_tickers = st.multiselect(
                    "Agrega o quita acciones de tu lista:",
                    options=valid_options,
                    key="wheel_multi_selection_v2"
                )
                
                # Filtrar DF por selección
                portfolio_df = df_w[df_w['Ticker'].isin(selected_tickers)].copy().reset_index(drop=True)
                total_used = portfolio_df['Capital Requerido'].sum()
                remaining = wheel_budget - total_used
                
                # Métricas de Portafolio
                st.markdown("<br>", unsafe_allow_html=True)
                m_col1, m_col2, m_col3, m_col4 = st.columns(4)
                m_col1.metric("Capital Utilizado", f"${total_used:,.0f}", delta=f"{ (total_used/wheel_budget)*100:.1f}%", delta_color="normal")
                m_col2.metric("Disponible", f"${remaining:,.0f}", delta=f"${wheel_budget:,.0f} Total", delta_color="off")
                m_col3.metric("Prima Mensual Est.", f"${portfolio_df['Prima'].sum()*100:,.2f}")
                
                curr_score = calc_score(portfolio_df, wheel_budget)
                sc_delta = "Óptimo 🚀" if curr_score >= 80 else "Aceptable" if curr_score >= 50 else "Mejorable"
                sc_color = "normal" if curr_score >= 80 else "off" if curr_score >= 50 else "inverse"
                m_col4.metric("⭐ Score Portafolio", f"{curr_score}/100", delta=sc_delta, delta_color=sc_color)
                
                # Visualización Detallada del Portafolio Actual
                if not portfolio_df.empty:
                    st.write("📋 **Tu Selección Actual:**")
                    disp_cols = ['Ticker', 'Sector', 'Strike', 'Prima', 'Anualizado', 'Capital Requerido']
                    if 'Días a Earnings' in portfolio_df.columns:
                        disp_cols.append('Días a Earnings')
                        styled_p = portfolio_df[disp_cols].style.map(color_dte, subset=['Días a Earnings']).format({'Días a Earnings': '{:.0f}', 'Strike': '${:.2f}', 'Prima': '${:.2f}', 'Capital Requerido': '${:,.2f}'})
                        st.dataframe(styled_p, use_container_width=True, hide_index=True)
                    else:
                        st.dataframe(portfolio_df[disp_cols], use_container_width=True, hide_index=True)
                    
                    if remaining < 0:
                        st.error(f"⚠️ ¡Has superado tu capital por ${abs(remaining):,.0f}! Revisa tu selección.")
                    
                    st.markdown("---")
                    
                    # Persistencia del Análisis IA
                    if 'wheel_ai_report' not in st.session_state:
                        st.session_state['wheel_ai_report'] = None

                    ai_col1, ai_col2 = st.columns([4, 1])
                    with ai_col1:
                        if st.button("🧠 Análisis de Salud del Portafolio (IA)", use_container_width=True, type='primary'):
                            if groq_api_key:
                                with st.spinner("📊 Calculando correlaciones y betas del portafolio..."):
                                    corr_data = calculate_portfolio_correlation(selected_tickers)
                                with st.spinner("🧠 Generando análisis estratégico con IA..."):
                                    report = ai_wheel_portfolio_analysis(groq_api_key, portfolio_df, wheel_budget, correlation_data=corr_data)
                                    st.session_state['wheel_ai_report'] = report
                            else:
                                st.warning("Configura la API Key para este análisis.")
                    
                    with ai_col2:
                        if st.session_state['wheel_ai_report'] and st.button("🗑️ Limpiar", use_container_width=True):
                            st.session_state['wheel_ai_report'] = None
                            st.rerun()

                    if st.session_state['wheel_ai_report']:
                        st.markdown(st.session_state['wheel_ai_report'])
                else:
                    st.info("La lista está vacía. Selecciona activos arriba para empezar.")
            else:
                st.warning("Ninguno de los activos en el historial cumple con tu capital o precio máximo. Inicia un nuevo escaneo.")
        else:
            total_assets = len(get_complete_universe())
            st.info(f"👋 Bienvenida/o al Escáner de la Rueda. Haz clic en **Escaneas {total_assets} Activos** para construir tu primera base de datos de oportunidades.")


    # --- DEEP DIVE TAB ---
    if active_tab == "🔍 Strategic Analysis":
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
            w_stage = dd_tech.get('weinstein', 'N/D') if dd_tech else 'N/D'
            w_color = "#28a745" if "2" in str(w_stage) else "#dc3545" if "4" in str(w_stage) else "#ffc107"
            w_badge = f'<span style="background:{w_color}; color:white; padding: 4px 12px; border-radius: 20px; font-size: 0.55em; margin-left: 10px; vertical-align: middle; display: inline-block; white-space: nowrap;">Etapa Weinstein: {w_stage}</span>' if w_stage != 'N/D' else ''

            st.markdown(f"""
            <div style="padding:20px; border-radius:12px; background: rgba(0,0,0,0.3); margin-bottom:15px; border: 1px solid rgba(255,255,255,0.05);">
                <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap: 15px;">
                    <div style="flex: 1; min-width: 300px;">
                        <h2 style="margin:0; color:white; line-height: 1.2;">{g['shortName']} ({dd_ticker}) {w_badge}</h2>
                        <p style="margin:8px 0 0 0; color:#aaa; font-size:0.9em;">{g['sector']} | {g['industry']} | {g['country']}</p>
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
            
            # FILA 1: VALORACIÓN ESTRUCTURADA
            v_data = dd_fin['valuation']
            mk_col1, mk_col2, mk_col3, mk_col4, mk_col5, mk_col6 = st.columns(6)
            
            # P/E con lógica inteligente (priorizando Trailing positivo)
            trailing_pe = v_data.get('trailingPE')
            has_trailing = trailing_pe is not None and trailing_pe > 0
            
            pe_val = trailing_pe if has_trailing else v_data.get('forwardPE')
            pe_label = "P/E (Trailing)" if has_trailing else "P/E (Forward)"
            if pe_val is None: pe_label = "P/E Ratio"

            mk_col1.metric(pe_label, fmt_v(pe_val, suffix='x'))
            mk_col2.metric("PEG Ratio", fmt_v(v_data['pegRatio']))
            mk_col3.metric("P/Sales", fmt_v(v_data['priceToSales'], suffix='x'))
            mk_col4.metric("P/Book", fmt_v(v_data['priceToBook'], suffix='x'))
            mk_col5.metric("Market Cap", fmt_b(g['marketCap']))
            mk_col6.metric("Enterprise Val.", fmt_b(g['enterpriseValue']))
            
            # FILA 2: RENDIMIENTO Y SOLVENCIA
            mk2_col1, mk2_col2, mk2_col3, mk2_col4, mk2_col5, mk2_col6 = st.columns(6)
            mk2_col1.metric("ROE", fmt_v(dd_fin['profitability']['returnOnEquity'], pct=True))
            mk2_col2.metric("Margen Neto", fmt_v(dd_fin['profitability']['profitMargin'], pct=True))
            mk2_col3.metric("Crec. Ingresos", fmt_v(dd_fin['growth']['revenueGrowth'], pct=True))
            mk2_col4.metric("Deuda/Equity", fmt_v(dd_fin['solvency']['debtToEquity']))
            mk2_col5.metric("Current Ratio", fmt_v(dd_fin['solvency']['currentRatio']))
            mk2_col6.metric("Free Cash Flow", fmt_b(dd_fin['solvency']['freeCashflow']))
            
            # FILA 3: RIESGO Y ANALISTAS
            st.markdown("<br>", unsafe_allow_html=True)
            mk3_col1, mk3_col2, mk3_col3, mk3_col4, mk3_col5, mk3_col6 = st.columns(6)
            mk3_col1.metric("Beta (5Y)", fmt_v(dd_fin['risk']['beta']))
            mk3_col2.metric("Short Ratio", fmt_v(dd_fin['risk']['shortRatio']))
            mk3_col3.metric("Div. Yield", fmt_v(dd_fin['dividends']['dividendYield'], pct=True))
            mk3_col4.metric("EPS Forward", fmt_v(dd_fin['analyst']['forwardEps']))
            mk3_col5.metric("Opiniones", dd_fin['analyst']['numberOfAnalystOpinions'])
            mk3_col6.metric("EV/EBITDA", fmt_v(v_data['evToEbitda'], suffix='x'))
            
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
                
                # --- NUEVA SECCIÓN: ETAPA DE WEINSTEIN ---
                w_stage = dd_tech.get('weinstein', 'N/D')
                if w_stage != 'N/D':
                    w_color = "#28a745" if "2" in w_stage else "#dc3545" if "4" in w_stage else "#ffc107"
                    st.markdown(f"""
                    <div style="background: rgba(0,0,0,0.2); border: 1px solid {w_color}; padding: 12px; border-radius: 10px; margin-bottom: 15px; display: flex; align-items: center; gap: 10px;">
                        <span style="font-size: 1.5em;">📊</span>
                        <div>
                            <p style="margin:0; font-size: 0.8em; color: #aaa;">Etapa de Weinstein (Ciclo de Largo Plazo):</p>
                            <p style="margin:0; font-size: 1.1em; font-weight: bold; color: {w_color};">{w_stage}</p>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

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
                summary = g.get('summary', 'Sin descripción disponible.')
                
                # Sistema de traducción on-demand
                summary_key = f"summary_es_{dd_ticker}"
                if summary_key in st.session_state:
                    st.info("✅ Traducción al español habilitada")
                    st.write(st.session_state[summary_key])
                else:
                    st.write(summary)
                    if groq_api_key and summary != 'Sin descripción disponible.':
                        if st.button("🌐 Traducir al Español (IA)", key=f"btn_trans_{dd_ticker}"):
                            with st.spinner("Traduciendo..."):
                                translated = translate_text(groq_api_key, summary)
                                st.session_state[summary_key] = translated
                                st.rerun()

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


    if active_tab == "📜 History":
        # Centrar contenido si es necesario
        st.subheader("📚 Centro de Registro y Bitácora")
        
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
        



    st.markdown("---")
    st.caption("⚠️ Advertencia: Plataforma de análisis algorítmico. No constituye asesoría financiera directa.")

if __name__ == '__main__':
    main()
