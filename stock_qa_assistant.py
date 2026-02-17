import requests
import json
import re
from datetime import datetime, timedelta
import pandas as pd

class StockQAAssistant:
    """
    Asistente de preguntas y respuestas financiero que utiliza Alpha Vantage API
    para responder preguntas sobre acciones, empresas y mercados financieros.
    """
    
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "https://www.alphavantage.co/query"
        
    def get_stock_quote(self, symbol):
        """Obtiene cotización actual de una acción"""
        params = {
            'function': 'GLOBAL_QUOTE',
            'symbol': symbol,
            'apikey': self.api_key
        }
        
        try:
            response = requests.get(self.base_url, params=params, timeout=30)
            data = response.json()
            
            if "Global Quote" in data:
                quote = data["Global Quote"]
                return {
                    'symbol': quote.get('01. symbol', ''),
                    'price': float(quote.get('05. price', 0)),
                    'change': float(quote.get('09. change', 0)),
                    'change_percent': quote.get('10. change percent', ''),
                    'high': float(quote.get('03. high', 0)),
                    'low': float(quote.get('04. low', 0)),
                    'volume': int(quote.get('06. volume', 0)),
                    'latest_trading_day': quote.get('07. latest trading day', '')
                }
        except Exception as e:
            print(f"Error obteniendo cotización: {e}")
        
        return None
    
    def get_company_overview(self, symbol):
        """Obtiene información general de la empresa"""
        params = {
            'function': 'OVERVIEW',
            'symbol': symbol,
            'apikey': self.api_key
        }
        
        try:
            response = requests.get(self.base_url, params=params, timeout=30)
            data = response.json()
            
            if data and 'Symbol' in data:
                return data
        except Exception as e:
            print(f"Error obteniendo información de empresa: {e}")
        
        return None
    
    def get_news_sentiment(self, symbol):
        """Obtiene noticias y análisis de sentimiento"""
        params = {
            'function': 'NEWS_SENTIMENT',
            'tickers': symbol,
            'apikey': self.api_key
        }
        
        try:
            response = requests.get(self.base_url, params=params, timeout=30)
            data = response.json()
            
            if 'feed' in data:
                return data['feed'][:5]  # Últimas 5 noticias
        except Exception as e:
            print(f"Error obteniendo noticias: {e}")
        
        return []
    
    def get_technical_indicators(self, symbol, indicator='RSI', time_period=14):
        """Obtiene indicadores técnicos"""
        params = {
            'function': indicator,
            'symbol': symbol,
            'interval': 'daily',
            'time_period': time_period,
            'series_type': 'close',
            'apikey': self.api_key
        }
        
        try:
            response = requests.get(self.base_url, params=params, timeout=30)
            data = response.json()
            
            technical_key = f"Technical Analysis: {indicator}"
            if technical_key in data:
                return data[technical_key]
        except Exception as e:
            print(f"Error obteniendo indicadores técnicos: {e}")
        
        return None
    
    def interpret_question(self, question):
        """Interpreta la pregunta del usuario y determina qué información necesita"""
        question_lower = question.lower()
        
        # Extraer símbolo de acción si está presente
        symbol_pattern = r'\b[A-Z]{1,5}\b'
        symbols = re.findall(symbol_pattern, question.upper())
        
        # Determinar tipo de pregunta
        question_type = 'general'
        
        if any(word in question_lower for word in ['precio', 'cotización', 'valor', 'cuesta', 'vale']):
            question_type = 'price'
        elif any(word in question_lower for word in ['empresa', 'compañía', 'información', 'sector', 'industria']):
            question_type = 'company'
        elif any(word in question_lower for word in ['noticias', 'news', 'últimas', 'recientes']):
            question_type = 'news'
        elif any(word in question_lower for word in ['rsi', 'indicador', 'técnico', 'análisis']):
            question_type = 'technical'
        elif any(word in question_lower for word in ['comparar', 'vs', 'versus', 'mejor']):
            question_type = 'comparison'
        
        return {
            'type': question_type,
            'symbols': symbols,
            'original_question': question
        }
    
    def answer_question(self, question):
        """Responde a la pregunta del usuario"""
        interpretation = self.interpret_question(question)
        question_type = interpretation['type']
        symbols = interpretation['symbols']
        
        if not symbols:
            return "Por favor, especifica el símbolo de la acción (ej: AAPL, GOOGL, TSLA) en tu pregunta."
        
        symbol = symbols[0]  # Usar el primer símbolo encontrado
        
        try:
            if question_type == 'price':
                return self._answer_price_question(symbol)
            elif question_type == 'company':
                return self._answer_company_question(symbol)
            elif question_type == 'news':
                return self._answer_news_question(symbol)
            elif question_type == 'technical':
                return self._answer_technical_question(symbol)
            elif question_type == 'comparison':
                if len(symbols) >= 2:
                    return self._answer_comparison_question(symbols[:2])
                else:
                    return f"Para comparar acciones, necesito al menos dos símbolos. Solo encontré: {symbol}"
            else:
                return self._answer_general_question(symbol)
                
        except Exception as e:
            return f"Lo siento, ocurrió un error al procesar tu pregunta: {str(e)}"
    
    def _answer_price_question(self, symbol):
        """Responde preguntas sobre precio"""
        quote = self.get_stock_quote(symbol)
        if not quote:
            return f"No pude obtener información de precio para {symbol}."
        
        change_direction = "subió" if quote['change'] > 0 else "bajó"
        
        return f"""
**Precio actual de {symbol}:**
• **Precio:** ${quote['price']:.2f}
• **Cambio:** ${quote['change']:.2f} ({quote['change_percent']}) - {change_direction}
• **Rango del día:** ${quote['low']:.2f} - ${quote['high']:.2f}
• **Volumen:** {quote['volume']:,}
• **Última actualización:** {quote['latest_trading_day']}
        """
    
    def _answer_company_question(self, symbol):
        """Responde preguntas sobre la empresa"""
        company = self.get_company_overview(symbol)
        if not company:
            return f"No pude obtener información de la empresa para {symbol}."
        
        return f"""
**Información de {company.get('Name', symbol)}:**
• **Sector:** {company.get('Sector', 'N/A')}
• **Industria:** {company.get('Industry', 'N/A')}
• **País:** {company.get('Country', 'N/A')}
• **Capitalización de mercado:** ${company.get('MarketCapitalization', 'N/A')}
• **P/E Ratio:** {company.get('PERatio', 'N/A')}
• **Dividend Yield:** {company.get('DividendYield', 'N/A')}
• **52 Week High:** ${company.get('52WeekHigh', 'N/A')}
• **52 Week Low:** ${company.get('52WeekLow', 'N/A')}

**Descripción:** {company.get('Description', 'No disponible')[:300]}...
        """
    
    def _answer_news_question(self, symbol):
        """Responde preguntas sobre noticias"""
        news = self.get_news_sentiment(symbol)
        if not news:
            return f"No pude obtener noticias recientes para {symbol}."
        
        response = f"**Últimas noticias de {symbol}:**\n\n"
        
        for i, article in enumerate(news[:3], 1):
            sentiment = article.get('overall_sentiment_label', 'Neutral')
            response += f"""
**{i}. {article.get('title', 'Sin título')}**
• **Fuente:** {article.get('source', 'N/A')}
• **Fecha:** {article.get('time_published', 'N/A')[:8]}
• **Sentimiento:** {sentiment}
• **URL:** {article.get('url', 'N/A')}

"""
        
        return response
    
    def _answer_technical_question(self, symbol):
        """Responde preguntas sobre análisis técnico"""
        rsi_data = self.get_technical_indicators(symbol, 'RSI')
        if not rsi_data:
            return f"No pude obtener indicadores técnicos para {symbol}."
        
        # Obtener el RSI más reciente
        latest_date = max(rsi_data.keys())
        latest_rsi = float(rsi_data[latest_date]['RSI'])
        
        # Interpretación del RSI
        if latest_rsi > 70:
            interpretation = "Sobrecomprado - Posible corrección a la baja"
        elif latest_rsi < 30:
            interpretation = "Sobrevendido - Posible rebote al alza"
        else:
            interpretation = "Neutral - Sin señales extremas"
        
        return f"""
**Análisis técnico de {symbol}:**
• **RSI actual:** {latest_rsi:.2f}
• **Interpretación:** {interpretation}
• **Fecha:** {latest_date}

**Guía de RSI:**
• RSI > 70: Sobrecomprado
• RSI < 30: Sobrevendido
• RSI 30-70: Rango neutral
        """
    
    def _answer_comparison_question(self, symbols):
        """Responde preguntas de comparación entre acciones"""
        comparisons = []
        
        for symbol in symbols:
            quote = self.get_stock_quote(symbol)
            if quote:
                comparisons.append({
                    'symbol': symbol,
                    'price': quote['price'],
                    'change_percent': quote['change_percent']
                })
        
        if len(comparisons) < 2:
            return "No pude obtener datos suficientes para hacer la comparación."
        
        response = "**Comparación de acciones:**\n\n"
        
        for comp in comparisons:
            response += f"• **{comp['symbol']}:** ${comp['price']:.2f} ({comp['change_percent']})\n"
        
        # Determinar cuál tiene mejor rendimiento
        best_performer = max(comparisons, key=lambda x: float(x['change_percent'].replace('%', '')))
        response += f"\n**Mejor rendimiento hoy:** {best_performer['symbol']} ({best_performer['change_percent']})"
        
        return response
    
    def _answer_general_question(self, symbol):
        """Responde preguntas generales combinando información"""
        quote = self.get_stock_quote(symbol)
        company = self.get_company_overview(symbol)
        
        if not quote and not company:
            return f"No pude obtener información para {symbol}."
        
        response = f"**Resumen general de {symbol}:**\n\n"
        
        if quote:
            response += f"**Precio actual:** ${quote['price']:.2f} ({quote['change_percent']})\n"
        
        if company:
            response += f"**Empresa:** {company.get('Name', symbol)}\n"
            response += f"**Sector:** {company.get('Sector', 'N/A')}\n"
            response += f"**Capitalización:** ${company.get('MarketCapitalization', 'N/A')}\n"
        
        return response