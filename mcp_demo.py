#!/usr/bin/env python3
"""
Demostración del Alpha Vantage MCP Server
Muestra cómo funciona cada herramienta disponible
"""

import asyncio
import sys
from alpha_vantage_mcp_server import (
    get_stock_quote,
    get_daily_prices,
    get_company_overview,
    get_news_sentiment,
    get_technical_indicators
)

async def demo_all_tools():
    """Demuestra todas las herramientas del servidor MCP"""
    
    print("🚀 DEMOSTRACIÓN DEL ALPHA VANTAGE MCP SERVER")
    print("=" * 60)
    
    # 1. Cotización en tiempo real
    print("\n1️⃣ COTIZACIÓN EN TIEMPO REAL (AAPL)")
    print("-" * 40)
    quote_result = await get_stock_quote("AAPL")
    print(quote_result)
    
    # 2. Precios históricos
    print("\n2️⃣ PRECIOS HISTÓRICOS DIARIOS (MSFT)")
    print("-" * 40)
    daily_result = await get_daily_prices("MSFT", "compact")
    print(daily_result[:500] + "..." if len(daily_result) > 500 else daily_result)
    
    # 3. Información de empresa
    print("\n3️⃣ INFORMACIÓN DE EMPRESA (GOOGL)")
    print("-" * 40)
    company_result = await get_company_overview("GOOGL")
    print(company_result[:600] + "..." if len(company_result) > 600 else company_result)
    
    # 4. Análisis de sentimiento de noticias
    print("\n4️⃣ ANÁLISIS DE SENTIMIENTO DE NOTICIAS (TSLA)")
    print("-" * 40)
    news_result = await get_news_sentiment("TSLA")
    print(news_result[:700] + "..." if len(news_result) > 700 else news_result)
    
    # 5. Indicadores técnicos
    print("\n5️⃣ INDICADORES TÉCNICOS - RSI (NVDA)")
    print("-" * 40)
    tech_result = await get_technical_indicators("NVDA", "RSI", 14)
    print(tech_result)
    
    print("\n" + "=" * 60)
    print("✅ DEMOSTRACIÓN COMPLETADA")
    print("\n💡 CÓMO USAR EN TRAE AI:")
    print("1. El servidor MCP se ejecuta en segundo plano")
    print("2. Trae AI puede llamar automáticamente a estas herramientas")
    print("3. Solo pregunta: '¿Cuál es el precio de AAPL?' y obtienes la respuesta")
    print("4. O: '¿Qué noticias hay sobre TSLA?' para análisis de sentimiento")

if __name__ == "__main__":
    # Ejecutar la demostración
    asyncio.run(demo_all_tools())