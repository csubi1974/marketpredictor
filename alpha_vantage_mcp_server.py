#!/usr/bin/env python3
"""
Alpha Vantage MCP Server for Trae AI IDE
Provides financial data tools through Model Context Protocol
"""

import os
import asyncio
from typing import Any, Dict, List
import httpx
from mcp.server.fastmcp import FastMCP
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize FastMCP server
mcp = FastMCP("alpha-vantage-financial")

# Constants
BASE_URL = "https://www.alphavantage.co/query"
API_KEY = "Z4LODLV2DNPLO3ED"  # Your provided API key

async def call_alpha_vantage(function: str, params: Dict[str, Any]) -> Dict[str, Any] | None:
    """Generic async caller to Alpha Vantage API."""
    all_params = {
        "function": function,
        "apikey": API_KEY,
        **params
    }
    
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(BASE_URL, params=all_params, timeout=30.0)
            response.raise_for_status()
            data = response.json()
            
            # Check for API error messages
            if "Error Message" in data:
                return {"error": data["Error Message"]}
            if "Note" in data:
                return {"error": f"API Limit: {data['Note']}"}
                
            return data
        except Exception as e:
            return {"error": f"Request failed: {str(e)}"}

@mcp.tool()
async def get_stock_quote(symbol: str) -> str:
    """Get real-time stock quote for a given symbol.
    
    Args:
        symbol: Stock ticker symbol (e.g., AAPL, MSFT, TSLA)
    """
    data = await call_alpha_vantage("GLOBAL_QUOTE", {"symbol": symbol.upper()})
    
    if not data or "error" in data:
        return f"❌ Error getting quote for {symbol}: {data.get('error', 'Unknown error')}"
    
    quote = data.get("Global Quote", {})
    if not quote:
        return f"❌ No quote data found for {symbol}"
    
    return f"""📈 **{symbol.upper()} Stock Quote**
💰 Price: ${quote.get('05. price', 'N/A')}
📊 Change: {quote.get('09. change', 'N/A')} ({quote.get('10. change percent', 'N/A')})
📅 Last Updated: {quote.get('07. latest trading day', 'N/A')}
🔼 High: ${quote.get('03. high', 'N/A')}
🔽 Low: ${quote.get('04. low', 'N/A')}
📦 Volume: {quote.get('06. volume', 'N/A')}"""

@mcp.tool()
async def get_daily_prices(symbol: str, outputsize: str = "compact") -> str:
    """Get daily historical prices for a stock.
    
    Args:
        symbol: Stock ticker symbol (e.g., AAPL, MSFT)
        outputsize: 'compact' (last 100 days) or 'full' (20+ years)
    """
    data = await call_alpha_vantage("TIME_SERIES_DAILY", {
        "symbol": symbol.upper(),
        "outputsize": outputsize
    })
    
    if not data or "error" in data:
        return f"❌ Error getting daily prices for {symbol}: {data.get('error', 'Unknown error')}"
    
    time_series = data.get("Time Series (Daily)", {})
    if not time_series:
        return f"❌ No daily price data found for {symbol}"
    
    # Get the last 5 trading days
    recent_dates = sorted(time_series.keys(), reverse=True)[:5]
    
    result = f"📊 **{symbol.upper()} - Last 5 Trading Days**\n\n"
    
    for date in recent_dates:
        day_data = time_series[date]
        result += f"📅 **{date}**\n"
        result += f"   🔓 Open: ${day_data.get('1. open', 'N/A')}\n"
        result += f"   🔒 Close: ${day_data.get('4. close', 'N/A')}\n"
        result += f"   🔼 High: ${day_data.get('2. high', 'N/A')}\n"
        result += f"   🔽 Low: ${day_data.get('3. low', 'N/A')}\n"
        result += f"   📦 Volume: {day_data.get('5. volume', 'N/A')}\n\n"
    
    return result

@mcp.tool()
async def get_company_overview(symbol: str) -> str:
    """Get comprehensive company information and financial metrics.
    
    Args:
        symbol: Stock ticker symbol (e.g., AAPL, MSFT)
    """
    data = await call_alpha_vantage("OVERVIEW", {"symbol": symbol.upper()})
    
    if not data or "error" in data:
        return f"❌ Error getting company overview for {symbol}: {data.get('error', 'Unknown error')}"
    
    if not data.get("Symbol"):
        return f"❌ No company data found for {symbol}"
    
    return f"""🏢 **{data.get('Name', 'N/A')} ({data.get('Symbol', 'N/A')})**

📋 **Basic Info:**
• Industry: {data.get('Industry', 'N/A')}
• Sector: {data.get('Sector', 'N/A')}
• Country: {data.get('Country', 'N/A')}
• Exchange: {data.get('Exchange', 'N/A')}

💰 **Financial Metrics:**
• Market Cap: ${data.get('MarketCapitalization', 'N/A')}
• P/E Ratio: {data.get('PERatio', 'N/A')}
• EPS: ${data.get('EPS', 'N/A')}
• Revenue (TTM): ${data.get('RevenueTTM', 'N/A')}
• Profit Margin: {data.get('ProfitMargin', 'N/A')}

📊 **Trading Info:**
• 52 Week High: ${data.get('52WeekHigh', 'N/A')}
• 52 Week Low: ${data.get('52WeekLow', 'N/A')}
• Beta: {data.get('Beta', 'N/A')}
• Dividend Yield: {data.get('DividendYield', 'N/A')}

📝 **Description:**
{data.get('Description', 'No description available')[:300]}..."""

@mcp.tool()
async def get_news_sentiment(symbol: str) -> str:
    """Get news sentiment analysis for a stock.
    
    Args:
        symbol: Stock ticker symbol (e.g., AAPL, MSFT)
    """
    data = await call_alpha_vantage("NEWS_SENTIMENT", {"tickers": symbol.upper()})
    
    if not data or "error" in data:
        return f"❌ Error getting news sentiment for {symbol}: {data.get('error', 'Unknown error')}"
    
    feed = data.get("feed", [])
    if not feed:
        return f"❌ No news sentiment data found for {symbol}"
    
    # Get the top 3 most recent articles
    recent_articles = feed[:3]
    
    result = f"📰 **News Sentiment for {symbol.upper()}**\n\n"
    
    for i, article in enumerate(recent_articles, 1):
        sentiment_score = article.get("overall_sentiment_score", 0)
        sentiment_label = article.get("overall_sentiment_label", "Neutral")
        
        # Convert sentiment score to emoji
        if sentiment_score > 0.15:
            sentiment_emoji = "🟢"
        elif sentiment_score < -0.15:
            sentiment_emoji = "🔴"
        else:
            sentiment_emoji = "🟡"
        
        result += f"**{i}. {article.get('title', 'No title')[:80]}...**\n"
        result += f"   {sentiment_emoji} Sentiment: {sentiment_label} ({sentiment_score:.3f})\n"
        result += f"   📅 Published: {article.get('time_published', 'N/A')}\n"
        result += f"   📰 Source: {article.get('source', 'N/A')}\n"
        result += f"   🔗 URL: {article.get('url', 'N/A')}\n\n"
    
    return result

@mcp.tool()
async def get_technical_indicators(symbol: str, indicator: str = "RSI", time_period: int = 14) -> str:
    """Get technical indicators for a stock.
    
    Args:
        symbol: Stock ticker symbol (e.g., AAPL, MSFT)
        indicator: Technical indicator (RSI, SMA, EMA, MACD, etc.)
        time_period: Time period for the indicator (default: 14)
    """
    data = await call_alpha_vantage(indicator.upper(), {
        "symbol": symbol.upper(),
        "interval": "daily",
        "time_period": str(time_period),
        "series_type": "close"
    })
    
    if not data or "error" in data:
        return f"❌ Error getting {indicator} for {symbol}: {data.get('error', 'Unknown error')}"
    
    # Find the technical analysis data key
    tech_key = None
    for key in data.keys():
        if "Technical Analysis" in key:
            tech_key = key
            break
    
    if not tech_key or not data.get(tech_key):
        return f"❌ No {indicator} data found for {symbol}"
    
    tech_data = data[tech_key]
    recent_dates = sorted(tech_data.keys(), reverse=True)[:5]
    
    result = f"📈 **{indicator.upper()} for {symbol.upper()} (Period: {time_period})**\n\n"
    
    for date in recent_dates:
        day_data = tech_data[date]
        # Get the first value (usually the indicator value)
        indicator_value = list(day_data.values())[0] if day_data else "N/A"
        result += f"📅 {date}: {indicator_value}\n"
    
    return result

@mcp.resource("config://alpha-vantage")
def get_config() -> str:
    """Get Alpha Vantage MCP server configuration."""
    return """Alpha Vantage MCP Server Configuration:
- API Key: Configured ✅
- Available Tools:
  • get_stock_quote: Real-time stock quotes
  • get_daily_prices: Historical daily prices
  • get_company_overview: Company information and metrics
  • get_news_sentiment: News sentiment analysis
  • get_technical_indicators: Technical analysis indicators
- Rate Limits: 25 requests per day (free tier)
- Supported Markets: Global stock markets
"""

if __name__ == "__main__":
    # Run the MCP server
    mcp.run(transport="stdio")