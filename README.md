# 📊 Market Predictor - AI Trading Control Tower

Una plataforma avanzada de análisis y predicción de mercados financieros impulsada por Machine Learning e Inteligencia Artificial.

## 🚀 Características

### 🤖 Motor de IA Avanzado
- **Ensemble Learning**: Comité de modelos (XGBoost + Random Forest + Gradient Boosting)
- **Predicción Direccional**: Alcista, Bajista o Neutral con niveles de confianza
- **Adaptación Dinámica**: Ajuste automático de umbrales según volatilidad (ADX)

### 📈 Análisis Técnico Completo
- 18+ indicadores técnicos e intermarket
- Análisis de correlación global (VIX, Futuros, Nikkei, DAX, DXY)
- Detección de riesgo sistémico (Crash Risk Analyzer)

### ⚡ Monitoreo en Tiempo Real
- **Sniper Monitor**: Momentum intradía con datos de 5 minutos
- **VIX Heartbeat**: Latido del miedo en tiempo real
- **Options Radar**: Análisis de sentimiento y Gamma Exposure (Call/Put Walls)

### 🧠 Copiloto Estratégico (LLM)
- Análisis táctico automatizado vía Groq (Llama 3.3)
- Briefing pre-mercado con síntesis de noticias
- Interpretación de contexto macro y técnico

## 🛠️ Tecnologías

- **Backend**: Python 3.9+
- **ML/AI**: scikit-learn, XGBoost, Groq API
- **Data**: yfinance (Yahoo Finance)
- **Frontend**: Streamlit
- **Visualización**: Plotly
- **Deployment**: Docker, EasyPanel

## 📦 Instalación Local

### Requisitos
- Python 3.9 o superior
- pip

### Pasos

1. **Clonar el repositorio**
```bash
git clone https://github.com/csubi1974/marketpredictor.git
cd marketpredictor
```

2. **Instalar dependencias**
```bash
pip install -r requirements.txt
```

3. **Configurar variables de entorno**
Crea un archivo `.env` en la raíz del proyecto:
```env
GROQ_API_KEY=tu_api_key_aqui
```

> 🔑 Obtén tu API key gratuita en [Groq Console](https://console.groq.com)

4. **Ejecutar la aplicación**
```bash
streamlit run app.py
```

La aplicación estará disponible en `http://localhost:8501`

## 🐳 Deployment con Docker

### Build
```bash
docker build -t marketpredictor .
```

### Run
```bash
docker run -p 8501:8501 -e GROQ_API_KEY=tu_api_key marketpredictor
```

## ☁️ Deployment en EasyPanel

1. Conecta este repositorio en EasyPanel
2. Configura las variables de entorno:
   - `GROQ_API_KEY`: Tu API key de Groq
3. Puerto: `8501`
4. Deploy 🚀

## 📊 Uso

### 1. Entrenar el Modelo
- Selecciona un activo del menú lateral
- Ajusta el rango de fechas (recomendado: 5 años)
- Click en **"Entrenar Modelo"**

### 2. Obtener Predicción
- Una vez entrenado, el modelo predice automáticamente
- Revisa el **Market Desk** para ver:
  - Predicción direccional
  - Nivel de confianza
  - Crash Risk
  - Momentum intradía

### 3. Análisis Avanzado
- **Informe Táctico**: Análisis LLM del contexto actual
- **Briefing Pre-Mercado**: Síntesis de noticias + técnico
- **Backtesting**: Prueba el modelo en fechas históricas

## 🎯 Activos Soportados

- Índices: S&P 500, Nasdaq 100, Dow Jones
- ETFs: SPY, QQQ, DIA
- Acciones: AAPL, MSFT, AMZN, TSLA, GOOGL, NVDA, META, y más

## ⚠️ Disclaimer

Esta herramienta es solo para fines educativos e informativos. **No constituye asesoramiento financiero**. Las predicciones de Machine Learning no garantizan resultados futuros. Siempre realiza tu propia investigación antes de tomar decisiones de inversión.

## 📝 Licencia

MIT License - Ver archivo LICENSE para más detalles

## 👨‍💻 Autor

**Cristian Subiaurre**
- GitHub: [@csubi1974](https://github.com/csubi1974)

---

⭐ Si te resulta útil, considera darle una estrella al repo!
