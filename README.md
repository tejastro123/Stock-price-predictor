# 📈 Stock Price Prediction System

An end-to-end **Machine Learning + MERN Stack** project for predicting stock prices. Built as a final year student project demonstrating ML pipeline design, model comparison, and full-stack web deployment.

---

## 🏗️ Architecture

```bash
                    ┌─────────────────┐
                    │   React (Vite)  │  ← Frontend (port 5173)
                    │   StockSage AI  │
                    └────────┬────────┘
                             │  HTTP
                    ┌────────▼────────┐
                    │    Express.js   │  ← Node Backend (port 5000)
                    │   + MongoDB     │     Proxy + Prediction Logging
                    └────────┬────────┘
                             │  HTTP
                    ┌────────▼────────┐
                    │     FastAPI     │  ← ML API (port 8000)
                    │   + sklearn     │     Train / Predict / History
                    └─────────────────┘
```

## 📦 Project Structure

```bash
stock-price-predictor/
├── data/                        # Cached stock CSV data
├── models/                      # Saved model artifacts (.pkl)
├── notebooks/
│   └── Stock_Price_Prediction.ipynb  # Complete ML pipeline notebook
├── src/
│   ├── data_fetch.py            # yfinance data fetching + caching
│   ├── features.py              # Technical indicator engineering
│   ├── train.py                 # Model training pipeline
│   └── predict.py               # Prediction logic
├── api/
│   └── app.py                   # FastAPI REST endpoint
├── client/                      # React frontend (Vite)
│   ├── src/
│   │   ├── App.jsx
│   │   ├── components/
│   │   │   ├── Navbar.jsx
│   │   │   ├── TickerInput.jsx
│   │   │   ├── StockChart.jsx
│   │   │   ├── PredictionTable.jsx
│   │   │   └── MetricsDisplay.jsx
│   │   └── index.css
│   └── package.json
├── server/                      # Express.js backend
│   ├── server.js
│   ├── routes/predict.js
│   └── models/Prediction.js
├── requirements.txt
└── README.md
```

## 🤖 ML Models Used

```bash
| Model | Type | Purpose |
|-------|------|---------|
| **Linear Regression** | Regression | Baseline price prediction |
| **Decision Tree** | Regression | Non-linear price prediction |
| **Random Forest** | Regression | Ensemble price prediction (typically best) |
| **Logistic Regression** | Classification | Directional (up/down) prediction |
```

## 📊 Technical Indicators

- **SMA** (7, 21, 50-day) — Simple Moving Averages
- **EMA** (12, 26-day) — Exponential Moving Averages
- **RSI** (14-day) — Relative Strength Index
- **MACD** — Moving Average Convergence Divergence
- **Bollinger Bands** (20-day) — Volatility bands
- **Daily Returns** — Percentage price changes
- **Lag Features** (t-1 to t-5) — Previous day prices

## 🚀 Getting Started

### Prerequisites

- **Python** 3.9+ with pip
- **Node.js** 18+ with npm
- **MongoDB** (optional — for prediction history logging)

### 1. Install Python Dependencies

```bash
cd stock-price-predictor
pip install -r requirements.txt
```

### 2. Start the FastAPI ML Service

```bash
cd stock-price-predictor
python -m uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload
```

### 3. Start the Express.js Backend

```bash
cd stock-price-predictor/server
npm install
npm start
```

### 4. Start the React Frontend

```bash
cd stock-price-predictor/client
npm install
npm run dev
```

### 5. Open the App

Navigate to **<http://localhost:5173>** in your browser.

### Quick Start Steps

1. Enter a ticker symbol (e.g., `AAPL`)
2. Click **Train Model** (takes 1-2 minutes)
3. Click **Predict** to see forecasted prices

## 📓 Jupyter Notebook

For the complete ML pipeline with all visualizations, analysis, and explanations:

```bash
cd stock-price-predictor/notebooks
jupyter notebook Stock_Price_Prediction.ipynb
```

The notebook covers:

1. Data Collection (yfinance)
2. Exploratory Data Analysis
3. Feature Engineering
4. Model Training & Comparison
5. Hyperparameter Tuning
6. Evaluation & Visualization
7. Model Interpretation (SHAP + Feature Importance)
8. Production Pipeline

## 🔌 API Endpoints

```bash
| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/predict` | Predict future prices — `{"ticker": "AAPL", "days": 5}` |
| `POST` | `/train` | Train model — `{"ticker": "AAPL", "period": "5y"}` |
| `GET` | `/history/{ticker}` | Get historical prices for charting |
| `GET` | `/health` | Health check |
```

## 📈 Evaluation Metrics

- **RMSE** — Root Mean Squared Error
- **MAE** — Mean Absolute Error
- **MAPE** — Mean Absolute Percentage Error
- **Directional Accuracy** — % of correct up/down predictions

## ⚠️ Disclaimer

This project is for **educational purposes only**. Stock price predictions are inherently uncertain and should not be used as the sole basis for investment decisions.

## 👨‍🎓 Author

Final Year Student — Machine Learning Engineering & Quantitative Development

---

### Built with Python, scikit-learn, FastAPI, React, Express.js, and MongoDB
