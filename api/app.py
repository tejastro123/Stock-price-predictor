"""
app.py — FastAPI Backend for Stock Price Prediction
=====================================================
REST API endpoints for training models and generating predictions.

Endpoints:
  POST /predict      — Predict future stock prices
  POST /train        — Train/retrain a model for a ticker
  GET  /history/{t}  — Get recent historical data for charting
  GET  /health       — Health check

Author : Student ML Engineer
Project: Stock Price Prediction System
"""

import os
import sys
import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Add project root to path so we can import src modules
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.predict import predict_stock, get_historical_data
from src.train import train_model

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# App Setup
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Stock Price Prediction API",
    description="ML-powered stock price forecasting service",
    version="1.0.0",
)

# CORS — allow React dev server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Request / Response Schemas
# ---------------------------------------------------------------------------
class PredictRequest(BaseModel):
    ticker: str = Field(..., description="Stock ticker symbol", example="AAPL")
    days: int = Field(default=5, ge=1, le=30, description="Days ahead to predict")


class TrainRequest(BaseModel):
    ticker: str = Field(..., description="Stock ticker symbol", example="AAPL")
    period: str = Field(default="5y", description="Historical data period")
    tune: bool = Field(default=True, description="Run hyperparameter tuning")


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "stock-price-predictor"}


@app.post("/predict")
async def predict(request: PredictRequest):
    """
    Predict future stock prices.

    Request body:
        {"ticker": "AAPL", "days": 5}

    Returns:
        Predicted prices, historical data, model info, and metrics.
    """
    try:
        logger.info("Prediction request: ticker=%s, days=%d", request.ticker, request.days)
        result = predict_stock(request.ticker, days_ahead=request.days)
        return result
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=404,
            detail=str(exc) + " Use the /train endpoint first.",
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.error("Prediction failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(exc)}")


@app.post("/train")
async def train(request: TrainRequest):
    """
    Train or retrain a model for a ticker.

    Request body:
        {"ticker": "AAPL", "period": "5y", "tune": true}

    Returns:
        Training results and metrics for all models.
    """
    try:
        logger.info(
            "Training request: ticker=%s, period=%s, tune=%s",
            request.ticker, request.period, request.tune,
        )
        result = train_model(
            request.ticker,
            period=request.period,
            tune=request.tune,
        )
        return result
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.error("Training failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Training failed: {str(exc)}")


@app.get("/history/{ticker}")
async def history(ticker: str, days: int = 90):
    """
    Get recent historical closing prices for charting.

    Path params:
        ticker — Stock ticker symbol

    Query params:
        days — Number of recent trading days (default 90)
    """
    try:
        data = get_historical_data(ticker, days=days)
        return {"ticker": ticker.upper(), "history": data}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.error("History fetch failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to fetch history: {str(exc)}")


# ---------------------------------------------------------------------------
# Run directly: python api/app.py
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
