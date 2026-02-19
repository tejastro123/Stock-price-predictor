"""
app.py — FastAPI Backend for Stock Price Prediction
=====================================================
REST API endpoints for training models and generating predictions.

Endpoints:
  POST /predict      — Predict future stock prices
  POST /train        — Train/retrain a model for a ticker
  GET  /history/{t}  — Get recent historical data for charting
  GET  /info/{t}     — Get stock metadata
  GET  /health       — Health check

Author : Student ML Engineer
Project: Stock Price Prediction System
"""

import os
import re
import sys
import asyncio
import logging
from functools import partial

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

# Load .env from project root (for local dev)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

# Add project root to path so we can import src modules
sys.path.insert(0, PROJECT_ROOT)

from src.predict import predict_stock, get_historical_data
from src.train import train_model

# ---------------------------------------------------------------------------
# Configuration (from environment)
# ---------------------------------------------------------------------------
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*").split(",")
ML_API_HOST = os.getenv("ML_API_HOST", "0.0.0.0")
ML_API_PORT = int(os.getenv("ML_API_PORT", "8000"))
TRAIN_TIMEOUT = int(os.getenv("TRAIN_TIMEOUT", "300"))  # seconds

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Ticker validation pattern
# ---------------------------------------------------------------------------
TICKER_REGEX = re.compile(r"^[A-Z]{1,5}$")

# ---------------------------------------------------------------------------
# App Setup
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Stock Price Prediction API",
    description="ML-powered stock price forecasting service",
    version="1.0.0",
)

# CORS — configurable origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
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

    @field_validator("ticker")
    @classmethod
    def validate_ticker(cls, v):
        v = v.strip().upper()
        if not TICKER_REGEX.match(v):
            raise ValueError("Ticker must be 1-5 uppercase letters (e.g. AAPL, TSLA)")
        return v


class TrainRequest(BaseModel):
    ticker: str = Field(..., description="Stock ticker symbol", example="AAPL")
    period: str = Field(default="5y", description="Historical data period")
    tune: bool = Field(default=True, description="Run hyperparameter tuning")

    @field_validator("ticker")
    @classmethod
    def validate_ticker(cls, v):
        v = v.strip().upper()
        if not TICKER_REGEX.match(v):
            raise ValueError("Ticker must be 1-5 uppercase letters (e.g. AAPL, TSLA)")
        return v


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
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None, partial(predict_stock, request.ticker, days_ahead=request.days)
        )
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
    Has a configurable timeout (default 300s) to prevent hanging.

    Request body:
        {"ticker": "AAPL", "period": "5y", "tune": true}

    Returns:
        Training results and metrics for all models.
    """
    try:
        logger.info(
            "Training request: ticker=%s, period=%s, tune=%s (timeout=%ds)",
            request.ticker, request.period, request.tune, TRAIN_TIMEOUT,
        )
        loop = asyncio.get_event_loop()
        result = await asyncio.wait_for(
            loop.run_in_executor(
                None,
                partial(
                    train_model,
                    request.ticker,
                    period=request.period,
                    tune=request.tune,
                ),
            ),
            timeout=TRAIN_TIMEOUT,
        )
        return result
    except asyncio.TimeoutError:
        logger.error("Training timed out after %ds for %s", TRAIN_TIMEOUT, request.ticker)
        raise HTTPException(
            status_code=504,
            detail=f"Training timed out after {TRAIN_TIMEOUT}s. Try with tune=false for faster results.",
        )
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
    ticker = ticker.strip().upper()
    if not TICKER_REGEX.match(ticker):
        raise HTTPException(status_code=400, detail="Invalid ticker format")

    try:
        data = get_historical_data(ticker, days=days)
        return {"ticker": ticker, "history": data}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.error("History fetch failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to fetch history: {str(exc)}")


@app.get("/info/{ticker}")
async def stock_info(ticker: str):
    """
    Get stock metadata (name, sector, current price, market cap, etc.).
    Uses a 15-second timeout on yfinance calls.
    """
    ticker = ticker.strip().upper()
    if not TICKER_REGEX.match(ticker):
        raise HTTPException(status_code=400, detail="Invalid ticker format")

    try:
        import yfinance as yf
        stock = yf.Ticker(ticker)
        info = stock.info

        # If yfinance returns nearly empty info, the ticker is likely invalid
        if not info.get("shortName") and not info.get("longName"):
            raise HTTPException(
                status_code=404,
                detail=f"Ticker '{ticker}' not found — no data returned from Yahoo Finance.",
            )

        return {
            "ticker": ticker,
            "name": info.get("shortName") or info.get("longName", ticker),
            "sector": info.get("sector", "N/A"),
            "industry": info.get("industry", "N/A"),
            "market_cap": info.get("marketCap"),
            "current_price": info.get("currentPrice") or info.get("regularMarketPrice"),
            "previous_close": info.get("previousClose"),
            "day_high": info.get("dayHigh") or info.get("regularMarketDayHigh"),
            "day_low": info.get("dayLow") or info.get("regularMarketDayLow"),
            "fifty_two_week_high": info.get("fiftyTwoWeekHigh"),
            "fifty_two_week_low": info.get("fiftyTwoWeekLow"),
            "pe_ratio": info.get("trailingPE"),
            "eps": info.get("trailingEps"),
            "dividend_yield": info.get("dividendYield"),
            "volume": info.get("volume") or info.get("regularMarketVolume"),
            "avg_volume": info.get("averageVolume"),
            "beta": info.get("beta"),
        }
    except HTTPException:
        raise  # re-raise our own 404
    except Exception as exc:
        logger.error("Stock info fetch failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to fetch stock info: {str(exc)}")


# ---------------------------------------------------------------------------
# Run directly: python api/app.py
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=ML_API_HOST, port=ML_API_PORT, reload=True)

