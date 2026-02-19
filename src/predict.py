"""
predict.py — Prediction Module
================================
Loads a trained model and generates future stock price predictions.

Author : Student ML Engineer
Project: Stock Price Prediction System
"""

import os
import logging
import numpy as np
import pandas as pd
import joblib
from datetime import datetime, timedelta

from src.data_fetch import fetch_stock_data
from src.features import add_all_technical_indicators, prepare_features, scale_features

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def _load_artifacts(ticker: str) -> tuple:
    """
    Load model, scaler, and metadata for a ticker.

    Returns
    -------
    tuple of (model, scaler, metadata_dict)
    """
    safe_ticker = ticker.upper().replace("/", "_")
    model_path = os.path.join(MODELS_DIR, f"{safe_ticker}_model.pkl")
    scaler_path = os.path.join(MODELS_DIR, f"{safe_ticker}_scaler.pkl")
    meta_path = os.path.join(MODELS_DIR, f"{safe_ticker}_meta.pkl")

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"No trained model found for '{ticker}'. "
            f"Please train the model first using train.py."
        )

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    meta = joblib.load(meta_path)

    return model, scaler, meta


def predict_stock(ticker: str, days_ahead: int = 5) -> dict:
    """
    Predict future stock prices for the next N trading days.

    The model uses a recursive (autoregressive) strategy:
      1. Use the latest available row of features to predict the next Close.
      2. Shift the data forward — the predicted Close becomes the new
         'Close' value, and all indicators are recomputed.
      3. Repeat for each day in the forecast horizon.

    Parameters
    ----------
    ticker : str
        Stock ticker symbol.
    days_ahead : int
        Number of trading days to predict (default 5 ≈ 1 week).

    Returns
    -------
    dict
        {
            "ticker": str,
            "model_used": str,
            "metrics": dict,
            "history": [{"date": str, "close": float}, ...],   # last 60 days
            "predictions": [{"date": str, "close": float}, ...]
        }
    """
    # ── Load artifacts ────────────────────────────────────────────────────
    model, scaler, meta = _load_artifacts(ticker)
    feature_names = meta["feature_names"]
    logger.info("Loaded model '%s' for %s", meta["best_model"], ticker)

    # ── Fetch latest data ─────────────────────────────────────────────────
    df = fetch_stock_data(ticker, period="2y", use_cache=False)
    df_enriched = add_all_technical_indicators(df)
    df_enriched = df_enriched.dropna()

    # ── Prepare history (last 60 trading days) ────────────────────────────
    history_df = df.tail(60)
    history = [
        {"date": d.strftime("%Y-%m-%d"), "close": round(float(row["Close"]), 2)}
        for d, row in history_df.iterrows()
    ]

    # ── Recursive prediction ──────────────────────────────────────────────
    predictions = []
    working_df = df.copy()
    last_date = working_df.index[-1]

    for day in range(1, days_ahead + 1):
        # Recompute indicators on the working data
        enriched = add_all_technical_indicators(working_df)
        enriched = enriched.dropna()

        # Extract the last row of features
        last_row = enriched[feature_names].iloc[[-1]]
        last_scaled = scaler.transform(last_row)

        # Predict
        pred_price = float(model.predict(last_scaled)[0])

        # Compute next business date
        next_date = last_date + timedelta(days=1)
        while next_date.weekday() >= 5:  # skip weekends
            next_date += timedelta(days=1)

        predictions.append({
            "date": next_date.strftime("%Y-%m-%d"),
            "close": round(pred_price, 2),
        })

        # Append the predicted row to working_df for next iteration
        new_row = pd.DataFrame({
            "Open": [pred_price],
            "High": [pred_price * 1.005],   # slight estimate
            "Low": [pred_price * 0.995],
            "Close": [pred_price],
            "Volume": [working_df["Volume"].tail(5).mean()],
        }, index=[next_date])

        if "Adj Close" in working_df.columns:
            new_row["Adj Close"] = pred_price

        working_df = pd.concat([working_df, new_row])
        last_date = next_date

    logger.info("Generated %d-day forecast for %s", days_ahead, ticker)

    return {
        "ticker": ticker.upper(),
        "model_used": meta["best_model"],
        "metrics": meta["metrics"],
        "history": history,
        "predictions": predictions,
    }


def get_historical_data(ticker: str, days: int = 90) -> list:
    """
    Return recent historical closing prices for charting.

    Returns
    -------
    list of dicts
        [{"date": "2025-01-01", "close": 150.25}, ...]
    """
    df = fetch_stock_data(ticker, period="2y")
    df = df.tail(days)
    return [
        {"date": d.strftime("%Y-%m-%d"), "close": round(float(row["Close"]), 2)}
        for d, row in df.iterrows()
    ]


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    result = predict_stock("AAPL", days_ahead=5)
    print(f"\nTicker: {result['ticker']}")
    print(f"Model:  {result['model_used']}")
    print(f"Metrics: {result['metrics']}")
    print("\nPredictions:")
    for p in result["predictions"]:
        print(f"  {p['date']}: ${p['close']}")
