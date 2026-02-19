"""
data_fetch.py — Stock Data Fetching & Caching Module
=====================================================
Fetches historical stock data from Yahoo Finance via yfinance.
Implements local CSV caching and retry logic for API failures.

Author : Student ML Engineer
Project: Stock Price Prediction System
"""

import os
import time
import logging
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
MAX_RETRIES = 3
RETRY_DELAY = 2  # seconds
CACHE_EXPIRY_HOURS = 6  # re-download if cache is older than this

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def _get_cache_path(ticker: str, period: str) -> str:
    """Generate a cache file path for the given ticker and period."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    safe_ticker = ticker.upper().replace("/", "_")
    return os.path.join(CACHE_DIR, f"{safe_ticker}_{period}.csv")


def _is_cache_valid(cache_path: str) -> bool:
    """Check whether the cached file exists and is recent enough."""
    if not os.path.exists(cache_path):
        return False
    modified_time = datetime.fromtimestamp(os.path.getmtime(cache_path))
    return (datetime.now() - modified_time) < timedelta(hours=CACHE_EXPIRY_HOURS)


def fetch_stock_data(
    ticker: str,
    period: str = "5y",
    use_cache: bool = True,
) -> pd.DataFrame:
    """
    Fetch historical OHLCV data for a given stock ticker.

    Parameters
    ----------
    ticker : str
        Stock ticker symbol (e.g., "AAPL", "GOOGL").
    period : str
        Data period to download (e.g., "1y", "2y", "5y", "max").
    use_cache : bool
        If True, return cached data if available and fresh.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: Open, High, Low, Close, Volume, Adj Close.
        Indexed by Date.

    Raises
    ------
    ValueError
        If the ticker is invalid or no data is returned after retries.
    """
    cache_path = _get_cache_path(ticker, period)

    # ── Try cache first ───────────────────────────────────────────────────
    if use_cache and _is_cache_valid(cache_path):
        logger.info("Loading cached data for %s from %s", ticker, cache_path)
        df = pd.read_csv(cache_path, index_col="Date", parse_dates=True)
        return df

    # ── Download with retry logic ────────────────────────────────────────
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            logger.info(
                "Downloading %s data (attempt %d/%d)...", ticker, attempt, MAX_RETRIES
            )
            stock = yf.Ticker(ticker)
            df = stock.history(period=period, auto_adjust=False)

            if df.empty:
                raise ValueError(f"No data returned for ticker '{ticker}'.")

            # Standardise column names
            df.index.name = "Date"

            # Keep only the columns we need
            keep_cols = ["Open", "High", "Low", "Close", "Volume"]
            if "Adj Close" in df.columns:
                keep_cols.append("Adj Close")
            df = df[[c for c in keep_cols if c in df.columns]]

            # Save to cache
            df.to_csv(cache_path)
            logger.info(
                "Successfully fetched %d rows for %s. Cached to %s.",
                len(df), ticker, cache_path,
            )
            return df

        except Exception as exc:
            logger.warning("Attempt %d failed for %s: %s", attempt, ticker, exc)
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY * attempt)  # exponential-ish back-off
            else:
                raise ValueError(
                    f"Failed to fetch data for '{ticker}' after {MAX_RETRIES} attempts: {exc}"
                ) from exc


def get_company_info(ticker: str) -> dict:
    """
    Return basic company information for a ticker.

    Returns
    -------
    dict
        Keys: shortName, sector, industry, marketCap, website.
    """
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        return {
            "shortName": info.get("shortName", ticker),
            "sector": info.get("sector", "N/A"),
            "industry": info.get("industry", "N/A"),
            "marketCap": info.get("marketCap", "N/A"),
            "website": info.get("website", "N/A"),
        }
    except Exception as exc:
        logger.warning("Could not fetch company info for %s: %s", ticker, exc)
        return {"shortName": ticker, "sector": "N/A", "industry": "N/A"}


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    data = fetch_stock_data("AAPL", period="2y")
    print(data.head())
    print(f"\nShape: {data.shape}")
    print(f"\nCompany info: {get_company_info('AAPL')}")
