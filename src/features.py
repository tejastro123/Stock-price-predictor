"""
features.py — Feature Engineering Module
=========================================
Computes technical indicators and prepares features for ML models.

Technical Indicators:
  • SMA (Simple Moving Average)   • EMA (Exponential Moving Average)
  • RSI (Relative Strength Index) • MACD (Moving Average Convergence Divergence)
  • Bollinger Bands               • Daily Returns
  • Lag features (t-1 … t-N)

Author : Student ML Engineer
Project: Stock Price Prediction System
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


# ═══════════════════════════════════════════════════════════════════════════
# Technical Indicators
# ═══════════════════════════════════════════════════════════════════════════

def add_sma(df: pd.DataFrame, windows: list = [7, 21, 50]) -> pd.DataFrame:
    """Add Simple Moving Averages for the given window sizes."""
    for w in windows:
        df[f"SMA_{w}"] = df["Close"].rolling(window=w).mean()
    return df


def add_ema(df: pd.DataFrame, windows: list = [12, 26]) -> pd.DataFrame:
    """Add Exponential Moving Averages."""
    for w in windows:
        df[f"EMA_{w}"] = df["Close"].ewm(span=w, adjust=False).mean()
    return df


def add_rsi(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    """Compute Relative Strength Index (RSI)."""
    delta = df["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(window=window, min_periods=window).mean()
    avg_loss = loss.rolling(window=window, min_periods=window).mean()

    rs = avg_gain / avg_loss
    df["RSI"] = 100 - (100 / (1 + rs))
    return df


def add_macd(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    """Compute MACD line, signal line, and histogram."""
    ema_fast = df["Close"].ewm(span=fast, adjust=False).mean()
    ema_slow = df["Close"].ewm(span=slow, adjust=False).mean()

    df["MACD"] = ema_fast - ema_slow
    df["MACD_Signal"] = df["MACD"].ewm(span=signal, adjust=False).mean()
    df["MACD_Hist"] = df["MACD"] - df["MACD_Signal"]
    return df


def add_bollinger_bands(df: pd.DataFrame, window: int = 20, num_std: float = 2.0) -> pd.DataFrame:
    """Compute Bollinger Bands (upper, middle, lower)."""
    sma = df["Close"].rolling(window=window).mean()
    std = df["Close"].rolling(window=window).std()

    df["BB_Upper"] = sma + num_std * std
    df["BB_Middle"] = sma
    df["BB_Lower"] = sma - num_std * std
    df["BB_Width"] = (df["BB_Upper"] - df["BB_Lower"]) / df["BB_Middle"]
    return df


def add_daily_returns(df: pd.DataFrame) -> pd.DataFrame:
    """Compute daily percentage returns."""
    df["Daily_Return"] = df["Close"].pct_change() * 100
    return df


def add_lag_features(df: pd.DataFrame, column: str = "Close", lags: int = 5) -> pd.DataFrame:
    """Add lag features (t-1, t-2, …, t-N) for the given column."""
    for lag in range(1, lags + 1):
        df[f"{column}_Lag_{lag}"] = df[column].shift(lag)
    return df


def add_volume_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add volume-based features."""
    df["Volume_SMA_20"] = df["Volume"].rolling(window=20).mean()
    df["Volume_Ratio"] = df["Volume"] / df["Volume_SMA_20"]
    return df


# ═══════════════════════════════════════════════════════════════════════════
# New Indicators (Feature 18)
# ═══════════════════════════════════════════════════════════════════════════

def add_atr(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    """Average True Range (Volatility)."""
    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    prev_close = close.shift(1)

    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()

    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df["ATR"] = tr.rolling(window=window).mean()
    return df


def add_obv(df: pd.DataFrame) -> pd.DataFrame:
    """On-Balance Volume (Momentum)."""
    # Calculate direction: 1 if up, -1 if down, 0 if flat
    direction = np.sign(df["Close"].diff())
    # Fill NaN (first row) with 0
    direction = direction.fillna(0)
    
    # Volume flow = direction * volume
    # Use cumsum to get OBV
    df["OBV"] = (direction * df["Volume"]).cumsum()
    return df


def add_stochastic_oscillator(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    """Stochastic Oscillator (Momentum)."""
    low_min = df["Low"].rolling(window=window).min()
    high_max = df["High"].rolling(window=window).max()

    # Fast Stochastic %K
    df["Stoch_K"] = 100 * ((df["Close"] - low_min) / (high_max - low_min))
    
    # Slow Stochastic %D (3-day SMA of %K)
    df["Stoch_D"] = df["Stoch_K"].rolling(window=3).mean()
    return df


def add_williams_r(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    """Williams %R (Momentum)."""
    low_min = df["Low"].rolling(window=window).min()
    high_max = df["High"].rolling(window=window).max()

    # Williams %R is inverse of Stochastic Fast %K somewhat
    # Formula: (High_max - Close) / (High_max - Low_min) * -100
    df["Williams_R"] = -100 * ((high_max - df["Close"]) / (high_max - low_min))
    return df


# ═══════════════════════════════════════════════════════════════════════════
# Full Pipeline
# ═══════════════════════════════════════════════════════════════════════════

def add_all_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all technical indicators in one call.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with at least 'Close', 'High', 'Low', 'Volume' columns.

    Returns
    -------
    pd.DataFrame
        Original DataFrame with indicator columns appended.
    """
    df = df.copy()
    
    # Trend
    df = add_sma(df)
    df = add_ema(df)
    df = add_macd(df)
    
    # Momentum
    df = add_rsi(df)
    df = add_stochastic_oscillator(df)
    df = add_williams_r(df)
    df = add_obv(df)
    
    # Volatility
    df = add_bollinger_bands(df)
    df = add_atr(df)
    
    # Other
    df = add_daily_returns(df)
    df = add_lag_features(df)
    df = add_volume_features(df)
    
    return df


def prepare_features(
    df: pd.DataFrame,
    target_col: str = "Close",
    drop_na: bool = True,
) -> tuple:
    """
    Prepare feature matrix X and target vector y.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with all indicator columns.
    target_col : str
        Column to predict.
    drop_na : bool
        If True, drop rows with NaN values (from rolling calculations).

    Returns
    -------
    tuple of (X: pd.DataFrame, y: pd.Series)
    """
    df = df.copy()

    if drop_na:
        df = df.dropna()

    # Features: everything except target, non-feature columns
    exclude_cols = [target_col, "Adj Close"]
    feature_cols = [c for c in df.columns if c not in exclude_cols]

    X = df[feature_cols]
    y = df[target_col]
    return X, y


def scale_features(X: pd.DataFrame, scaler: MinMaxScaler = None) -> tuple:
    """
    Scale features using MinMaxScaler.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix.
    scaler : MinMaxScaler or None
        If None, a new scaler is fitted.

    Returns
    -------
    tuple of (X_scaled: np.ndarray, scaler: MinMaxScaler)
    """
    if scaler is None:
        scaler = MinMaxScaler(feature_range=(0, 1))
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = scaler.transform(X)
    return X_scaled, scaler


def time_based_split(
    X: pd.DataFrame,
    y: pd.Series,
    test_ratio: float = 0.2,
) -> tuple:
    """
    Split data chronologically — no shuffling to prevent data leakage.

    Returns
    -------
    tuple of (X_train, X_test, y_train, y_test)
    """
    split_idx = int(len(X) * (1 - test_ratio))
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    return X_train, X_test, y_train, y_test


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Create sample data for testing
    dates = pd.date_range("2022-01-01", periods=300, freq="B")
    np.random.seed(42)
    sample = pd.DataFrame({
        "Open": np.random.uniform(100, 200, 300),
        "High": np.random.uniform(100, 200, 300),
        "Low": np.random.uniform(100, 200, 300),
        "Close": np.cumsum(np.random.randn(300)) + 150,
        "Volume": np.random.randint(1_000_000, 10_000_000, 300),
    }, index=dates)

    enriched = add_all_technical_indicators(sample)
    print(f"Columns after feature engineering: {enriched.columns.tolist()}")
    print(f"\nShape: {enriched.shape}")

    # Check for new columns
    for col in ["ATR", "OBV", "Stoch_K", "Stoch_D", "Williams_R"]:
        if col in enriched.columns:
            print(f"✅ {col} added successfully")
        else:
            print(f"❌ {col} MISSING")

    X, y = prepare_features(enriched)
    print(f"\nFeature matrix shape: {X.shape}")
    print(f"Target shape: {y.shape}")
