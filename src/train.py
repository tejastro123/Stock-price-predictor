"""
train.py — Model Training Pipeline
====================================
Trains multiple models and saves the best-performing model to disk.

Regression models (price prediction):
  • Linear Regression   • Decision Tree   • Random Forest

Classification model (direction prediction):
  • Logistic Regression (Up / Down)

Author : Student ML Engineer
Project: Stock Price Prediction System
"""

import os
import logging
import numpy as np
import pandas as pd
import joblib

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV

from src.data_fetch import fetch_stock_data
from src.features import (
    add_all_technical_indicators,
    prepare_features,
    scale_features,
    time_based_split,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
RANDOM_SEED = 42

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

np.random.seed(RANDOM_SEED)


# ═══════════════════════════════════════════════════════════════════════════
# Evaluation Helpers
# ═══════════════════════════════════════════════════════════════════════════

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute RMSE, MAE, MAPE, and Directional Accuracy."""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)

    # MAPE — guard against zero
    mask = y_true != 0
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

    # Directional Accuracy
    if len(y_true) > 1:
        actual_dir = np.sign(np.diff(y_true))
        pred_dir = np.sign(np.diff(y_pred))
        dir_acc = np.mean(actual_dir == pred_dir) * 100
    else:
        dir_acc = 0.0

    return {
        "rmse": round(rmse, 4),
        "mae": round(mae, 4),
        "mape": round(mape, 4),
        "directional_accuracy": round(dir_acc, 2),
    }


# ═══════════════════════════════════════════════════════════════════════════
# Model Definitions
# ═══════════════════════════════════════════════════════════════════════════

def _get_regression_models() -> dict:
    """Return a dictionary of candidate regression models for price prediction."""
    return {
        "LinearRegression": LinearRegression(),
        "DecisionTree": DecisionTreeRegressor(
            max_depth=10,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=RANDOM_SEED,
        ),
        "RandomForest": RandomForestRegressor(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            random_state=RANDOM_SEED,
            n_jobs=-1,
        ),
    }


def _get_direction_model() -> LogisticRegression:
    """Return a Logistic Regression model for directional (up/down) prediction."""
    return LogisticRegression(
        max_iter=1000,
        random_state=RANDOM_SEED,
        class_weight="balanced",
    )


# ═══════════════════════════════════════════════════════════════════════════
# Hyperparameter Tuning (Random Forest)
# ═══════════════════════════════════════════════════════════════════════════

def tune_random_forest(X_train: np.ndarray, y_train: np.ndarray) -> RandomForestRegressor:
    """
    Tune Random Forest using TimeSeriesSplit + RandomizedSearchCV.

    Returns
    -------
    RandomForestRegressor
        Best estimator from the search.
    """
    param_distributions = {
        "n_estimators": [100, 200, 300, 500],
        "max_depth": [5, 10, 15, 20, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2", 0.5, 0.8],
    }

    tscv = TimeSeriesSplit(n_splits=5)
    base_model = RandomForestRegressor(random_state=RANDOM_SEED, n_jobs=-1)

    search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_distributions,
        n_iter=20,
        cv=tscv,
        scoring="neg_root_mean_squared_error",
        random_state=RANDOM_SEED,
        n_jobs=-1,
        verbose=0,
    )

    logger.info("Starting Random Forest hyperparameter tuning (20 iterations, 5-fold TS CV)...")
    search.fit(X_train, y_train)
    logger.info("Best params: %s", search.best_params_)
    logger.info("Best CV RMSE: %.4f", -search.best_score_)

    return search.best_estimator_


# ═══════════════════════════════════════════════════════════════════════════
# Main Training Pipeline
# ═══════════════════════════════════════════════════════════════════════════

def train_model(
    ticker: str,
    period: str = "5y",
    tune: bool = True,
) -> dict:
    """
    End-to-end training pipeline for a given ticker.

    Steps:
      1. Fetch data
      2. Engineer features
      3. Split (time-based)
      4. Scale
      5. Train & compare regression models (LR, DT, RF)
      6. Train Logistic Regression for direction (up/down)
      7. (Optionally) tune Random Forest
      8. Save the best model + scaler + metadata

    Returns
    -------
    dict
        Training results including metrics for each model.
    """
    os.makedirs(MODELS_DIR, exist_ok=True)

    # ── Step 1: Fetch ─────────────────────────────────────────────────────
    logger.info("Fetching data for %s (period=%s)...", ticker, period)
    df = fetch_stock_data(ticker, period=period)

    # ── Step 2: Feature Engineering ───────────────────────────────────────
    df = add_all_technical_indicators(df)
    X, y = prepare_features(df, target_col="Close")
    feature_names = X.columns.tolist()

    # ── Step 3: Time-based Split ──────────────────────────────────────────
    X_train, X_test, y_train, y_test = time_based_split(X, y, test_ratio=0.2)
    logger.info("Train size: %d | Test size: %d", len(X_train), len(X_test))

    # ── Step 4: Scale ─────────────────────────────────────────────────────
    X_train_scaled, scaler = scale_features(X_train)
    X_test_scaled, _ = scale_features(X_test, scaler=scaler)

    # ── Step 5: Train & Compare Regression Models ─────────────────────────
    models = _get_regression_models()
    results = {}
    best_model_name = None
    best_rmse = float("inf")

    for name, model in models.items():
        logger.info("Training %s...", name)
        model.fit(X_train_scaled, y_train)
        preds = model.predict(X_test_scaled)
        metrics = compute_metrics(y_test.values, preds)
        results[name] = metrics
        logger.info("%s — RMSE: %.4f | MAE: %.4f | Dir Acc: %.2f%%",
                     name, metrics["rmse"], metrics["mae"], metrics["directional_accuracy"])

        if metrics["rmse"] < best_rmse:
            best_rmse = metrics["rmse"]
            best_model_name = name

    # ── Step 6: Logistic Regression for Direction ─────────────────────────
    # Create binary target: 1 = price went up, 0 = price went down
    y_train_dir = (y_train.diff().dropna() > 0).astype(int)
    y_test_dir = (y_test.diff().dropna() > 0).astype(int)
    X_train_dir = X_train_scaled[1:]  # align with diff (drops first row)
    X_test_dir = X_test_scaled[1:]

    try:
        log_model = _get_direction_model()
        log_model.fit(X_train_dir, y_train_dir)
        dir_preds = log_model.predict(X_test_dir)
        dir_accuracy = accuracy_score(y_test_dir, dir_preds) * 100

        results["LogisticRegression_Direction"] = {
            "rmse": None,
            "mae": None,
            "mape": None,
            "directional_accuracy": round(dir_accuracy, 2),
        }
        logger.info("LogisticRegression (Direction) — Accuracy: %.2f%%", dir_accuracy)

        # Save direction model separately
        safe_ticker = ticker.upper().replace("/", "_")
        joblib.dump(log_model, os.path.join(MODELS_DIR, f"{safe_ticker}_direction_model.pkl"))
    except Exception as exc:
        logger.warning("Logistic Regression direction model failed: %s", exc)

    # ── Step 7: Tune (Random Forest) ──────────────────────────────────────
    if tune and best_model_name == "RandomForest":
        logger.info("Tuning Random Forest...")
        tuned_model = tune_random_forest(X_train_scaled, y_train)
        preds = tuned_model.predict(X_test_scaled)
        tuned_metrics = compute_metrics(y_test.values, preds)
        results["RandomForest_Tuned"] = tuned_metrics
        logger.info("Tuned RandomForest — RMSE: %.4f | MAE: %.4f",
                     tuned_metrics["rmse"], tuned_metrics["mae"])

        if tuned_metrics["rmse"] < best_rmse:
            best_model_name = "RandomForest_Tuned"
            models["RandomForest_Tuned"] = tuned_model

    # ── Step 8: Save ──────────────────────────────────────────────────────
    best_model = models[best_model_name]
    safe_ticker = ticker.upper().replace("/", "_")
    model_path = os.path.join(MODELS_DIR, f"{safe_ticker}_model.pkl")
    scaler_path = os.path.join(MODELS_DIR, f"{safe_ticker}_scaler.pkl")
    meta_path = os.path.join(MODELS_DIR, f"{safe_ticker}_meta.pkl")

    joblib.dump(best_model, model_path)
    joblib.dump(scaler, scaler_path)
    joblib.dump({
        "ticker": ticker,
        "best_model": best_model_name,
        "feature_names": feature_names,
        "metrics": results[best_model_name],
    }, meta_path)

    logger.info("Saved best model (%s) to %s", best_model_name, model_path)

    return {
        "ticker": ticker,
        "best_model": best_model_name,
        "all_results": results,
        "best_metrics": results[best_model_name],
    }


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    result = train_model("AAPL", period="2y", tune=True)
    print("\n=== Training Results ===")
    for name, metrics in result["all_results"].items():
        print(f"  {name}: {metrics}")
    print(f"\nBest model: {result['best_model']}")
