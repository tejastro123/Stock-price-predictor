"""
train.py — Model Training Pipeline
====================================
Trains multiple models and saves the best-performing model to disk.

Regression models (price prediction):
  • Decision Tree   • Random Forest   • XGBoost   • LightGBM

Author : Student ML Engineer
Project: Stock Price Prediction System
"""

import os
import logging
import numpy as np
import pandas as pd
import joblib

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

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
        "XGBoost": XGBRegressor(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=RANDOM_SEED,
            n_jobs=-1,
            verbosity=0,
        ),
        "LightGBM": LGBMRegressor(
            n_estimators=300,
            max_depth=10,
            learning_rate=0.05,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=RANDOM_SEED,
            n_jobs=-1,
            verbose=-1,
        ),
    }


# ═══════════════════════════════════════════════════════════════════════════
# Hyperparameter Tuning (Random Forest)
# ═══════════════════════════════════════════════════════════════════════════

def _tune_model(name, base_model, param_distributions, X_train, y_train, n_iter=20):
    """
    Tune a model using TimeSeriesSplit + RandomizedSearchCV.

    Returns the best estimator from the search.
    """
    tscv = TimeSeriesSplit(n_splits=5)

    search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_distributions,
        n_iter=n_iter,
        cv=tscv,
        scoring="neg_root_mean_squared_error",
        random_state=RANDOM_SEED,
        n_jobs=-1,
        verbose=0,
    )

    logger.info("Starting %s hyperparameter tuning (%d iterations, 5-fold TS CV)...", name, n_iter)
    search.fit(X_train, y_train)
    logger.info("%s best params: %s", name, search.best_params_)
    logger.info("%s best CV RMSE: %.4f", name, -search.best_score_)

    return search.best_estimator_


def tune_random_forest(X_train: np.ndarray, y_train: np.ndarray) -> RandomForestRegressor:
    """Tune Random Forest."""
    return _tune_model(
        "RandomForest",
        RandomForestRegressor(random_state=RANDOM_SEED, n_jobs=-1),
        {
            "n_estimators": [100, 200, 300, 500],
            "max_depth": [5, 10, 15, 20, None],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "max_features": ["sqrt", "log2", 0.5, 0.8],
        },
        X_train, y_train,
    )


def tune_xgboost(X_train: np.ndarray, y_train: np.ndarray) -> XGBRegressor:
    """Tune XGBoost."""
    return _tune_model(
        "XGBoost",
        XGBRegressor(random_state=RANDOM_SEED, n_jobs=-1, verbosity=0),
        {
            "n_estimators": [100, 200, 300, 500],
            "max_depth": [3, 5, 6, 8, 10],
            "learning_rate": [0.01, 0.03, 0.05, 0.1],
            "subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
            "colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
            "reg_alpha": [0, 0.01, 0.1, 1],
            "reg_lambda": [0.5, 1, 2, 5],
        },
        X_train, y_train,
    )


def tune_lightgbm(X_train: np.ndarray, y_train: np.ndarray) -> LGBMRegressor:
    """Tune LightGBM."""
    return _tune_model(
        "LightGBM",
        LGBMRegressor(random_state=RANDOM_SEED, n_jobs=-1, verbose=-1),
        {
            "n_estimators": [100, 200, 300, 500],
            "max_depth": [5, 8, 10, 15, -1],
            "learning_rate": [0.01, 0.03, 0.05, 0.1],
            "num_leaves": [15, 31, 50, 80],
            "subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
            "colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
            "reg_alpha": [0, 0.01, 0.1, 1],
            "reg_lambda": [0.5, 1, 2, 5],
        },
        X_train, y_train,
    )


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
      5. Train & compare regression models (DT, RF, XGBoost, LightGBM)
      6. (Optionally) tune the best model
      7. Save the best model + scaler + metadata

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

    # ── Step 6: Tune the best model ───────────────────────────────────────
    tuners = {
        "RandomForest": tune_random_forest,
        "XGBoost": tune_xgboost,
        "LightGBM": tune_lightgbm,
    }

    if tune and best_model_name in tuners:
        tuner_fn = tuners[best_model_name]
        logger.info("Tuning %s...", best_model_name)
        tuned_model = tuner_fn(X_train_scaled, y_train)
        preds = tuned_model.predict(X_test_scaled)
        tuned_metrics = compute_metrics(y_test.values, preds)
        tuned_name = f"{best_model_name}_Tuned"
        results[tuned_name] = tuned_metrics
        logger.info("Tuned %s — RMSE: %.4f | MAE: %.4f",
                     best_model_name, tuned_metrics["rmse"], tuned_metrics["mae"])

        if tuned_metrics["rmse"] < best_rmse:
            best_model_name = tuned_name
            models[tuned_name] = tuned_model

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

    # ── Extract feature importances ───────────────────────────────────────
    feature_importance = []
    if hasattr(best_model, "feature_importances_"):
        importances = best_model.feature_importances_
        sorted_idx = np.argsort(importances)[::-1]
        for i in sorted_idx[:15]:  # top 15 features
            feature_importance.append({
                "feature": feature_names[i],
                "importance": round(float(importances[i]), 4),
            })
        logger.info("Top 3 features: %s",
                     ", ".join(f["feature"] for f in feature_importance[:3]))

    return {
        "ticker": ticker,
        "best_model": best_model_name,
        "all_results": results,
        "best_metrics": results[best_model_name],
        "feature_importance": feature_importance,
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
