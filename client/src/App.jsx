/**
 * App.jsx — Main Application Component
 * =======================================
 * Orchestrates state management and renders all child components.
 * Supports Single prediction and Compare modes.
 *
 * Author : Student ML Engineer
 * Project: Stock Price Prediction System
 */

import React, { useState } from "react";
import axios from "axios";

import Navbar from "./components/Navbar";
import TickerInput from "./components/TickerInput";
import MetricsDisplay from "./components/MetricsDisplay";
import StockChart from "./components/StockChart";
import PredictionTable from "./components/PredictionTable";
import ModelComparison from "./components/ModelComparison";
import FeatureImportance from "./components/FeatureImportance";
import StockInfo from "./components/StockInfo";
import ProgressStepper from "./components/ProgressStepper";
import CompareMode from "./components/CompareMode";

// API base URL — configurable via env, defaults to /api (Vite proxy handles it in dev)
const API_BASE = import.meta.env.VITE_API_BASE_URL || "/api";

export default function App() {
  // ── State ────────────────────────────────────────────────────────────
  const [appMode, setAppMode] = useState("single"); // "single" | "compare"
  const [loading, setLoading] = useState(false);
  const [training, setTraining] = useState(false);
  const [error, setError] = useState(null);
  const [statusMsg, setStatusMsg] = useState(null);

  // Prediction result
  const [result, setResult] = useState(null);

  // Training result
  const [trainResult, setTrainResult] = useState(null);

  // Stock info
  const [stockInfo, setStockInfo] = useState(null);

  // ── Handlers ─────────────────────────────────────────────────────────

  /**
   * Fetch stock info (non-blocking)
   */
  const fetchStockInfo = async (ticker) => {
    try {
      const response = await axios.get(`${API_BASE}/info/${ticker.toUpperCase()}`);
      setStockInfo(response.data);
    } catch (err) {
      console.warn("Stock info fetch failed (non-critical):", err.message);
    }
  };

  /**
   * Handle prediction request
   */
  const handlePredict = async (ticker, days) => {
    setLoading(true);
    setError(null);
    setStatusMsg("Generating predictions...");

    // Fetch stock info in parallel (non-blocking)
    fetchStockInfo(ticker);

    try {
      const response = await axios.post(`${API_BASE}/predict`, {
        ticker,
        days,
      });

      setResult(response.data);
      setStatusMsg(null);
    } catch (err) {
      const msg =
        err.response?.data?.error ||
        err.response?.data?.detail ||
        "Prediction failed. Make sure the ML API is running.";
      setError(msg);
      setStatusMsg(null);
    } finally {
      setLoading(false);
    }
  };

  /**
   * Handle training request
   */
  const handleTrain = async (ticker) => {
    setTraining(true);
    setError(null);
    setStatusMsg(`Training model for ${ticker}... This may take 1-2 minutes.`);

    // Fetch stock info in parallel
    fetchStockInfo(ticker);

    try {
      const response = await axios.post(`${API_BASE}/train`, {
        ticker,
        period: "5y",
        tune: true,
      });

      setTrainResult(response.data);
      setStatusMsg(`✅ Model trained for ${ticker}! Best model: ${response.data.best_model}`);

      // Clear success message after 5 seconds
      setTimeout(() => setStatusMsg(null), 5000);
    } catch (err) {
      const msg =
        err.response?.data?.error ||
        err.response?.data?.detail ||
        "Training failed. Make sure the ML API is running.";
      setError(msg);
      setStatusMsg(null);
    } finally {
      setTraining(false);
    }
  };

  // ── Render ───────────────────────────────────────────────────────────
  return (
    <div className="app">
      <Navbar />

      {/* Mode Tabs */}
      <div className="app-mode-tabs">
        <button
          className={`app-mode-tab ${appMode === "single" ? "app-mode-tab--active" : ""}`}
          onClick={() => setAppMode("single")}
        >
          📊 Single Stock
        </button>
        <button
          className={`app-mode-tab ${appMode === "compare" ? "app-mode-tab--active" : ""}`}
          onClick={() => setAppMode("compare")}
        >
          🔀 Compare Stocks
        </button>
      </div>

      {/* ── SINGLE MODE ─────────────────────────────────────────── */}
      {appMode === "single" && (
        <>
          <TickerInput
            onPredict={handlePredict}
            onTrain={handleTrain}
            loading={loading}
            training={training}
          />

          {/* Progress Stepper */}
          {(training || loading) && (
            <div style={{ marginTop: 16 }}>
              <ProgressStepper
                mode={training ? "training" : "prediction"}
                active={training || loading}
              />
            </div>
          )}

          {/* Status Messages */}
          {statusMsg && !training && !loading && (
            <div className="status status--loading fade-in" style={{ margin: "16px 0" }}>
              <div className="pulse-dot" />
              {statusMsg}
            </div>
          )}

          {error && (
            <div className="status status--error fade-in" style={{ margin: "16px 0" }}>
              ⚠️ {error}
            </div>
          )}

          {/* Stock Info Card */}
          {stockInfo && (result || trainResult) && (
            <div style={{ marginTop: 24 }}>
              <StockInfo info={stockInfo} />
            </div>
          )}

          {/* Training Results */}
          {trainResult && (
            <>
              <div style={{ marginTop: 24 }}>
                <ModelComparison
                  allResults={trainResult.all_results}
                  bestModel={trainResult.best_model}
                />
              </div>
              {trainResult.feature_importance && trainResult.feature_importance.length > 0 && (
                <div style={{ marginTop: 24 }}>
                  <FeatureImportance
                    features={trainResult.feature_importance}
                    modelName={trainResult.best_model}
                  />
                </div>
              )}
            </>
          )}

          {/* Prediction Results */}
          {result && (
            <div className="mobile-card-scroll">
              <div className="mobile-card-scroll__inner">
                {/* Metrics */}
                <div className="mobile-card-scroll__item" style={{ marginTop: 24 }}>
                  <MetricsDisplay
                    metrics={result.metrics}
                    modelName={result.model_used}
                  />
                </div>

                {/* Chart */}
                <div className="mobile-card-scroll__item" style={{ marginTop: 24 }}>
                  <StockChart
                    history={result.history}
                    predictions={result.predictions}
                    ticker={result.ticker}
                  />
                </div>

                {/* Prediction Table */}
                <div className="mobile-card-scroll__item" style={{ marginTop: 24 }}>
                  <PredictionTable
                    predictions={result.predictions}
                    ticker={result.ticker}
                  />
                </div>
              </div>
            </div>
          )}

          {/* Hero landing when no results */}
          {!result && !trainResult && !loading && !training && (
            <div className="hero fade-in" style={{ marginTop: 24 }}>
              <div className="hero__icon">🚀</div>
              <h1 className="hero__title">
                AI-Powered Stock<br />Price Predictions
              </h1>
              <p className="hero__subtitle">
                Train machine learning models on historical data and generate
                multi-day price forecasts with confidence intervals.
                Powered by XGBoost, LightGBM, and Random Forest.
              </p>
              <div className="hero__features">
                <div className="hero__feature">
                  <span className="hero__feature-icon">📈</span>
                  95% Confidence Bands
                </div>
                <div className="hero__feature">
                  <span className="hero__feature-icon">🤖</span>
                  4 ML Models Compared
                </div>
                <div className="hero__feature">
                  <span className="hero__feature-icon">⚡</span>
                  Up to 30-Day Forecast
                </div>
                <div className="hero__feature">
                  <span className="hero__feature-icon">🔍</span>
                  Feature Importance
                </div>
              </div>
            </div>
          )}
        </>
      )}

      {/* ── COMPARE MODE ────────────────────────────────────────── */}
      {appMode === "compare" && <CompareMode />}

      {/* Footer */}
      <footer className="footer">
        Stock Price Prediction System — Built with ML &amp; MERN Stack
        <br />
        <span style={{ fontSize: 11 }}>
          Models: Decision Tree · Random Forest · XGBoost · LightGBM
        </span>
      </footer>
    </div>
  );
}
