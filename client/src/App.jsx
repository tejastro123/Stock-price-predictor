/**
 * App.jsx — Main Application Component
 * =======================================
 * Orchestrates state management and renders all child components.
 * Communicates with the Express backend via Axios.
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

// API base URL — Vite proxy handles /api prefix in dev
const API_BASE = "/api";

export default function App() {
  // ── State ────────────────────────────────────────────────────────────
  const [loading, setLoading] = useState(false);
  const [training, setTraining] = useState(false);
  const [error, setError] = useState(null);
  const [statusMsg, setStatusMsg] = useState(null);

  // Prediction result
  const [result, setResult] = useState(null);

  // Training result
  const [trainResult, setTrainResult] = useState(null);

  // ── Handlers ─────────────────────────────────────────────────────────

  /**
   * Handle prediction request
   */
  const handlePredict = async (ticker, days) => {
    setLoading(true);
    setError(null);
    setStatusMsg("Generating predictions...");

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

      {/* Input Section */}
      <TickerInput
        onPredict={handlePredict}
        onTrain={handleTrain}
        loading={loading}
        training={training}
      />

      {/* Status Messages */}
      {statusMsg && (
        <div className="status status--loading fade-in" style={{ margin: "16px 0" }}>
          {training || loading ? <div className="spinner" /> : <div className="pulse-dot" />}
          {statusMsg}
        </div>
      )}

      {error && (
        <div className="status status--error fade-in" style={{ margin: "16px 0" }}>
          ⚠️ {error}
        </div>
      )}

      {/* Training Results */}
      {trainResult && !result && (
        <div className="card fade-in" style={{ margin: "16px 0" }}>
          <div className="card__header">
            <div className="card__icon card__icon--green">🏆</div>
            <h2 className="card__title">Training Complete</h2>
          </div>
          <p style={{ color: "var(--text-secondary)", marginBottom: 12, fontSize: 14 }}>
            Best model: <strong style={{ color: "var(--accent-3)" }}>{trainResult.best_model}</strong>
          </p>
          <div className="metrics-grid">
            {Object.entries(trainResult.all_results).map(([name, metrics]) => (
              <div key={name} className="metric-card">
                <div className="metric-card__label">{name}</div>
                <div className="metric-card__value">
                  {metrics.rmse != null ? metrics.rmse.toFixed(2) : "—"}
                </div>
                <div style={{ fontSize: 11, color: "var(--text-muted)", marginTop: 4 }}>
                  {metrics.rmse != null ? "RMSE" : `Dir: ${metrics.directional_accuracy}%`}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Prediction Results */}
      {result && (
        <>
          {/* Metrics */}
          <div style={{ marginTop: 24 }}>
            <MetricsDisplay
              metrics={result.metrics}
              modelName={result.model_used}
            />
          </div>

          {/* Chart */}
          <div style={{ marginTop: 24 }}>
            <StockChart
              history={result.history}
              predictions={result.predictions}
              ticker={result.ticker}
            />
          </div>

          {/* Prediction Table */}
          <div style={{ marginTop: 24 }}>
            <PredictionTable
              predictions={result.predictions}
              ticker={result.ticker}
            />
          </div>
        </>
      )}

      {/* Empty state when no results */}
      {!result && !trainResult && !loading && !training && (
        <div className="card fade-in" style={{ marginTop: 24 }}>
          <div className="empty-state">
            <div className="empty-state__icon">🚀</div>
            <div className="empty-state__title">Ready to Predict</div>
            <div className="empty-state__text">
              Enter a stock ticker symbol (e.g., AAPL, GOOGL, MSFT) and click
              <strong> Train Model</strong> first, then <strong>Predict</strong> to
              see AI-powered price forecasts.
            </div>
          </div>
        </div>
      )}

      {/* Footer */}
      <footer className="footer">
        Stock Price Prediction System — Built with ML &amp; MERN Stack
        <br />
        <span style={{ fontSize: 11 }}>
          Models: Linear Regression · Decision Tree · Random Forest · Logistic Regression
        </span>
      </footer>
    </div>
  );
}
