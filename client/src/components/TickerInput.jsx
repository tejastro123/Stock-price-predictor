/**
 * TickerInput.jsx — Stock Ticker Input Form
 * ============================================
 * Input form for ticker symbol, forecast days, and action buttons.
 * Handles both training and prediction requests.
 * Includes client-side ticker validation (format + existence check).
 */

import React, { useState } from "react";
import axios from "axios";
import { FiSearch, FiCpu, FiZap, FiAlertTriangle } from "react-icons/fi";

const API_BASE = import.meta.env.VITE_API_BASE_URL || "/api";
const TICKER_REGEX = /^[A-Z]{1,5}$/;

export default function TickerInput({ onPredict, onTrain, loading, training }) {
  const [ticker, setTicker] = useState("AAPL");
  const [days, setDays] = useState(5);
  const [tickerError, setTickerError] = useState(null);
  const [validating, setValidating] = useState(false);

  /**
   * Validate ticker format and existence via /api/info endpoint.
   * Returns true if valid, false otherwise (and sets tickerError).
   */
  const validateTicker = async (symbol) => {
    const clean = symbol.trim().toUpperCase();

    // Format check
    if (!clean) {
      setTickerError("Please enter a ticker symbol.");
      return false;
    }
    if (!TICKER_REGEX.test(clean)) {
      setTickerError("Ticker must be 1–5 letters (e.g. AAPL, TSLA).");
      return false;
    }

    // Existence check via API
    setValidating(true);
    try {
      await axios.get(`${API_BASE}/info/${clean}`, { timeout: 10000 });
      setTickerError(null);
      return true;
    } catch (err) {
      const status = err.response?.status;
      if (status === 404 || status === 400) {
        setTickerError(`Ticker "${clean}" not found. Check the symbol and try again.`);
      } else {
        // Network or server error — allow the user to proceed (don't block)
        setTickerError(null);
        return true;
      }
      return false;
    } finally {
      setValidating(false);
    }
  };

  const handlePredict = async (e) => {
    e.preventDefault();
    const isValid = await validateTicker(ticker);
    if (isValid) {
      onPredict(ticker.trim().toUpperCase(), days);
    }
  };

  const handleTrain = async () => {
    const isValid = await validateTicker(ticker);
    if (isValid) {
      onTrain(ticker.trim().toUpperCase());
    }
  };

  return (
    <div className="card fade-in">
      <div className="card__header">
        <div className="card__icon card__icon--purple">
          <FiSearch />
        </div>
        <h2 className="card__title">Stock Prediction</h2>
      </div>

      <form onSubmit={handlePredict}>
        <div className="input-group">
          <div className="input-field">
            <label htmlFor="ticker-input">Ticker Symbol</label>
            <input
              id="ticker-input"
              type="text"
              value={ticker}
              onChange={(e) => {
                setTicker(e.target.value.toUpperCase());
                setTickerError(null); // Clear error on change
              }}
              placeholder="e.g. AAPL, GOOGL, TSLA"
              maxLength={5}
              required
              style={tickerError ? { borderColor: "#ef4444" } : {}}
            />
          </div>

          <div className="input-field" style={{ maxWidth: 160 }}>
            <label htmlFor="days-select">Forecast Days</label>
            <select
              id="days-select"
              value={days}
              onChange={(e) => setDays(parseInt(e.target.value))}
            >
              {[1, 3, 5, 7, 10, 14, 21, 30].map((d) => (
                <option key={d} value={d}>
                  {d} day{d > 1 ? "s" : ""}
                </option>
              ))}
            </select>
          </div>

          <div style={{ display: "flex", gap: 10, alignItems: "flex-end" }}>
            <button
              type="button"
              className={`btn btn--secondary ${training ? "btn--loading" : ""}`}
              onClick={handleTrain}
              disabled={loading || training || validating}
              title="Train model for this ticker first"
            >
              <FiCpu />
              {training ? "Training..." : validating ? "Validating..." : "Train Model"}
            </button>

            <button
              type="submit"
              className={`btn btn--primary ${loading ? "btn--loading" : ""}`}
              disabled={loading || training || validating}
            >
              <FiZap />
              {loading ? "Predicting..." : validating ? "Validating..." : "Predict"}
            </button>
          </div>
        </div>
      </form>

      {/* Ticker validation error */}
      {tickerError && (
        <div
          style={{
            marginTop: 10,
            padding: "8px 12px",
            borderRadius: 8,
            background: "rgba(239, 68, 68, 0.1)",
            border: "1px solid rgba(239, 68, 68, 0.3)",
            color: "#ef4444",
            fontSize: 12,
            display: "flex",
            alignItems: "center",
            gap: 6,
          }}
        >
          <FiAlertTriangle />
          {tickerError}
        </div>
      )}

      <div style={{ marginTop: 12, fontSize: 12, color: "var(--text-muted)" }}>
        💡 First time? Click <strong>Train Model</strong> to build the ML model, then <strong>Predict</strong>.
      </div>
    </div>
  );
}

