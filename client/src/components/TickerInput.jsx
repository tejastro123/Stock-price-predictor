/**
 * TickerInput.jsx — Stock Ticker Input Form
 * ============================================
 * Input form for ticker symbol, forecast days, and action buttons.
 * Handles both training and prediction requests.
 */

import React, { useState } from "react";
import { FiSearch, FiCpu, FiZap } from "react-icons/fi";

export default function TickerInput({ onPredict, onTrain, loading, training }) {
  const [ticker, setTicker] = useState("AAPL");
  const [days, setDays] = useState(5);

  const handlePredict = (e) => {
    e.preventDefault();
    if (ticker.trim()) {
      onPredict(ticker.trim().toUpperCase(), days);
    }
  };

  const handleTrain = () => {
    if (ticker.trim()) {
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
              onChange={(e) => setTicker(e.target.value.toUpperCase())}
              placeholder="e.g. AAPL, GOOGL, TSLA"
              maxLength={10}
              required
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
              disabled={loading || training}
              title="Train model for this ticker first"
            >
              <FiCpu />
              {training ? "Training..." : "Train Model"}
            </button>

            <button
              type="submit"
              className={`btn btn--primary ${loading ? "btn--loading" : ""}`}
              disabled={loading || training}
            >
              <FiZap />
              {loading ? "Predicting..." : "Predict"}
            </button>
          </div>
        </div>
      </form>

      <div style={{ marginTop: 12, fontSize: 12, color: "var(--text-muted)" }}>
        💡 First time? Click <strong>Train Model</strong> to build the ML model, then <strong>Predict</strong>.
      </div>
    </div>
  );
}
