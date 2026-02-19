/**
 * CompareMode.jsx — Multi-Ticker Comparison
 * ============================================
 * Allows users to compare predictions for up to 4 tickers side-by-side.
 * Features:
 *   • Multi-ticker chip input
 *   • Overlay chart with color-coded forecast lines
 *   • Side-by-side metrics comparison table
 */

import React, { useState } from "react";
import axios from "axios";
import {
  ResponsiveContainer,
  ComposedChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ReferenceLine,
} from "recharts";
import { FiBarChart2, FiX, FiPlus, FiTrendingUp } from "react-icons/fi";

const API_BASE = "/api";

// Color palette for different tickers
const TICKER_COLORS = ["#6366f1", "#06b6d4", "#f59e0b", "#10b981"];
const TICKER_BG = [
  "rgba(99, 102, 241, 0.12)",
  "rgba(6, 182, 212, 0.12)",
  "rgba(245, 158, 11, 0.12)",
  "rgba(16, 185, 129, 0.12)",
];

export default function CompareMode() {
  const [tickers, setTickers] = useState([]);
  const [inputValue, setInputValue] = useState("");
  const [days, setDays] = useState(7);
  const [results, setResults] = useState({});  // { AAPL: {...}, GOOGL: {...} }
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  // ── Ticker Management ────────────────────────────────────────────
  const addTicker = () => {
    const ticker = inputValue.trim().toUpperCase();
    if (!ticker || tickers.includes(ticker) || tickers.length >= 4) return;
    setTickers([...tickers, ticker]);
    setInputValue("");
  };

  const removeTicker = (ticker) => {
    setTickers(tickers.filter((t) => t !== ticker));
    const newResults = { ...results };
    delete newResults[ticker];
    setResults(newResults);
  };

  const handleKeyDown = (e) => {
    if (e.key === "Enter") {
      e.preventDefault();
      addTicker();
    }
  };

  // ── Fetch All Predictions ────────────────────────────────────────
  const handleCompare = async () => {
    if (tickers.length < 2) {
      setError("Add at least 2 tickers to compare.");
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const promises = tickers.map((ticker) =>
        axios.post(`${API_BASE}/predict`, { ticker, days })
      );
      const responses = await Promise.all(promises);

      const newResults = {};
      responses.forEach((res, idx) => {
        newResults[tickers[idx]] = res.data;
      });
      setResults(newResults);
    } catch (err) {
      const msg =
        err.response?.data?.error ||
        err.response?.data?.detail ||
        "Comparison failed. Ensure models are trained for all tickers.";
      setError(msg);
    } finally {
      setLoading(false);
    }
  };

  // ── Build overlay chart data ─────────────────────────────────────
  const hasResults = Object.keys(results).length >= 2;

  const buildChartData = () => {
    if (!hasResults) return [];

    // Collect all unique dates across all tickers
    const dateSet = new Set();
    Object.values(results).forEach((r) => {
      r.predictions?.forEach((p) => dateSet.add(p.date));
    });

    const dates = [...dateSet].sort();
    return dates.map((date) => {
      const point = { date };
      Object.entries(results).forEach(([ticker, r]) => {
        const pred = r.predictions?.find((p) => p.date === date);
        if (pred) point[ticker] = pred.close;
      });
      return point;
    });
  };

  const chartData = buildChartData();

  // Y-axis domain
  const allPrices = chartData.flatMap((d) =>
    Object.entries(d)
      .filter(([k]) => k !== "date")
      .map(([, v]) => v)
  ).filter(Boolean);
  const minPrice = allPrices.length ? Math.min(...allPrices) : 0;
  const maxPrice = allPrices.length ? Math.max(...allPrices) : 100;

  // Normalize for percentage comparison
  const buildNormalizedData = () => {
    if (!hasResults) return [];

    const dateSet = new Set();
    Object.values(results).forEach((r) => {
      r.predictions?.forEach((p) => dateSet.add(p.date));
    });

    const dates = [...dateSet].sort();
    return dates.map((date) => {
      const point = { date };
      Object.entries(results).forEach(([ticker, r]) => {
        const preds = r.predictions || [];
        const basePrice = preds[0]?.close || 1;
        const pred = preds.find((p) => p.date === date);
        if (pred) {
          point[`${ticker}_%`] = (((pred.close - basePrice) / basePrice) * 100).toFixed(2);
        }
      });
      return point;
    });
  };

  const activeTickers = Object.keys(results);

  return (
    <div className="compare-mode">
      {/* Ticker Input Area */}
      <div className="card fade-in">
        <div className="card__header">
          <div className="card__icon card__icon--indigo">
            <FiTrendingUp />
          </div>
          <h2 className="card__title">Compare Tickers</h2>
        </div>

        <div className="compare-input-area">
          {/* Ticker Chips */}
          <div className="compare-chips">
            {tickers.map((ticker, idx) => (
              <span
                key={ticker}
                className="compare-chip"
                style={{
                  background: TICKER_BG[idx % TICKER_BG.length],
                  borderColor: TICKER_COLORS[idx % TICKER_COLORS.length],
                  color: TICKER_COLORS[idx % TICKER_COLORS.length],
                }}
              >
                {ticker}
                <button
                  className="compare-chip__remove"
                  onClick={() => removeTicker(ticker)}
                  aria-label={`Remove ${ticker}`}
                >
                  <FiX size={12} />
                </button>
              </span>
            ))}

            {/* Add ticker input */}
            {tickers.length < 4 && (
              <div className="compare-add">
                <input
                  type="text"
                  value={inputValue}
                  onChange={(e) => setInputValue(e.target.value.toUpperCase())}
                  onKeyDown={handleKeyDown}
                  placeholder="Add ticker..."
                  className="compare-add__input"
                  maxLength={5}
                />
                <button className="compare-add__btn" onClick={addTicker}>
                  <FiPlus size={14} />
                </button>
              </div>
            )}
          </div>

          {/* Days + Compare Button */}
          <div className="compare-actions">
            <label className="compare-days">
              <span style={{ fontSize: 12, color: "var(--text-muted)" }}>Days</span>
              <select
                value={days}
                onChange={(e) => setDays(parseInt(e.target.value))}
                className="input-field"
                style={{ width: 70, padding: "6px 8px", fontSize: 13 }}
              >
                {[5, 7, 14, 21, 30].map((d) => (
                  <option key={d} value={d}>{d}</option>
                ))}
              </select>
            </label>
            <button
              className="btn btn--primary"
              onClick={handleCompare}
              disabled={loading || tickers.length < 2}
              style={{ padding: "8px 20px" }}
            >
              {loading ? (
                <><div className="spinner" style={{ width: 14, height: 14 }} /> Comparing...</>
              ) : (
                "Compare"
              )}
            </button>
          </div>
        </div>

        {tickers.length === 0 && (
          <div style={{ fontSize: 12, color: "var(--text-muted)", marginTop: 8 }}>
            Add 2–4 tickers to compare their predictions side-by-side.
          </div>
        )}

        {error && (
          <div className="status status--error fade-in" style={{ marginTop: 12 }}>
            ⚠️ {error}
          </div>
        )}
      </div>

      {/* Overlay Chart */}
      {hasResults && (
        <div className="card fade-in" style={{ marginTop: 24 }}>
          <div className="card__header">
            <div className="card__icon card__icon--cyan">
              <FiBarChart2 />
            </div>
            <h2 className="card__title">
              Forecast Comparison — {days} Days
            </h2>
          </div>

          <div className="chart-container">
            <ResponsiveContainer width="100%" height="100%">
              <ComposedChart data={chartData} margin={{ top: 10, right: 30, left: 10, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="var(--chart-grid)" vertical={false} />
                <XAxis
                  dataKey="date"
                  tick={{ fontSize: 11, fill: "var(--chart-tick)" }}
                  tickLine={false}
                  axisLine={{ stroke: "var(--chart-axis)" }}
                  minTickGap={40}
                />
                <YAxis
                  tick={{ fontSize: 11, fill: "var(--chart-tick)" }}
                  tickLine={false}
                  axisLine={false}
                  tickFormatter={(v) => `$${v.toFixed(0)}`}
                />
                <Tooltip
                  contentStyle={{
                    background: "var(--tooltip-bg)",
                    border: "1px solid var(--tooltip-border)",
                    borderRadius: 8,
                    backdropFilter: "blur(10px)",
                    fontSize: 13,
                  }}
                  formatter={(value, name) => [`$${Number(value).toFixed(2)}`, name]}
                  labelStyle={{ color: "var(--text-muted)", fontSize: 12 }}
                />
                <Legend wrapperStyle={{ fontSize: 12 }} />

                {activeTickers.map((ticker, idx) => (
                  <Line
                    key={ticker}
                    type="monotone"
                    dataKey={ticker}
                    stroke={TICKER_COLORS[tickers.indexOf(ticker) % TICKER_COLORS.length]}
                    strokeWidth={2}
                    dot={{ r: 3 }}
                    name={ticker}
                    connectNulls={false}
                    isAnimationActive={true}
                    animationDuration={800 + idx * 300}
                  />
                ))}
              </ComposedChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}

      {/* Side-by-Side Metrics Table */}
      {hasResults && (
        <div className="card fade-in" style={{ marginTop: 24 }}>
          <div className="card__header">
            <div className="card__icon card__icon--green">
              <FiTrendingUp />
            </div>
            <h2 className="card__title">Comparison Summary</h2>
          </div>

          <div className="mobile-scroll-wrapper">
            <table className="compare-table">
              <thead>
                <tr>
                  <th>Metric</th>
                  {activeTickers.map((ticker, idx) => (
                    <th key={ticker} style={{ color: TICKER_COLORS[tickers.indexOf(ticker) % TICKER_COLORS.length] }}>
                      {ticker}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                <tr>
                  <td className="compare-table__label">Model Used</td>
                  {activeTickers.map((t) => (
                    <td key={t}>{results[t].model_used}</td>
                  ))}
                </tr>
                <tr>
                  <td className="compare-table__label">Current Price</td>
                  {activeTickers.map((t) => {
                    const hist = results[t].history;
                    return <td key={t}>${hist?.[hist.length - 1]?.close?.toFixed(2) || "—"}</td>;
                  })}
                </tr>
                <tr>
                  <td className="compare-table__label">Day 1 Forecast</td>
                  {activeTickers.map((t) => (
                    <td key={t}>${results[t].predictions?.[0]?.close?.toFixed(2) || "—"}</td>
                  ))}
                </tr>
                <tr>
                  <td className="compare-table__label">Final Forecast</td>
                  {activeTickers.map((t) => {
                    const preds = results[t].predictions || [];
                    return <td key={t}>${preds[preds.length - 1]?.close?.toFixed(2) || "—"}</td>;
                  })}
                </tr>
                <tr>
                  <td className="compare-table__label">Predicted Change</td>
                  {activeTickers.map((t) => {
                    const hist = results[t].history;
                    const preds = results[t].predictions || [];
                    const current = hist?.[hist.length - 1]?.close;
                    const final_ = preds[preds.length - 1]?.close;
                    if (!current || !final_) return <td key={t}>—</td>;
                    const change = ((final_ - current) / current * 100).toFixed(2);
                    const isUp = change >= 0;
                    return (
                      <td key={t} style={{ color: isUp ? "#10b981" : "#ef4444", fontWeight: 600 }}>
                        {isUp ? "▲" : "▼"} {Math.abs(change)}%
                      </td>
                    );
                  })}
                </tr>
                <tr>
                  <td className="compare-table__label">RMSE</td>
                  {activeTickers.map((t) => (
                    <td key={t}>{results[t].metrics?.rmse?.toFixed(4) || "—"}</td>
                  ))}
                </tr>
                <tr>
                  <td className="compare-table__label">MAE</td>
                  {activeTickers.map((t) => (
                    <td key={t}>{results[t].metrics?.mae?.toFixed(4) || "—"}</td>
                  ))}
                </tr>
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}
