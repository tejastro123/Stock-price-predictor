/**
 * ModelComparison.jsx — Model Comparison Visualization
 * =====================================================
 * Displays all trained models' metrics as a visual comparison chart
 * and detailed table, so users can see how each model performed.
 */

import React, { useState } from "react";
import { FiAward } from "react-icons/fi";

const METRIC_LABELS = {
  rmse: { label: "RMSE", lower: true, unit: "", desc: "Root Mean Squared Error" },
  mae: { label: "MAE", lower: true, unit: "", desc: "Mean Absolute Error" },
  mape: { label: "MAPE", lower: true, unit: "%", desc: "Mean Abs % Error" },
  directional_accuracy: {
    label: "Dir. Accuracy",
    lower: false,
    unit: "%",
    desc: "Directional Accuracy",
  },
};

const BAR_COLORS = [
  "#6366f1", // indigo
  "#06b6d4", // cyan
  "#f59e0b", // amber
  "#10b981", // emerald
  "#ef4444", // red (for tuned)
  "#8b5cf6", // violet
];

export default function ModelComparison({ allResults, bestModel }) {
  const [selectedMetric, setSelectedMetric] = useState("rmse");

  if (!allResults || Object.keys(allResults).length === 0) return null;

  const modelNames = Object.keys(allResults);
  const metricInfo = METRIC_LABELS[selectedMetric];

  // Get values for selected metric and compute bar widths
  const values = modelNames.map((name) => ({
    name,
    value: allResults[name]?.[selectedMetric],
    isBest: name === bestModel,
  }));

  const validValues = values
    .map((v) => v.value)
    .filter((v) => v != null && !isNaN(v));
  const maxVal = Math.max(...validValues) || 1;

  return (
    <div className="card fade-in stagger-1">
      <div className="card__header">
        <div className="card__icon card__icon--amber">
          <FiAward />
        </div>
        <h2 className="card__title">
          Model Comparison
          <span
            style={{
              fontSize: 13,
              color: "var(--text-muted)",
              fontWeight: 400,
              marginLeft: 8,
            }}
          >
            {modelNames.length} models trained
          </span>
        </h2>
      </div>

      {/* Metric selector tabs */}
      <div className="mc-tabs">
        {Object.entries(METRIC_LABELS).map(([key, info]) => (
          <button
            key={key}
            className={`mc-tab ${selectedMetric === key ? "mc-tab--active" : ""}`}
            onClick={() => setSelectedMetric(key)}
          >
            {info.label}
          </button>
        ))}
      </div>

      {/* Bar chart */}
      <div className="mc-chart">
        {values.map((v, idx) => {
          if (v.value == null) return null;
          const pct = (v.value / maxVal) * 100;
          const color = BAR_COLORS[idx % BAR_COLORS.length];

          return (
            <div key={v.name} className="mc-bar-row">
              <div className="mc-bar-row__label">
                {v.name}
                {v.isBest && <span className="mc-best-badge">★ Best</span>}
              </div>
              <div className="mc-bar-row__track">
                <div
                  className="mc-bar-row__fill"
                  style={{
                    width: `${pct}%`,
                    backgroundColor: color,
                    animationDelay: `${idx * 80}ms`,
                  }}
                />
              </div>
              <div className="mc-bar-row__value">
                {v.value.toFixed(2)}
                {metricInfo.unit}
              </div>
            </div>
          );
        })}
      </div>

      {/* Legend note */}
      <div
        style={{
          fontSize: 11,
          color: "var(--text-muted)",
          textAlign: "center",
          marginTop: 8,
        }}
      >
        {metricInfo.lower ? "↓ Lower is better" : "↑ Higher is better"} •{" "}
        {metricInfo.desc}
      </div>

      {/* Full metrics table */}
      <div style={{ overflowX: "auto", marginTop: 20 }}>
        <table className="prediction-table">
          <thead>
            <tr>
              <th>Model</th>
              <th>RMSE</th>
              <th>MAE</th>
              <th>MAPE</th>
              <th>Dir. Accuracy</th>
            </tr>
          </thead>
          <tbody>
            {modelNames.map((name) => {
              const m = allResults[name];
              const isBest = name === bestModel;
              return (
                <tr
                  key={name}
                  style={
                    isBest
                      ? {
                        background: "rgba(99, 102, 241, 0.08)",
                        borderLeft: "3px solid #6366f1",
                      }
                      : {}
                  }
                >
                  <td
                    className="day-cell"
                    style={{ fontWeight: isBest ? 700 : 500 }}
                  >
                    {name}{" "}
                    {isBest && (
                      <span style={{ color: "#f59e0b", fontSize: 12 }}>★</span>
                    )}
                  </td>
                  <td className="price-cell">
                    {m.rmse != null ? m.rmse.toFixed(4) : "—"}
                  </td>
                  <td className="price-cell">
                    {m.mae != null ? m.mae.toFixed(4) : "—"}
                  </td>
                  <td className="price-cell">
                    {m.mape != null ? `${m.mape.toFixed(2)}%` : "—"}
                  </td>
                  <td className="price-cell">
                    {m.directional_accuracy != null
                      ? `${m.directional_accuracy}%`
                      : "—"}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}
