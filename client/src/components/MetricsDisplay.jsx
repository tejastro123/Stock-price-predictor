/**
 * MetricsDisplay.jsx — Model Performance Metrics
 * =================================================
 * Displays RMSE, MAE, MAPE, and Directional Accuracy as metric cards.
 */

import React from "react";
import { FiTarget, FiBarChart2, FiPercent, FiTrendingUp } from "react-icons/fi";

const metricConfig = [
  {
    key: "rmse",
    label: "RMSE",
    icon: <FiTarget />,
    format: (v) => (v != null ? v.toFixed(4) : "N/A"),
    description: "Root Mean Squared Error",
  },
  {
    key: "mae",
    label: "MAE",
    icon: <FiBarChart2 />,
    format: (v) => (v != null ? v.toFixed(4) : "N/A"),
    description: "Mean Absolute Error",
  },
  {
    key: "mape",
    label: "MAPE",
    icon: <FiPercent />,
    format: (v) => (v != null ? v.toFixed(2) + "%" : "N/A"),
    description: "Mean Abs Percentage Error",
  },
  {
    key: "directional_accuracy",
    label: "Direction Acc.",
    icon: <FiTrendingUp />,
    format: (v) => (v != null ? v.toFixed(1) + "%" : "N/A"),
    description: "Up/Down Accuracy",
    isGreen: true,
  },
];

export default function MetricsDisplay({ metrics, modelName }) {
  if (!metrics) return null;

  return (
    <div className="fade-in">
      <div style={{ display: "flex", alignItems: "center", gap: 12, marginBottom: 16 }}>
        <span className="model-badge">
          <FiTarget size={14} />
          {modelName || "Model"}
        </span>
      </div>

      <div className="metrics-grid">
        {metricConfig.map((m, idx) => (
          <div
            key={m.key}
            className={`metric-card fade-in stagger-${idx + 1}`}
          >
            <div className="metric-card__label">
              {m.label}
            </div>
            <div className={`metric-card__value ${m.isGreen ? "metric-card__value--green" : ""}`}>
              {m.format(metrics[m.key])}
            </div>
            <div style={{ fontSize: 11, color: "var(--text-muted)", marginTop: 4 }}>
              {m.description}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
