/**
 * PredictionTable.jsx — Forecast Results Table
 * ================================================
 * Displays predicted prices in a styled table with day numbering.
 */

import React from "react";
import { FiCalendar } from "react-icons/fi";

export default function PredictionTable({ predictions, ticker }) {
  if (!predictions || predictions.length === 0) return null;

  return (
    <div className="card fade-in stagger-3">
      <div className="card__header">
        <div className="card__icon card__icon--green">
          <FiCalendar />
        </div>
        <h2 className="card__title">
          {ticker} Forecast
          <span style={{ fontSize: 13, color: "var(--text-muted)", fontWeight: 400, marginLeft: 8 }}>
            {predictions.length} trading day{predictions.length > 1 ? "s" : ""}
          </span>
        </h2>
      </div>

      <div style={{ overflowX: "auto" }}>
        <table className="prediction-table">
          <thead>
            <tr>
              <th>Day</th>
              <th>Date</th>
              <th>Predicted Close</th>
              <th>Change</th>
            </tr>
          </thead>
          <tbody>
            {predictions.map((pred, idx) => {
              const prevPrice =
                idx === 0
                  ? predictions[0].close
                  : predictions[idx - 1].close;
              const change = idx === 0 ? 0 : pred.close - prevPrice;
              const changePercent =
                idx === 0 ? 0 : (change / prevPrice) * 100;
              const isUp = change >= 0;

              return (
                <tr key={pred.date}>
                  <td className="day-cell">Day {idx + 1}</td>
                  <td className="date-cell">{pred.date}</td>
                  <td className="price-cell">${pred.close.toFixed(2)}</td>
                  <td
                    style={{
                      color: idx === 0 ? "var(--text-muted)" : isUp ? "#10b981" : "#ef4444",
                      fontFamily: "var(--font-mono)",
                      fontSize: 13,
                      fontWeight: 500,
                    }}
                  >
                    {idx === 0
                      ? "—"
                      : `${isUp ? "+" : ""}${change.toFixed(2)} (${isUp ? "+" : ""}${changePercent.toFixed(2)}%)`}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      {/* Summary row */}
      {predictions.length > 1 && (
        <div
          style={{
            display: "flex",
            justifyContent: "space-between",
            alignItems: "center",
            marginTop: 16,
            padding: "12px 16px",
            background: "var(--bg-glass)",
            borderRadius: "var(--radius-sm)",
            fontSize: 13,
          }}
        >
          <span style={{ color: "var(--text-muted)" }}>
            Total forecast range
          </span>
          <span style={{ fontFamily: "var(--font-mono)", fontWeight: 600 }}>
            ${predictions[0].close.toFixed(2)} → ${predictions[predictions.length - 1].close.toFixed(2)}
            <span
              style={{
                marginLeft: 8,
                color:
                  predictions[predictions.length - 1].close >=
                    predictions[0].close
                    ? "#10b981"
                    : "#ef4444",
              }}
            >
              ({(
                ((predictions[predictions.length - 1].close -
                  predictions[0].close) /
                  predictions[0].close) *
                100
              ).toFixed(2)}
              %)
            </span>
          </span>
        </div>
      )}
    </div>
  );
}
