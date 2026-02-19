/**
 * StockChart.jsx — Stock Price Chart
 * =====================================
 * Displays historical and predicted stock prices using Recharts.
 * Historical data shown in indigo, forecast shown in cyan with dashed line.
 */

import React from "react";
import {
  ResponsiveContainer,
  ComposedChart,
  Line,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ReferenceLine,
} from "recharts";
import { FiBarChart2 } from "react-icons/fi";

/**
 * Custom tooltip for the chart
 */
function CustomTooltip({ active, payload, label }) {
  if (!active || !payload || payload.length === 0) return null;

  return (
    <div
      style={{
        background: "rgba(17, 24, 39, 0.95)",
        border: "1px solid rgba(99, 102, 241, 0.3)",
        borderRadius: 8,
        padding: "10px 14px",
        backdropFilter: "blur(10px)",
      }}
    >
      <div style={{ fontSize: 12, color: "#94a3b8", marginBottom: 6 }}>{label}</div>
      {payload.map((entry, idx) => (
        <div
          key={idx}
          style={{
            fontSize: 14,
            fontWeight: 600,
            color: entry.color,
            fontFamily: "var(--font-mono)",
          }}
        >
          {entry.name}: ${entry.value?.toFixed(2)}
        </div>
      ))}
    </div>
  );
}

export default function StockChart({ history, predictions, ticker }) {
  if (!history || history.length === 0) {
    return (
      <div className="card fade-in">
        <div className="card__header">
          <div className="card__icon card__icon--cyan">
            <FiBarChart2 />
          </div>
          <h2 className="card__title">Price Chart</h2>
        </div>
        <div className="empty-state">
          <div className="empty-state__icon">📈</div>
          <div className="empty-state__title">No data yet</div>
          <div className="empty-state__text">
            Enter a ticker symbol and click Predict to see the chart.
          </div>
        </div>
      </div>
    );
  }

  // Build chart data: history points have "history" key, forecast points have "forecast"
  // The last history point is duplicated into forecast for visual continuity.
  const chartData = [];

  // Add history
  history.forEach((point) => {
    chartData.push({
      date: point.date,
      history: point.close,
      forecast: null,
    });
  });

  // Bridge: last history point also gets a forecast value for line continuity
  if (predictions && predictions.length > 0 && chartData.length > 0) {
    chartData[chartData.length - 1].forecast = chartData[chartData.length - 1].history;

    // Add predictions
    predictions.forEach((point) => {
      chartData.push({
        date: point.date,
        history: null,
        forecast: point.close,
      });
    });
  }

  // Calculate Y-axis domain
  const allValues = chartData
    .flatMap((d) => [d.history, d.forecast])
    .filter((v) => v != null);
  const minVal = Math.min(...allValues);
  const maxVal = Math.max(...allValues);
  const padding = (maxVal - minVal) * 0.05;

  // Find the split date (last history date)
  const splitDate = history[history.length - 1]?.date;

  return (
    <div className="card fade-in stagger-2">
      <div className="card__header">
        <div className="card__icon card__icon--cyan">
          <FiBarChart2 />
        </div>
        <h2 className="card__title">
          {ticker} Price Chart
          {predictions && predictions.length > 0 && (
            <span style={{ fontSize: 13, color: "var(--text-muted)", fontWeight: 400, marginLeft: 8 }}>
              +{predictions.length} day forecast
            </span>
          )}
        </h2>
      </div>

      <div className="chart-container">
        <ResponsiveContainer width="100%" height="100%">
          <ComposedChart
            data={chartData}
            margin={{ top: 10, right: 30, left: 10, bottom: 0 }}
          >
            <defs>
              <linearGradient id="historyGradient" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#6366f1" stopOpacity={0.2} />
                <stop offset="95%" stopColor="#6366f1" stopOpacity={0} />
              </linearGradient>
              <linearGradient id="forecastGradient" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#06b6d4" stopOpacity={0.2} />
                <stop offset="95%" stopColor="#06b6d4" stopOpacity={0} />
              </linearGradient>
            </defs>

            <CartesianGrid
              strokeDasharray="3 3"
              stroke="rgba(255,255,255,0.05)"
              vertical={false}
            />
            <XAxis
              dataKey="date"
              tick={{ fontSize: 11, fill: "#64748b" }}
              tickLine={false}
              axisLine={{ stroke: "rgba(255,255,255,0.08)" }}
              interval="preserveStartEnd"
              minTickGap={40}
            />
            <YAxis
              domain={[minVal - padding, maxVal + padding]}
              tick={{ fontSize: 11, fill: "#64748b" }}
              tickLine={false}
              axisLine={false}
              tickFormatter={(v) => `$${v.toFixed(0)}`}
            />
            <Tooltip content={<CustomTooltip />} />
            <Legend
              wrapperStyle={{ fontSize: 12, color: "#94a3b8" }}
            />

            {/* Split reference line */}
            {splitDate && predictions && predictions.length > 0 && (
              <ReferenceLine
                x={splitDate}
                stroke="rgba(255,255,255,0.15)"
                strokeDasharray="5 5"
                label={{
                  value: "Forecast →",
                  position: "top",
                  fill: "#64748b",
                  fontSize: 11,
                }}
              />
            )}

            {/* Historical area + line */}
            <Area
              type="monotone"
              dataKey="history"
              fill="url(#historyGradient)"
              stroke="none"
              connectNulls={false}
            />
            <Line
              type="monotone"
              dataKey="history"
              stroke="#6366f1"
              strokeWidth={2}
              dot={false}
              name="Historical"
              connectNulls={false}
            />

            {/* Forecast area + line */}
            <Area
              type="monotone"
              dataKey="forecast"
              fill="url(#forecastGradient)"
              stroke="none"
              connectNulls={false}
            />
            <Line
              type="monotone"
              dataKey="forecast"
              stroke="#06b6d4"
              strokeWidth={2}
              strokeDasharray="6 3"
              dot={{ r: 4, fill: "#06b6d4", stroke: "#0a0e1a", strokeWidth: 2 }}
              name="Forecast"
              connectNulls={false}
            />
          </ComposedChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
