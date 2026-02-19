/**
 * StockChart.jsx — Enhanced Stock Price Chart
 * ==============================================
 * Features:
 *   • Line / Candlestick chart toggle
 *   • Volume bar chart below main price chart
 *   • Time range selector (1W, 1M, 3M, ALL)
 *   • Animated forecast line drawing
 *   • 95% Confidence bands
 *
 * Uses Recharts ComposedChart for maximum flexibility.
 */

import React, { useState, useEffect } from "react";
import {
  ResponsiveContainer,
  ComposedChart,
  Line,
  Area,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ReferenceLine,
  Cell,
  Rectangle,
} from "recharts";
import { FiBarChart2 } from "react-icons/fi";

// ── Time Range Options ───────────────────────────────────────────────
const TIME_RANGES = [
  { label: "1W", days: 5 },
  { label: "1M", days: 21 },
  { label: "3M", days: 63 },
  { label: "ALL", days: Infinity },
];

// ── Custom Candlestick Shape ─────────────────────────────────────────
function CandlestickBar(props) {
  const { x, y, width, height, payload } = props;
  if (!payload || payload.open == null) return null;

  const { open, high, low, close } = payload;
  const isUp = close >= open;
  const color = isUp ? "#10b981" : "#ef4444";
  const bodyWidth = Math.max(width * 0.6, 4);

  // Scale: we need to map price → pixel. Use y and height as reference.
  // The bar is positioned by Recharts based on a dummy "candle" value.
  // We need to calculate pixel positions from OHLC values.
  const yScale = props.background?.height || 300;
  const yMin = props.yAxisMin || 0;
  const yMax = props.yAxisMax || 100;
  const pxPerUnit = yScale / (yMax - yMin);

  const topY = props.background?.y || 0;
  const priceToY = (price) => topY + (yMax - price) * pxPerUnit;

  const openY = priceToY(open);
  const closeY = priceToY(close);
  const highY = priceToY(high);
  const lowY = priceToY(low);

  const bodyTop = Math.min(openY, closeY);
  const bodyHeight = Math.max(Math.abs(openY - closeY), 1);
  const cx = x + width / 2;

  return (
    <g>
      {/* Wick */}
      <line
        x1={cx}
        y1={highY}
        x2={cx}
        y2={lowY}
        stroke={color}
        strokeWidth={1}
      />
      {/* Body */}
      <rect
        x={cx - bodyWidth / 2}
        y={bodyTop}
        width={bodyWidth}
        height={bodyHeight}
        fill={isUp ? color : color}
        stroke={color}
        strokeWidth={1}
        rx={1}
      />
    </g>
  );
}

// ── Custom Tooltip ───────────────────────────────────────────────────
function CustomTooltip({ active, payload, label, chartMode }) {
  if (!active || !payload || payload.length === 0) return null;

  const dataPoint = payload[0]?.payload;
  const hasConfidence = dataPoint?.upper != null && dataPoint?.lower != null;
  const hasOHLC = dataPoint?.open != null;

  return (
    <div
      style={{
        background: "var(--tooltip-bg)",
        border: "1px solid var(--tooltip-border)",
        borderRadius: 8,
        padding: "10px 14px",
        backdropFilter: "blur(10px)",
      }}
    >
      <div style={{ fontSize: 12, color: "var(--text-muted)", marginBottom: 6 }}>
        {label}
      </div>

      {/* OHLC data for candlestick mode */}
      {hasOHLC && chartMode === "candlestick" && (
        <div style={{ fontFamily: "var(--font-mono)", fontSize: 12, lineHeight: 1.6 }}>
          <div>
            <span style={{ color: "var(--text-muted)" }}>O:</span>{" "}
            <span style={{ color: "var(--text-primary)" }}>${dataPoint.open?.toFixed(2)}</span>
          </div>
          <div>
            <span style={{ color: "var(--text-muted)" }}>H:</span>{" "}
            <span style={{ color: "#10b981" }}>${dataPoint.high?.toFixed(2)}</span>
          </div>
          <div>
            <span style={{ color: "var(--text-muted)" }}>L:</span>{" "}
            <span style={{ color: "#ef4444" }}>${dataPoint.low?.toFixed(2)}</span>
          </div>
          <div>
            <span style={{ color: "var(--text-muted)" }}>C:</span>{" "}
            <span style={{ color: "var(--text-primary)", fontWeight: 600 }}>
              ${dataPoint.close?.toFixed(2)}
            </span>
          </div>
        </div>
      )}

      {/* Line mode values */}
      {chartMode === "line" &&
        payload
          .filter((entry) => entry.dataKey !== "confidenceBand" && entry.dataKey !== "volume")
          .map((entry, idx) => (
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

      {/* Volume */}
      {dataPoint?.volume != null && (
        <div style={{ fontSize: 11, color: "var(--text-muted)", marginTop: 4 }}>
          Vol: {(dataPoint.volume / 1e6).toFixed(2)}M
        </div>
      )}

      {/* Confidence interval */}
      {hasConfidence && (
        <div
          style={{
            fontSize: 12,
            color: "#8b5cf6",
            fontFamily: "var(--font-mono)",
            marginTop: 4,
            borderTop: "1px solid var(--border-color)",
            paddingTop: 4,
          }}
        >
          95% CI: ${dataPoint.lower.toFixed(2)} – ${dataPoint.upper.toFixed(2)}
        </div>
      )}
    </div>
  );
}

// ── Main Component ───────────────────────────────────────────────────
export default function StockChart({ history, predictions, ticker }) {
  const [chartMode, setChartMode] = useState("line"); // "line" | "candlestick"
  const [timeRange, setTimeRange] = useState("ALL");
  const [animatedPredictions, setAnimatedPredictions] = useState(0);

  // Animate forecast line drawing in
  useEffect(() => {
    if (!predictions || predictions.length === 0) {
      setAnimatedPredictions(0);
      return;
    }
    setAnimatedPredictions(0);
    let count = 0;
    const interval = setInterval(() => {
      count++;
      setAnimatedPredictions(count);
      if (count >= predictions.length) clearInterval(interval);
    }, 200);
    return () => clearInterval(interval);
  }, [predictions]);

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

  // ── Filter history by time range ─────────────────────────────────
  const rangeConfig = TIME_RANGES.find((r) => r.label === timeRange);
  const filteredHistory =
    rangeConfig.days === Infinity
      ? history
      : history.slice(-rangeConfig.days);

  // ── Build chart data ─────────────────────────────────────────────
  const chartData = [];

  filteredHistory.forEach((point) => {
    chartData.push({
      date: point.date,
      history: point.close,
      open: point.open,
      high: point.high,
      low: point.low,
      close: point.close,
      volume: point.volume,
      forecast: null,
      confidenceBand: null,
      upper: null,
      lower: null,
      candle: point.close, // dummy value for candlestick bar positioning
    });
  });

  // Bridge + animated predictions
  if (predictions && predictions.length > 0 && chartData.length > 0) {
    chartData[chartData.length - 1].forecast =
      chartData[chartData.length - 1].history;

    const visiblePreds = predictions.slice(0, animatedPredictions);
    visiblePreds.forEach((point) => {
      const hasCI = point.upper != null && point.lower != null;
      chartData.push({
        date: point.date,
        history: null,
        open: null,
        high: null,
        low: null,
        close: point.close,
        volume: null,
        forecast: point.close,
        confidenceBand: hasCI ? [point.lower, point.upper] : null,
        upper: hasCI ? point.upper : null,
        lower: hasCI ? point.lower : null,
        candle: null,
      });
    });
  }

  // ── Y-axis domain ────────────────────────────────────────────────
  const priceValues = chartData.flatMap((d) => {
    const vals = [d.history, d.forecast, d.upper, d.lower];
    if (chartMode === "candlestick" && d.high != null) vals.push(d.high, d.low);
    return vals;
  }).filter((v) => v != null);
  const minVal = Math.min(...priceValues);
  const maxVal = Math.max(...priceValues);
  const padding = (maxVal - minVal) * 0.05;

  // Volume max for secondary Y-axis
  const maxVolume = Math.max(
    ...chartData.map((d) => d.volume || 0).filter(Boolean)
  ) || 1;

  // Split date
  const splitDate = filteredHistory[filteredHistory.length - 1]?.date;

  return (
    <div className="card fade-in stagger-2">
      <div className="card__header">
        <div className="card__icon card__icon--cyan">
          <FiBarChart2 />
        </div>
        <h2 className="card__title">
          {ticker} Price Chart
          {predictions && predictions.length > 0 && (
            <span
              style={{
                fontSize: 13,
                color: "var(--text-muted)",
                fontWeight: 400,
                marginLeft: 8,
              }}
            >
              +{predictions.length} day forecast
            </span>
          )}
        </h2>
      </div>

      {/* Chart Controls */}
      <div className="chart-controls">
        {/* Chart type toggle */}
        <div className="mc-tabs" style={{ marginBottom: 0 }}>
          <button
            className={`mc-tab ${chartMode === "line" ? "mc-tab--active" : ""}`}
            onClick={() => setChartMode("line")}
          >
            📈 Line
          </button>
          <button
            className={`mc-tab ${chartMode === "candlestick" ? "mc-tab--active" : ""}`}
            onClick={() => setChartMode("candlestick")}
          >
            🕯️ Candlestick
          </button>
        </div>

        {/* Time range selector */}
        <div className="mc-tabs" style={{ marginBottom: 0 }}>
          {TIME_RANGES.map((range) => (
            <button
              key={range.label}
              className={`mc-tab ${timeRange === range.label ? "mc-tab--active" : ""}`}
              onClick={() => setTimeRange(range.label)}
            >
              {range.label}
            </button>
          ))}
        </div>
      </div>

      {/* Main Price Chart */}
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
              <linearGradient id="confidenceGradient" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor="#8b5cf6" stopOpacity={0.15} />
                <stop offset="50%" stopColor="#8b5cf6" stopOpacity={0.08} />
                <stop offset="100%" stopColor="#8b5cf6" stopOpacity={0.15} />
              </linearGradient>
            </defs>

            <CartesianGrid
              strokeDasharray="3 3"
              stroke="var(--chart-grid)"
              vertical={false}
            />
            <XAxis
              dataKey="date"
              tick={{ fontSize: 11, fill: "var(--chart-tick)" }}
              tickLine={false}
              axisLine={{ stroke: "var(--chart-axis)" }}
              interval="preserveStartEnd"
              minTickGap={40}
            />
            <YAxis
              yAxisId="price"
              domain={[minVal - padding, maxVal + padding]}
              tick={{ fontSize: 11, fill: "var(--chart-tick)" }}
              tickLine={false}
              axisLine={false}
              tickFormatter={(v) => `$${v.toFixed(0)}`}
            />
            <Tooltip content={<CustomTooltip chartMode={chartMode} />} />
            <Legend wrapperStyle={{ fontSize: 12, color: "var(--text-secondary)" }} />

            {/* Split reference line */}
            {splitDate && predictions && predictions.length > 0 && (
              <ReferenceLine
                yAxisId="price"
                x={splitDate}
                stroke="var(--border-color)"
                strokeDasharray="5 5"
                label={{
                  value: "Forecast →",
                  position: "top",
                  fill: "var(--text-muted)",
                  fontSize: 11,
                }}
              />
            )}

            {/* ── LINE MODE ────────────────────────────────────────── */}
            {chartMode === "line" && (
              <>
                <Area
                  yAxisId="price"
                  type="monotone"
                  dataKey="history"
                  fill="url(#historyGradient)"
                  stroke="none"
                  connectNulls={false}
                />
                <Line
                  yAxisId="price"
                  type="monotone"
                  dataKey="history"
                  stroke="#6366f1"
                  strokeWidth={2}
                  dot={false}
                  name="Historical"
                  connectNulls={false}
                />
              </>
            )}

            {/* ── CANDLESTICK MODE ─────────────────────────────────── */}
            {chartMode === "candlestick" && (
              <Bar
                yAxisId="price"
                dataKey="candle"
                name="OHLC"
                legendType="none"
                shape={(props) => {
                  const d = props.payload;
                  if (d.open == null || d.high == null) return null;

                  const isUp = d.close >= d.open;
                  const color = isUp ? "#10b981" : "#ef4444";
                  const { x, width } = props;
                  const bodyWidth = Math.max(width * 0.6, 3);

                  // Use the Y-axis scale from the chart
                  const yDomain = [minVal - padding, maxVal + padding];
                  const chartHeight = props.background?.height || 300;
                  const chartTop = props.background?.y || 0;
                  const range = yDomain[1] - yDomain[0];
                  const pxPerUnit = chartHeight / range;

                  const priceToY = (p) => chartTop + (yDomain[1] - p) * pxPerUnit;
                  const cx = x + width / 2;

                  return (
                    <g>
                      <line
                        x1={cx} y1={priceToY(d.high)}
                        x2={cx} y2={priceToY(d.low)}
                        stroke={color} strokeWidth={1}
                      />
                      <rect
                        x={cx - bodyWidth / 2}
                        y={priceToY(Math.max(d.open, d.close))}
                        width={bodyWidth}
                        height={Math.max(Math.abs(d.open - d.close) * pxPerUnit, 1)}
                        fill={color}
                        rx={1}
                      />
                    </g>
                  );
                }}
              >
              </Bar>
            )}

            {/* ── CONFIDENCE BAND ──────────────────────────────────── */}
            <Area
              yAxisId="price"
              type="monotone"
              dataKey="confidenceBand"
              fill="url(#confidenceGradient)"
              stroke="#8b5cf6"
              strokeWidth={1}
              strokeOpacity={0.25}
              strokeDasharray="4 2"
              connectNulls={false}
              name="95% CI"
              legendType="none"
            />

            {/* ── FORECAST LINE ────────────────────────────────────── */}
            <Area
              yAxisId="price"
              type="monotone"
              dataKey="forecast"
              fill="url(#forecastGradient)"
              stroke="none"
              connectNulls={false}
            />
            <Line
              yAxisId="price"
              type="monotone"
              dataKey="forecast"
              stroke="#06b6d4"
              strokeWidth={2}
              strokeDasharray="6 3"
              dot={{ r: 4, fill: "#06b6d4", stroke: "var(--bg-primary)", strokeWidth: 2 }}
              name="Forecast"
              connectNulls={false}
              isAnimationActive={true}
              animationDuration={800}
              animationEasing="ease-out"
            />
          </ComposedChart>
        </ResponsiveContainer>
      </div>

      {/* Volume Bar Chart */}
      <div className="chart-volume-container">
        <ResponsiveContainer width="100%" height="100%">
          <ComposedChart
            data={chartData}
            margin={{ top: 0, right: 30, left: 10, bottom: 0 }}
          >
            <CartesianGrid
              strokeDasharray="3 3"
              stroke="var(--chart-grid)"
              vertical={false}
              horizontal={false}
            />
            <XAxis dataKey="date" hide={true} />
            <YAxis
              domain={[0, maxVolume * 1.2]}
              tick={{ fontSize: 9, fill: "var(--chart-tick)" }}
              tickLine={false}
              axisLine={false}
              tickFormatter={(v) =>
                v >= 1e6 ? `${(v / 1e6).toFixed(0)}M` : `${(v / 1e3).toFixed(0)}K`
              }
              width={50}
            />
            <Bar dataKey="volume" name="Volume" legendType="none" radius={[2, 2, 0, 0]}>
              {chartData.map((entry, idx) => {
                if (!entry.volume) return <Cell key={idx} fill="transparent" />;
                const isUp = entry.close >= entry.open;
                return (
                  <Cell
                    key={idx}
                    fill={isUp ? "rgba(16, 185, 129, 0.35)" : "rgba(239, 68, 68, 0.35)"}
                  />
                );
              })}
            </Bar>
          </ComposedChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
