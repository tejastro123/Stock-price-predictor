/**
 * StockInfo.jsx — Stock Information Card
 * ========================================
 * Displays current stock metadata: price, market cap, P/E, 52-week range,
 * sector, volume, etc. Fetched from the /info endpoint.
 */

import React from "react";
import { FiInfo, FiTrendingUp, FiTrendingDown } from "react-icons/fi";

function formatNumber(num) {
  if (num == null) return "—";
  if (num >= 1e12) return `$${(num / 1e12).toFixed(2)}T`;
  if (num >= 1e9) return `$${(num / 1e9).toFixed(2)}B`;
  if (num >= 1e6) return `$${(num / 1e6).toFixed(2)}M`;
  return num.toLocaleString();
}

function formatPrice(num) {
  if (num == null) return "—";
  return `$${num.toFixed(2)}`;
}

export default function StockInfo({ info }) {
  if (!info) return null;

  const priceChange =
    info.current_price && info.previous_close
      ? info.current_price - info.previous_close
      : null;
  const priceChangePct =
    priceChange != null && info.previous_close
      ? (priceChange / info.previous_close) * 100
      : null;
  const isUp = priceChange >= 0;

  const metrics = [
    {
      label: "Current Price",
      value: formatPrice(info.current_price),
      accent: true,
      sub: priceChangePct != null
        ? `${isUp ? "+" : ""}${priceChange.toFixed(2)} (${isUp ? "+" : ""}${priceChangePct.toFixed(2)}%)`
        : null,
      subColor: isUp ? "#10b981" : "#ef4444",
    },
    { label: "Market Cap", value: info.market_cap ? formatNumber(info.market_cap) : "—" },
    { label: "P/E Ratio", value: info.pe_ratio ? info.pe_ratio.toFixed(2) : "—" },
    { label: "EPS", value: info.eps ? `$${info.eps.toFixed(2)}` : "—" },
    {
      label: "52W Range",
      value: info.fifty_two_week_low && info.fifty_two_week_high
        ? `${formatPrice(info.fifty_two_week_low)} — ${formatPrice(info.fifty_two_week_high)}`
        : "—",
    },
    {
      label: "Day Range",
      value: info.day_low && info.day_high
        ? `${formatPrice(info.day_low)} — ${formatPrice(info.day_high)}`
        : "—",
    },
    { label: "Volume", value: info.volume ? info.volume.toLocaleString() : "—" },
    { label: "Beta", value: info.beta ? info.beta.toFixed(2) : "—" },
  ];

  return (
    <div className="card fade-in stagger-1">
      <div className="card__header">
        <div className="card__icon card__icon--cyan">
          <FiInfo />
        </div>
        <h2 className="card__title">
          {info.name || info.ticker}
          <span
            style={{
              fontSize: 13,
              color: "var(--text-muted)",
              fontWeight: 400,
              marginLeft: 8,
            }}
          >
            {info.ticker}
            {info.sector && info.sector !== "N/A" && ` · ${info.sector}`}
          </span>
        </h2>
      </div>

      <div className="stock-info-grid">
        {metrics.map((m) => (
          <div key={m.label} className="si-metric">
            <div className="si-metric__label">{m.label}</div>
            <div
              className="si-metric__value"
              style={m.accent ? { fontSize: 20 } : {}}
            >
              {m.value}
              {m.accent && priceChange != null && (
                <span style={{ marginLeft: 6, fontSize: 14 }}>
                  {isUp ? (
                    <FiTrendingUp color="#10b981" size={16} />
                  ) : (
                    <FiTrendingDown color="#ef4444" size={16} />
                  )}
                </span>
              )}
            </div>
            {m.sub && (
              <div
                className="si-metric__sub"
                style={{ color: m.subColor }}
              >
                {m.sub}
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}
