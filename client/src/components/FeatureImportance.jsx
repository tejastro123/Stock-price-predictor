/**
 * FeatureImportance.jsx — Feature Importance Chart
 * ==================================================
 * Displays a horizontal bar chart showing which features matter most
 * to the trained model, using pure CSS bars.
 */

import React from "react";
import { FiLayers } from "react-icons/fi";

export default function FeatureImportance({ features, modelName }) {
  if (!features || features.length === 0) return null;

  // Normalize importances for bar widths (relative to max)
  const maxImportance = Math.max(...features.map((f) => f.importance));

  return (
    <div className="card fade-in stagger-2">
      <div className="card__header">
        <div className="card__icon card__icon--purple">
          <FiLayers />
        </div>
        <h2 className="card__title">
          Feature Importance
          <span
            style={{
              fontSize: 13,
              color: "var(--text-muted)",
              fontWeight: 400,
              marginLeft: 8,
            }}
          >
            {modelName} — top {features.length}
          </span>
        </h2>
      </div>

      <div className="feature-importance">
        {features.map((f, idx) => {
          const pct = (f.importance / maxImportance) * 100;
          return (
            <div key={f.feature} className="fi-row">
              <div className="fi-row__rank">{idx + 1}</div>
              <div className="fi-row__name">{f.feature}</div>
              <div className="fi-row__bar-track">
                <div
                  className="fi-row__bar-fill"
                  style={{
                    width: `${pct}%`,
                    animationDelay: `${idx * 50}ms`,
                  }}
                />
              </div>
              <div className="fi-row__value">
                {(f.importance * 100).toFixed(1)}%
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
