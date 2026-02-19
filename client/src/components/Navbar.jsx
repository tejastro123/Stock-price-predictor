/**
 * Navbar.jsx — Application Header
 * =================================
 * Premium navigation bar with branding, status indicator, and theme toggle.
 */

import React from "react";
import { FiActivity, FiTrendingUp } from "react-icons/fi";
import ThemeToggle from "./ThemeToggle";

export default function Navbar() {
  return (
    <nav className="navbar">
      <div className="navbar__brand">
        <div className="navbar__logo">
          <FiTrendingUp />
        </div>
        <div>
          <div className="navbar__title">StockSage AI</div>
          <div className="navbar__subtitle">ML-Powered Stock Price Prediction</div>
        </div>
      </div>
      <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
        <a
          href="http://localhost:8000/docs"
          target="_blank"
          rel="noopener noreferrer"
          className="btn btn--sm btn--secondary"
          style={{ textDecoration: "none", display: "flex", alignItems: "center", gap: 6 }}
        >
          <FiActivity /> API Docs
        </a>
        <ThemeToggle />
        <div className="navbar__badge">
          <FiTrendingUp size={12} style={{ marginRight: 4 }} />
          Live System
        </div>
      </div>
    </nav>
  );
}
