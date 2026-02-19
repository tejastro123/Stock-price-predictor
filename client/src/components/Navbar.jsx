/**
 * Navbar.jsx — Application Header
 * =================================
 * Premium navigation bar with branding and status indicator.
 */

import React from "react";
import { FiActivity, FiTrendingUp } from "react-icons/fi";

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
      <div className="navbar__badge">
        <FiActivity size={12} style={{ marginRight: 4 }} />
        Live System
      </div>
    </nav>
  );
}
